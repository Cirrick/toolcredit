"""Build the fixed veRL parquet inputs for M4 / E3.

The training split is exactly the M3 SFT pool (the L3--L5 training subset with
the 200-question SFT held-out set removed).  Validation is a deterministic,
level-stratified 100-question subset of MATH500.  The source JSONL files remain
the canonical data; parquet is only the veRL adapter format used for training.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SUBSET = PROJECT_ROOT / "data/processed/train_subset.jsonl"
HELDOUT = PROJECT_ROOT / "sft/data/heldout_200.jsonl"
SFT_POOL = PROJECT_ROOT / "sft/data/sft_pool.jsonl"
MATH500 = PROJECT_ROOT / "data/processed/math500.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "rl/data"
TRAIN_PARQUET = OUTPUT_DIR / "e3_train.parquet"
VAL_PARQUET = OUTPUT_DIR / "e3_val_math500_100.parquet"
SMOKE_TRAIN_PARQUET = OUTPUT_DIR / "e3_smoke_train_20.parquet"
SMOKE_VAL_PARQUET = OUTPUT_DIR / "e3_smoke_val_8.parquet"
MANIFEST = OUTPUT_DIR / "manifest.json"
MODEL_PATH = PROJECT_ROOT / "sft/checkpoints/qwen3-1.7b-sft"
TOOL_CONFIG = PROJECT_ROOT / "rl/custom/tool_config.yaml"
PROMPT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."
VAL_SIZE = 100
SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _assert_unique_ids(rows: Iterable[dict[str, Any]], label: str) -> set[str]:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in {label}")
    return set(ids)


def validate_split(
    train_subset: list[dict[str, Any]], heldout: list[dict[str, Any]], train_rows: list[dict[str, Any]]
) -> None:
    subset_ids = _assert_unique_ids(train_subset, "train_subset")
    heldout_ids = _assert_unique_ids(heldout, "heldout")
    train_ids = _assert_unique_ids(train_rows, "M4 train")
    if heldout_ids & train_ids:
        raise ValueError("M3 held-out questions leaked into the M4 training split")
    expected = subset_ids - heldout_ids
    if train_ids != expected:
        missing = sorted(expected - train_ids)[:5]
        unexpected = sorted(train_ids - expected)[:5]
        raise ValueError(f"M4 train split mismatch: missing={missing}, unexpected={unexpected}")
    if len(train_rows) != 5203:
        raise ValueError(f"M4 train split must contain 5203 rows, got {len(train_rows)}")


def select_validation(rows: list[dict[str, Any]], size: int = VAL_SIZE, seed: int = SEED) -> list[dict[str, Any]]:
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[int(row["level"])].append(row)
    if set(by_level) != {1, 2, 3, 4, 5}:
        raise ValueError(f"unexpected MATH500 levels: {sorted(by_level)}")
    if size % len(by_level) != 0:
        raise ValueError("validation size must be divisible by the number of levels")

    per_level = size // len(by_level)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for level in sorted(by_level):
        pool = sorted(by_level[level], key=lambda row: row["id"])
        if len(pool) < per_level:
            raise ValueError(f"not enough level-{level} validation rows")
        selected.extend(rng.sample(pool, per_level))
    return sorted(selected, key=lambda row: row["id"])


def to_verl_row(row: dict[str, Any], data_source: str) -> dict[str, Any]:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": row["question"] + PROMPT_SUFFIX}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": row["answer"]},
        "extra_info": {
            "id": row["id"],
            "level": row["level"],
            "subject": row["subject"],
            "source": row["source"],
            "split": row["split"],
        },
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    roundtrip = pq.read_table(path).to_pylist()
    if roundtrip != rows:
        raise ValueError(f"parquet round-trip changed rows in {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_length_stats(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.tools.schemas import OpenAIFunctionToolSchema

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    raw_schema = OmegaConf.to_container(OmegaConf.load(TOOL_CONFIG).tools[0].tool_schema, resolve=True)
    schema = OpenAIFunctionToolSchema.model_validate(raw_schema).model_dump(exclude_none=True)

    def lengths(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
        return [
            (
                str(row["extra_info"]["id"]),
                len(
                    tokenizer.apply_chat_template(
                        row["prompt"],
                        tools=[schema],
                        add_generation_prompt=True,
                        tokenize=True,
                        enable_thinking=False,
                    )
                ),
            )
            for row in rows
        ]

    train_lengths = lengths(train_rows)
    val_lengths = lengths(val_rows)
    overlong_train = [{"id": id_, "tokens": length} for id_, length in train_lengths if length > 1024]
    overlong_val = [{"id": id_, "tokens": length} for id_, length in val_lengths if length > 1024]
    return {
        "max_prompt_tokens": 1024,
        "train_max": max(length for _, length in train_lengths),
        "validation_max": max(length for _, length in val_lengths),
        "overlong_train": overlong_train,
        "overlong_validation": overlong_val,
        "effective_train_after_verl_filter": len(train_rows) - len(overlong_train),
    }


def build() -> dict[str, Any]:
    train_subset = read_jsonl(TRAIN_SUBSET)
    heldout = read_jsonl(HELDOUT)
    train_rows = read_jsonl(SFT_POOL)
    math500 = read_jsonl(MATH500)
    validate_split(train_subset, heldout, train_rows)

    val_rows = select_validation(math500)
    train_verl = [to_verl_row(row, "toolcredit_math") for row in train_rows]
    val_verl = [to_verl_row(row, "toolcredit_math500") for row in val_rows]
    write_parquet(TRAIN_PARQUET, train_verl)
    write_parquet(VAL_PARQUET, val_verl)
    write_parquet(SMOKE_TRAIN_PARQUET, train_verl[:20])
    write_parquet(SMOKE_VAL_PARQUET, val_verl[:8])
    length_stats = prompt_length_stats(train_verl, val_verl)

    outputs = [TRAIN_PARQUET, VAL_PARQUET, SMOKE_TRAIN_PARQUET, SMOKE_VAL_PARQUET]
    manifest = {
        "seed": SEED,
        "prompt_suffix": PROMPT_SUFFIX,
        "train_source": str(SFT_POOL.relative_to(PROJECT_ROOT)),
        "heldout_source": str(HELDOUT.relative_to(PROJECT_ROOT)),
        "validation_source": str(MATH500.relative_to(PROJECT_ROOT)),
        "counts": {
            "train": len(train_verl),
            "validation": len(val_verl),
            "smoke_train": 20,
            "smoke_validation": 8,
        },
        "train_levels": dict(sorted(Counter(row["level"] for row in train_rows).items())),
        "validation_levels": dict(sorted(Counter(row["level"] for row in val_rows).items())),
        "heldout_overlap": 0,
        "prompt_lengths": length_stats,
        "files": {
            str(path.relative_to(PROJECT_ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
