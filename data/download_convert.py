"""M1: download + convert all datasets to the unified JSONL schema (PLAN §3.1).

Outputs under data/processed/:
  math_train.jsonl     (~7.5k)  train pool, from local lighteval-MATH parquet (M0 download)
  math500.jsonl        (500)    main eval
  aime24.jsonl         (30)     low-contamination eval
  aime25.jsonl         (30)     low-contamination eval
  gsm8k_test200.jsonl  (200)    sanity-check eval

Unified schema: {id, question, answer, level, subject, source, split}
(level/subject are None where the source dataset has no such field).
"""

import json
import os
import re
from typing import Any, Iterable

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
MATH_PARQUET = os.path.expanduser("~/verl-team/lighteval-MATH-preprocessed/train.parquet")
# Fixed instruction appended by verl-team preprocessing; verified to match all 12500 rows.
MATH_PROMPT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def make_row(
    id_: str, question: str, answer: str, level: int | None, subject: str | None, source: str, split: str
) -> dict[str, Any]:
    assert question.strip() and str(answer).strip(), f"empty question/answer in {id_}"
    return {
        "id": id_,
        "question": question.strip(),
        "answer": str(answer).strip(),
        "level": level,
        "subject": subject,
        "source": source,
        "split": split,
    }


def parse_math_level(level_str: str) -> int | None:
    m = re.fullmatch(r"Level (\d)", level_str)
    return int(m.group(1)) if m else None


def convert_math_train() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(MATH_PARQUET)
    rows: list[dict[str, Any]] = []
    n_dropped = 0
    for i in range(table.num_rows):
        prompt = table["prompt"][i].as_py()[0]["content"]
        assert prompt.endswith(MATH_PROMPT_SUFFIX), f"unexpected prompt format at row {i}"
        level = parse_math_level(table["level"][i].as_py())
        answer = table["reward_model"][i].as_py()["ground_truth"]
        if level is None or not str(answer).strip():  # 2 rows 'Level ?', 2 rows empty ground_truth
            n_dropped += 1
            continue
        rows.append(
            make_row(
                id_=f"math_train_{i:06d}",
                question=prompt.removesuffix(MATH_PROMPT_SUFFIX),
                answer=answer,
                level=level,
                subject=table["type"][i].as_py(),
                source="MATH",
                split="train",
            )
        )
    print(f"math_train: kept {len(rows)}, dropped {n_dropped} (unknown level or empty answer)")
    return rows


def convert_math500() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        make_row(
            id_=f"math500_{i:06d}",
            question=ex["problem"],
            answer=ex["answer"],
            level=int(ex["level"]),
            subject=ex["subject"],
            source="MATH500",
            split="test",
        )
        for i, ex in enumerate(ds)
    ]


def convert_aime24() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [
        make_row(
            id_=f"aime24_{i:06d}",
            question=ex["Problem"],
            answer=str(ex["Answer"]),
            level=None,
            subject=None,
            source="AIME2024",
            split="test",
        )
        for i, ex in enumerate(ds)
    ]


def convert_aime25() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("math-ai/aime25", split="test")
    return [
        make_row(
            id_=f"aime25_{i:06d}",
            question=ex["problem"],
            answer=str(ex["answer"]),
            level=None,
            subject=None,
            source="AIME2025",
            split="test",
        )
        for i, ex in enumerate(ds)
    ]


def convert_gsm8k_200() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, ex in enumerate(ds):
        if i >= 200:
            break
        answer = ex["answer"].split("####")[-1].strip().replace(",", "")
        rows.append(
            make_row(
                id_=f"gsm8k_{i:06d}",
                question=ex["question"],
                answer=answer,
                level=None,
                subject=None,
                source="GSM8K",
                split="test",
            )
        )
    return rows


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    converters = {
        "math_train.jsonl": convert_math_train,
        "math500.jsonl": convert_math500,
        "aime24.jsonl": convert_aime24,
        "aime25.jsonl": convert_aime25,
        "gsm8k_test200.jsonl": convert_gsm8k_200,
    }
    for filename, fn in converters.items():
        path = os.path.join(PROCESSED_DIR, filename)
        n = write_jsonl(path, fn())
        print(f"wrote {path}: {n} rows")


if __name__ == "__main__":
    main()
