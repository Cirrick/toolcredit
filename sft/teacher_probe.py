"""Fixed-subset stronger-teacher probe for the M3 follow-up experiment.

The probe deliberately reuses M3's generation loop and five rejection filters.
It compares the existing Qwen3-8B raw trajectories with a candidate teacher on
the exact same 200 questions (L3/L4/L5 = 50/50/100), two samples per question.

Usage:
  python sft/teacher_probe.py prepare
  python sft/teacher_probe.py generate --port 30000
  python sft/teacher_probe.py analyze
"""

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from rewards.verifier import verify_answer  # noqa: E402
from sft.gen_trajectories import (  # noqa: E402
    MAX_RESPONSE_TOKENS,
    STUDENT_TOKENIZER_PATH,
    generate,
    response_token_len,
)

POOL_PATH = os.path.join(PROJECT_DIR, "sft", "data", "sft_pool.jsonl")
BASELINE_RAW_PATH = os.path.join(PROJECT_DIR, "sft", "data", "gen", "raw_traces.jsonl")
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "sft", "experiments", "m3_minimal", "teacher_probe")
LEVEL_QUOTAS = {3: 50, 4: 50, 5: 100}
SEED = 20260819
EXPECTED_SAMPLES = 2

# Fixed before seeing candidate results: both yield and hard-question coverage
# must move materially, while reliability may not regress.
DECISION_THRESHOLDS = {
    "min_yield_delta": 0.05,
    "min_l5_coverage_delta": 0.10,
    "max_tool_error_delta": 0.02,
    "max_malformed_call_delta": 0.01,
}


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(out_dir: str) -> None:
    """Freeze the question set and extract its already-generated 8B baseline."""
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(POOL_PATH):
        if row["level"] in LEVEL_QUOTAS:
            by_level[row["level"]].append(row)

    rng = random.Random(SEED)
    subset: list[dict[str, Any]] = []
    for level, quota in LEVEL_QUOTAS.items():
        rows = sorted(by_level[level], key=lambda row: row["id"])
        if len(rows) < quota:
            raise ValueError(f"level {level}: requested {quota}, found {len(rows)}")
        subset.extend(rng.sample(rows, quota))
    subset.sort(key=lambda row: (row["level"], row["id"]))
    subset_path = os.path.join(out_dir, "subset.jsonl")
    write_jsonl(subset_path, subset)

    wanted_ids = {row["id"] for row in subset}
    baseline = [row for row in read_jsonl(BASELINE_RAW_PATH) if row["id"] in wanted_ids]
    validate_raw(baseline, wanted_ids, "Qwen3-8B")
    baseline.sort(key=lambda row: (row["level"], row["id"], row["sample_idx"]))
    write_jsonl(os.path.join(out_dir, "qwen3_8b", "raw_traces.jsonl"), baseline)

    manifest = {
        "seed": SEED,
        "pool": os.path.relpath(POOL_PATH, PROJECT_DIR),
        "baseline_raw": os.path.relpath(BASELINE_RAW_PATH, PROJECT_DIR),
        "level_quotas": {str(k): v for k, v in LEVEL_QUOTAS.items()},
        "n_questions": len(subset),
        "n_samples": EXPECTED_SAMPLES,
        "decision_thresholds": DECISION_THRESHOLDS,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def validate_raw(rows: list[dict[str, Any]], expected_ids: set[str], label: str) -> None:
    observed = {(row["id"], row["sample_idx"]) for row in rows}
    expected = {(question_id, sample_idx) for question_id in expected_ids for sample_idx in range(EXPECTED_SAMPLES)}
    if observed != expected or len(rows) != len(expected):
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(
            f"{label}: expected {len(expected)} unique trajectories, found {len(rows)}; "
            f"missing={missing}, extra={extra}"
        )


def malformed_tool_call(tool_call: dict[str, Any]) -> bool:
    try:
        arguments = tool_call["function"]["arguments"]
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (KeyError, TypeError, json.JSONDecodeError):
        return True
    return not isinstance(parsed, dict) or set(parsed) != {"code"}


def rejection_reason(row: dict[str, Any], tokenizer: Any) -> tuple[str | None, int | None]:
    if row["truncated"]:
        return "truncated", None
    if row["n_tool_calls"] - row["n_tool_errors"] < 1:
        return "no_successful_tool_call", None
    calls = [call for message in row["messages"] for call in (message.get("tool_calls") or [])]
    if any(malformed_tool_call(call) for call in calls):
        return "bad_tool_args", None
    final = row["messages"][-1]["content"] if row["messages"] else ""
    verdict = verify_answer(final, row["gold"], strict_boxed=True)
    if verdict["invalid"]:
        return "verify_invalid", None
    if not verdict["correct"]:
        return "wrong_answer", None
    n_tokens = response_token_len(tokenizer, row["messages"])
    if n_tokens > MAX_RESPONSE_TOKENS:
        return "too_long", n_tokens
    return None, n_tokens


def analyze_one(
    raw_path: str, subset: list[dict[str, Any]], teacher_model: str, tokenizer: Any
) -> dict[str, Any]:
    rows = read_jsonl(raw_path)
    expected_ids = {row["id"] for row in subset}
    validate_raw(rows, expected_ids, teacher_model)

    question_levels = {row["id"]: row["level"] for row in subset}
    kept_ids: dict[int, set[str]] = defaultdict(set)
    level_raw: Counter[int] = Counter()
    level_kept: Counter[int] = Counter()
    reasons: Counter[str] = Counter()
    kept_tokens: list[int] = []
    total_calls = sum(row["n_tool_calls"] for row in rows)
    total_errors = sum(row["n_tool_errors"] for row in rows)
    all_calls = [call for row in rows for message in row["messages"] for call in (message.get("tool_calls") or [])]
    malformed_calls = sum(malformed_tool_call(call) for call in all_calls)
    malformed_trajectories = sum(
        any(malformed_tool_call(call) for message in row["messages"] for call in (message.get("tool_calls") or []))
        for row in rows
    )

    for row in rows:
        level = row["level"]
        level_raw[level] += 1
        reason, n_tokens = rejection_reason(row, tokenizer)
        if reason is not None:
            reasons[reason] += 1
            continue
        level_kept[level] += 1
        kept_ids[level].add(row["id"])
        if n_tokens is not None:
            kept_tokens.append(n_tokens)

    level_questions = Counter(question_levels.values())
    kept = sum(level_kept.values())
    covered_ids = set().union(*kept_ids.values()) if kept_ids else set()
    stats: dict[str, Any] = {
        "teacher": teacher_model,
        "raw": len(rows),
        "kept": kept,
        "effective_yield": round(kept / len(rows), 4),
        "question_coverage": round(len(covered_ids) / len(subset), 4),
        "questions_covered": len(covered_ids),
        "tool_error_rate": round(total_errors / max(1, total_calls), 4),
        "malformed_tool_call_rate": round(malformed_calls / max(1, len(all_calls)), 4),
        "malformed_trajectory_rate": round(malformed_trajectories / len(rows), 4),
        "filter_reasons": dict(reasons),
        "avg_response_tokens_kept": round(sum(kept_tokens) / max(1, len(kept_tokens)), 1),
        "levels": {},
    }
    for level in sorted(LEVEL_QUOTAS):
        stats["levels"][str(level)] = {
            "questions": level_questions[level],
            "raw": level_raw[level],
            "kept": level_kept[level],
            "effective_yield": round(level_kept[level] / max(1, level_raw[level]), 4),
            "questions_covered": len(kept_ids[level]),
            "question_coverage": round(len(kept_ids[level]) / max(1, level_questions[level]), 4),
        }
    return stats


def analyze(out_dir: str, candidate_label: str) -> None:
    from transformers import AutoTokenizer

    subset = read_jsonl(os.path.join(out_dir, "subset.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_TOKENIZER_PATH)
    baseline = analyze_one(
        os.path.join(out_dir, "qwen3_8b", "raw_traces.jsonl"), subset, "Qwen3-8B", tokenizer
    )
    candidate_dir = os.path.join(out_dir, "qwen3_30b_a3b_instruct_2507")
    candidate = analyze_one(
        os.path.join(candidate_dir, "raw_traces.jsonl"), subset, candidate_label, tokenizer
    )
    for path, stats in [
        (os.path.join(out_dir, "qwen3_8b", "stats.json"), baseline),
        (os.path.join(candidate_dir, "stats.json"), candidate),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    deltas = {
        "effective_yield": round(candidate["effective_yield"] - baseline["effective_yield"], 4),
        "question_coverage": round(candidate["question_coverage"] - baseline["question_coverage"], 4),
        "l5_question_coverage": round(
            candidate["levels"]["5"]["question_coverage"] - baseline["levels"]["5"]["question_coverage"], 4
        ),
        "tool_error_rate": round(candidate["tool_error_rate"] - baseline["tool_error_rate"], 4),
        "malformed_tool_call_rate": round(
            candidate["malformed_tool_call_rate"] - baseline["malformed_tool_call_rate"], 4
        ),
    }
    checks = {
        "yield": deltas["effective_yield"] >= DECISION_THRESHOLDS["min_yield_delta"],
        "l5_coverage": deltas["l5_question_coverage"] >= DECISION_THRESHOLDS["min_l5_coverage_delta"],
        "tool_errors": deltas["tool_error_rate"] <= DECISION_THRESHOLDS["max_tool_error_delta"],
        "malformed_calls": deltas["malformed_tool_call_rate"] <= DECISION_THRESHOLDS["max_malformed_call_delta"],
    }
    comparison = {
        "baseline": baseline,
        "candidate": candidate,
        "deltas_candidate_minus_baseline": deltas,
        "decision_thresholds": DECISION_THRESHOLDS,
        "decision_checks": checks,
        "worth_switching_teacher": all(checks.values()),
    }
    with open(os.path.join(out_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


def generate_candidate(out_dir: str, port: int, concurrency: int) -> None:
    subset_path = os.path.join(out_dir, "subset.jsonl")
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"prepare the fixed subset first: {subset_path}")
    questions = read_jsonl(subset_path)
    args = SimpleNamespace(
        port=port,
        out_dir=os.path.join(out_dir, "qwen3_30b_a3b_instruct_2507"),
        concurrency=concurrency,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    asyncio.run(generate(args, questions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "generate", "analyze"])
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--candidate-label", default="Qwen3-30B-A3B-Instruct-2507")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.command == "prepare":
        prepare(args.out_dir)
    elif args.command == "generate":
        generate_candidate(args.out_dir, args.port, args.concurrency)
    else:
        analyze(args.out_dir, args.candidate_label)


if __name__ == "__main__":
    main()
