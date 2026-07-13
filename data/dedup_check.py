"""M1: train/eval contamination check (PLAN §5.1).

Checks the training pool against every eval set with:
  1. exact match on normalized question text;
  2. word-level 13-gram overlap (GPT-3-style: lowercase, strip punctuation).

Usage:
  python data/dedup_check.py                          # math_train.jsonl -> math_train_clean.jsonl + report
  python data/dedup_check.py --train data/processed/train_subset.jsonl --no-write-clean
  python data/dedup_check.py --self-test
"""

import argparse
import json
import os
import re
from typing import Any

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EVAL_FILES = ["math500.jsonl", "aime24.jsonl", "aime25.jsonl", "gsm8k_test200.jsonl"]
NGRAM_N = 13


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize(text: str) -> str:
    text = re.sub(r"[^0-9a-z]+", " ", text.lower())
    return " ".join(text.split())


def ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def find_hits(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return one hit record per contaminated train row (first eval match wins)."""
    exact_index: dict[str, str] = {normalize(r["question"]): r["id"] for r in eval_rows}
    gram_index: dict[tuple[str, ...], str] = {}
    for r in eval_rows:
        for g in ngrams(r["question"]):
            gram_index.setdefault(g, r["id"])

    hits = []
    for r in train_rows:
        norm = normalize(r["question"])
        if norm in exact_index:
            hits.append({"train_id": r["id"], "eval_id": exact_index[norm], "match": "exact"})
            continue
        overlap = ngrams(r["question"]) & gram_index.keys()
        if overlap:
            g = next(iter(overlap))
            hits.append({"train_id": r["id"], "eval_id": gram_index[g], "match": f"{NGRAM_N}-gram: {' '.join(g)}"})
    return hits


def run_check(train_path: str, eval_paths: list[str], report_path: str, clean_path: str | None) -> int:
    train_rows = read_jsonl(train_path)
    lines = [
        "# 污染检查报告",
        "",
        f"训练池: `{train_path}`（{len(train_rows)} 条）｜方法: 归一化精确匹配 + 词级 {NGRAM_N}-gram 重叠",
        "",
        "| 评测集 | 条数 | 命中数 |",
        "|---|---|---|",
    ]
    all_hits: list[dict[str, str]] = []
    for ep in eval_paths:
        eval_rows = read_jsonl(ep)
        hits = find_hits(train_rows, eval_rows)
        all_hits.extend(hits)
        lines.append(f"| {os.path.basename(ep)} | {len(eval_rows)} | {len(hits)} |")

    contaminated_ids = {h["train_id"] for h in all_hits}
    n_exact = sum(h["match"] == "exact" for h in all_hits)
    lines += [
        "",
        f"**共剔除训练池条目: {len(contaminated_ids)}**（精确匹配 {n_exact}，其余为 {NGRAM_N}-gram 命中）",
        "",
        f"注：{NGRAM_N}-gram 命中大多是答案格式模板句（如 'where m and n are relatively prime"
        " positive integers find m n'）而非题目本身重复——按 PLAN §5.1 保守处理，命中即剔除。",
        "",
    ]
    if all_hits:
        lines += ["| train_id | eval_id | 匹配方式 |", "|---|---|---|"]
        lines += [f"| {h['train_id']} | {h['eval_id']} | {h['match']} |" for h in all_hits]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"report -> {report_path}")

    if clean_path is not None:
        clean = [r for r in train_rows if r["id"] not in contaminated_ids]
        with open(clean_path, "w", encoding="utf-8") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"clean train pool -> {clean_path}: {len(clean)} rows")
    return len(contaminated_ids)


def self_test() -> None:
    eval_rows = read_jsonl(os.path.join(PROCESSED_DIR, "math500.jsonl"))
    long_q = next(r for r in eval_rows if len(normalize(r["question"]).split()) >= NGRAM_N + 5)
    fake_train = [
        {"id": "fake_exact", "question": long_q["question"]},
        {"id": "fake_ngram", "question": "Unrelated preamble sentence here. " + long_q["question"] + " Extra tail."},
        {"id": "fake_clean", "question": "What is one plus one? Nothing in common with MATH500 at all."},
    ]
    hits = find_hits(fake_train, eval_rows)
    hit_ids = {h["train_id"]: h["match"] for h in hits}
    assert hit_ids.get("fake_exact") == "exact", hits
    assert "fake_ngram" in hit_ids and hit_ids["fake_ngram"].startswith(f"{NGRAM_N}-gram"), hits
    assert "fake_clean" not in hit_ids, hits
    print("self-test passed: exact + n-gram detected, clean row untouched")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(PROCESSED_DIR, "math_train.jsonl"))
    parser.add_argument("--report", default=os.path.join(DATA_DIR, "contamination_report.md"))
    parser.add_argument("--no-write-clean", action="store_true", help="check only, don't write cleaned pool")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    clean_path = None if args.no_write_clean else args.train.replace(".jsonl", "_clean.jsonl")
    assert clean_path != args.train, "train path must end in .jsonl"
    run_check(args.train, [os.path.join(PROCESSED_DIR, f) for f in EVAL_FILES], args.report, clean_path)


if __name__ == "__main__":
    main()
