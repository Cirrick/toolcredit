"""Strict M7 scoring and canonical completeness validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from rewards.verifier import verify_answer
from rl.custom.reward import compute_score

from eval.generate import (
    ROLE_NAMES,
    canonical_key,
    expected_keys,
    load_panel_questions,
    validate_raw_generation_row,
)


def score_trajectory(raw: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    """Enrich one valid raw trajectory without overwriting its generation evidence."""
    validate_raw_generation_row(raw, question)
    if raw.get("infrastructure_status") != "ok":
        raise ValueError("infrastructure failures must be retried, not scored as wrong")
    tool_metadata = raw.get("tool_metadata")
    if not isinstance(tool_metadata, dict):
        raise TypeError("tool_metadata must be a mapping")
    score = compute_score(
        data_source="toolcredit_math",
        solution_str=str(raw["response"]),
        ground_truth=str(question["answer"]),
        extra_info={"id": raw["question_id"], **tool_metadata},
    )
    strict = verify_answer(str(raw["response"]), str(question["answer"]), strict_boxed=True)
    if bool(score["acc"]) != bool(strict["correct"]):
        raise AssertionError("project reward and strict verifier correctness diverged")
    enriched = dict(raw)
    enriched.update(
        {
            "reward": score,
            "verifier_outcome": strict,
            "correct": bool(strict["correct"]),
            "format_ok": bool(score["format_ok"]),
            "invalid": bool(score["invalid"]),
            "truncated": bool(score["truncated"]) or bool(tool_metadata.get("turn_credit_ledger", {}).get("trajectory_truncated", False)),
            "budget_exhausted": bool(float(score["n_tool_calls"]) >= 4 and not strict["correct"]),
            "primary_failure": None,
            "secondary_labels": [],
            "label_source": "pending_frozen_taxonomy_v1",
            "adjudication_rationale": None,
            "raw_evidence_pointer": {
                "canonical_key": list(canonical_key(raw)),
                "raw_output_sha256": raw["raw_output_sha256"],
            },
        }
    )
    return enriched


def validate_scored_row(row: dict[str, Any]) -> None:
    required = {
        "reward", "verifier_outcome", "correct", "format_ok", "invalid", "truncated",
        "budget_exhausted", "primary_failure", "secondary_labels", "label_source",
        "adjudication_rationale", "raw_evidence_pointer",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"scored trajectory is missing fields: {missing}")
    if row.get("infrastructure_status") != "ok":
        raise ValueError("canonical scored data cannot contain infrastructure failures")
    if bool(row["correct"]) != bool(row["verifier_outcome"]["correct"]):
        raise ValueError("scored correctness does not match verifier outcome")
    if bool(row["invalid"]) != bool(row["reward"]["invalid"]):
        raise ValueError("scored invalid flag does not match reward audit")


def completeness_report(rows: Iterable[dict[str, Any]], require_all_roles: bool = True) -> dict[str, Any]:
    observed: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        validate_scored_row(row)
        key = canonical_key(row)
        if key in observed:
            raise ValueError(f"duplicate canonical scored key: {key}")
        observed[key] = row
        group = f"{row['checkpoint_role']}:{row['evaluation_setting']}:{row['source']}"
        counts[group] = counts.get(group, 0) + 1
    roles = ROLE_NAMES if require_all_roles else tuple(sorted({key[0] for key in observed}))
    expected = set(expected_keys(roles=roles))
    actual = set(observed)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    report = {
        "expected": len(expected),
        "observed": len(actual),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "counts": counts,
        "complete": not missing and not unexpected,
        "exact_key_digest": hashlib.sha256(
            json.dumps(sorted(actual), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    if missing or unexpected:
        raise ValueError(
            f"M7 canonical key universe incomplete: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"partial JSONL input at {path}:{line_number}")
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completeness-only", action="store_true")
    args = parser.parse_args()
    rows = _read_jsonl(args.input)
    if args.completeness_only:
        print(json.dumps(completeness_report(rows), indent=2, sort_keys=True))
        return
    questions = {row["question_id"]: row for row in load_panel_questions()}
    scored = [score_trajectory(row, questions[row["question_id"]]) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
