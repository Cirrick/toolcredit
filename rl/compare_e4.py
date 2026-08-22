"""Compute the preregistered E3/E4-A/E4-B comparisons with one metric protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rl.analyze_e4 import _tool_result_used, code_hash, extract_codes, is_trivial_code

THRESHOLDS = (0.67, 0.70, 0.73)


def _prediction_paths(run_dir: Path, split: str) -> list[Path]:
    return sorted(
        (run_dir / f"predictions/{split}").glob("*.jsonl"), key=lambda path: int(path.stem)
    )


def validation_curve(run_dir: Path) -> dict[int, float]:
    curve: dict[int, float] = {}
    for path in _prediction_paths(run_dir, "validation"):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if len(rows) != 100:
            raise ValueError(f"fixed panel must have 100 rows at {path}")
        curve[int(path.stem)] = sum(float(row["acc"]) for row in rows) / len(rows)
    return curve


def early_metrics(curve: dict[int, float]) -> dict[str, Any]:
    points = [(step, curve[step]) for step in sorted(curve) if step <= 100]
    required = {0, 25, 50, 75, 100}
    if {step for step, _ in points} != required:
        raise ValueError(f"early AUC requires validation steps {sorted(required)}")
    auc = sum((right_step - left_step) * (left_value + right_value) / 2 for (left_step, left_value), (right_step, right_value) in zip(points, points[1:]))
    first_reached = {
        f"{threshold:.2f}": next(
            (step for step, value in points if value >= threshold), None
        )
        for threshold in THRESHOLDS
    }
    return {
        "auc_0_100": auc,
        "auc_0_100_normalized": auc / 100.0,
        "first_validation_step_reaching": first_reached,
    }


def final_metrics(curve: dict[int, float]) -> dict[str, Any]:
    if 200 not in curve:
        raise ValueError("final comparison requires step-200 validation")
    peak_step, peak_value = max(curve.items(), key=lambda item: (item[1], -item[0]))
    return {"final_step_200": curve[200], "peak": peak_value, "peak_step": peak_step}


def behavior_metrics(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in _prediction_paths(run_dir, "train"):
        rows.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    if not rows:
        raise ValueError(f"no training rows in {run_dir}")
    calls = [int(float(row["n_tool_calls"])) for row in rows]
    total_tool_calls = sum(calls)
    estimated_tool_successes = sum(
        call_count * (1.0 - float(row["tool_error_rate"]))
        for call_count, row in zip(calls, rows)
    )
    call_counts = Counter(min(value, 4) for value in calls)
    repeated = trivial = unused = 0
    for row in rows:
        codes = extract_codes(str(row.get("output", "")))
        hashes = [code_hash(code) for code in codes]
        repeated += int(len(set(hashes)) < len(hashes))
        trivial += int(any(is_trivial_code(code) for code in codes))
        if int(float(row["n_tool_calls"])) > 0 and not _tool_result_used(str(row.get("output", ""))):
            unused += 1
    count = len(rows)
    return {
        "rows": count,
        "mean_tool_calls": sum(calls) / count,
        "total_tool_calls": total_tool_calls,
        "estimated_tool_successes": estimated_tool_successes,
        "mean_tool_successes": estimated_tool_successes / count,
        "tool_success_rate": (
            estimated_tool_successes / total_tool_calls if total_tool_calls else 0.0
        ),
        "call_fractions_0_1_2_3_4plus": {
            str(value): call_counts[value] / count for value in range(5)
        },
        "tool_error_rate": sum(float(row["tool_error_rate"]) for row in rows) / count,
        "parser_error_rate": sum(float(row.get("tool_parse_errors", 0.0)) > 0 for row in rows) / count,
        "invalid_rate": sum(float(row["invalid"]) for row in rows) / count,
        "truncated_rate": sum(float(row["truncated"]) for row in rows) / count,
        "repeated_code_trajectory_rate": repeated / count,
        "trivial_code_trajectory_rate": trivial / count,
        "unused_tool_result_candidate_rate": unused / count,
        "mean_base_score": (
            sum(float(row.get("base_score", row["score"])) for row in rows) / count
        ),
        "mean_shaped_score": sum(float(row["score"]) for row in rows) / count,
        "mean_exec_bonus": sum(float(row.get("exec_bonus", 0.0)) for row in rows) / count,
        "mean_budget_penalty": (
            sum(float(row.get("budget_penalty", 0.0)) for row in rows) / count
        ),
    }


def compare(runs: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    expected = {"e3", "e4a", "e4b"}
    if set(runs) != expected:
        raise ValueError(f"comparison requires exactly {sorted(expected)}")
    run_metrics: dict[str, Any] = {}
    for name, run_dir in runs.items():
        curve = validation_curve(run_dir)
        run_metrics[name] = {
            "run_name": run_dir.name,
            "validation_curve": {str(step): value for step, value in curve.items()},
            "early": early_metrics(curve),
            "final": final_metrics(curve),
            "behavior": behavior_metrics(run_dir),
        }
    pairs = {
        "e3_vs_e4a": ("e3", "e4a"),
        "e4a_vs_e4b": ("e4a", "e4b"),
        "e3_vs_e4b": ("e3", "e4b"),
    }
    pairwise: dict[str, Any] = {}
    for label, (left, right) in pairs.items():
        left_metrics, right_metrics = run_metrics[left], run_metrics[right]
        pairwise[label] = {
            "delta_auc_0_100_normalized": (
                right_metrics["early"]["auc_0_100_normalized"]
                - left_metrics["early"]["auc_0_100_normalized"]
            ),
            "delta_final_step_200": (
                right_metrics["final"]["final_step_200"] - left_metrics["final"]["final_step_200"]
            ),
            "delta_mean_tool_calls": (
                right_metrics["behavior"]["mean_tool_calls"]
                - left_metrics["behavior"]["mean_tool_calls"]
            ),
            "delta_tool_success_rate": (
                right_metrics["behavior"]["tool_success_rate"]
                - left_metrics["behavior"]["tool_success_rate"]
            ),
            "delta_tool_error_rate": (
                right_metrics["behavior"]["tool_error_rate"]
                - left_metrics["behavior"]["tool_error_rate"]
            ),
            "delta_invalid_rate": (
                right_metrics["behavior"]["invalid_rate"]
                - left_metrics["behavior"]["invalid_rate"]
            ),
            "delta_parser_error_rate": (
                right_metrics["behavior"]["parser_error_rate"]
                - left_metrics["behavior"]["parser_error_rate"]
            ),
            "delta_truncated_rate": (
                right_metrics["behavior"]["truncated_rate"]
                - left_metrics["behavior"]["truncated_rate"]
            ),
            "delta_repeated_code_rate": (
                right_metrics["behavior"]["repeated_code_trajectory_rate"]
                - left_metrics["behavior"]["repeated_code_trajectory_rate"]
            ),
            "delta_trivial_code_rate": (
                right_metrics["behavior"]["trivial_code_trajectory_rate"]
                - left_metrics["behavior"]["trivial_code_trajectory_rate"]
            ),
            "delta_unused_result_candidate_rate": (
                right_metrics["behavior"]["unused_tool_result_candidate_rate"]
                - left_metrics["behavior"]["unused_tool_result_candidate_rate"]
            ),
        }
    payload = {"runs": run_metrics, "pairwise": pairwise}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e3_e4a_e4b_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3", required=True, type=Path)
    parser.add_argument("--e4a", required=True, type=Path)
    parser.add_argument("--e4b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                {"e3": args.e3.resolve(), "e4a": args.e4a.resolve(), "e4b": args.e4b.resolve()},
                args.output_dir.resolve(),
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
