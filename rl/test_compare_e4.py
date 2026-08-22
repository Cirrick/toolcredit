from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.compare_e4 import behavior_metrics, early_metrics, final_metrics, validation_curve


def test_early_auc_and_thresholds() -> None:
    curve = {0: 0.60, 25: 0.67, 50: 0.69, 75: 0.72, 100: 0.74}
    result = early_metrics(curve)
    expected = 25 * ((0.60 + 0.67) / 2 + (0.67 + 0.69) / 2 + (0.69 + 0.72) / 2 + (0.72 + 0.74) / 2)
    assert result["auc_0_100"] == pytest.approx(expected)
    assert result["auc_0_100_normalized"] == pytest.approx(expected / 100)
    assert result["first_validation_step_reaching"] == {"0.67": 25, "0.70": 75, "0.73": 100}


def test_final_uses_step_200_and_reports_earliest_peak() -> None:
    result = final_metrics({0: 0.6, 175: 0.77, 200: 0.76})
    assert result == {"final_step_200": 0.76, "peak": 0.77, "peak_step": 175}


def test_validation_curve_requires_fixed_100_rows(tmp_path: Path) -> None:
    path = tmp_path / "predictions/validation"
    path.mkdir(parents=True)
    (path / "0.jsonl").write_text(
        "".join(json.dumps({"acc": float(index < 60)}) + "\n" for index in range(100)),
        encoding="utf-8",
    )
    assert validation_curve(tmp_path) == {0: 0.6}
    (path / "25.jsonl").write_text(json.dumps({"acc": 1.0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="100 rows"):
        validation_curve(tmp_path)


def test_behavior_metrics_uses_same_hacking_heuristics(tmp_path: Path) -> None:
    path = tmp_path / "predictions/train"
    path.mkdir(parents=True)
    output = (
        '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(42)"}}</tool_call>'
        "<tool_response>42</tool_response>"
        '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(42)"}}</tool_call>'
        "<tool_response>42</tool_response>assistant unrelated"
    )
    row = {
        "output": output,
        "n_tool_calls": 2.0,
        "tool_error_rate": 0.0,
        "tool_parse_errors": 0.0,
        "invalid": 0.0,
        "truncated": 0.0,
        "score": 0.3,
        "base_score": 0.1,
        "exec_bonus": 0.2,
        "budget_penalty": 0.0,
    }
    (path / "1.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = behavior_metrics(tmp_path)
    assert result["repeated_code_trajectory_rate"] == 1.0
    assert result["trivial_code_trajectory_rate"] == 1.0
    assert result["unused_tool_result_candidate_rate"] == 1.0
    assert result["total_tool_calls"] == 2
    assert result["estimated_tool_successes"] == 2.0
    assert result["mean_tool_successes"] == 2.0
    assert result["tool_success_rate"] == 1.0


def test_behavior_metrics_aggregates_success_per_tool_call(tmp_path: Path) -> None:
    path = tmp_path / "predictions/train"
    path.mkdir(parents=True)
    rows = [
        {
            "output": "no tools",
            "n_tool_calls": 0.0,
            "tool_error_rate": 0.0,
            "tool_parse_errors": 0.0,
            "invalid": 0.0,
            "truncated": 0.0,
            "score": 0.0,
        },
        {
            "output": "two calls",
            "n_tool_calls": 2.0,
            "tool_error_rate": 0.5,
            "tool_parse_errors": 0.0,
            "invalid": 0.0,
            "truncated": 0.0,
            "score": 0.0,
        },
    ]
    (path / "1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = behavior_metrics(tmp_path)
    assert result["total_tool_calls"] == 2
    assert result["estimated_tool_successes"] == 1.0
    assert result["mean_tool_successes"] == 0.5
    assert result["tool_success_rate"] == 0.5
