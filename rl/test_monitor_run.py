from __future__ import annotations

import json

from rl.monitor_run import read_rollout_metrics


def test_rollout_monitor_computes_group_and_error_metrics(tmp_path) -> None:
    rows = [
        {
            "sample_id": "a",
            "acc": 1.0,
            "score": 1.1,
            "n_tool_calls": 1,
            "tool_error_rate": 0.0,
            "format_ok": 1.0,
            "invalid": 0.0,
            "truncated": 0.0,
        },
        {
            "sample_id": "a",
            "acc": 1.0,
            "score": 1.1,
            "n_tool_calls": 2,
            "tool_error_rate": 0.5,
            "format_ok": 1.0,
            "invalid": 0.0,
            "truncated": 0.0,
        },
        {
            "sample_id": "b",
            "acc": 0.0,
            "score": 0.0,
            "n_tool_calls": 0,
            "tool_error_rate": 0.0,
            "format_ok": 0.0,
            "invalid": 0.0,
            "truncated": 0.0,
        },
        {
            "sample_id": "b",
            "acc": 0.0,
            "score": 0.0,
            "n_tool_calls": 0,
            "tool_error_rate": 0.0,
            "format_ok": 0.0,
            "invalid": 0.0,
            "truncated": 0.0,
        },
    ]
    path = tmp_path / "3.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metrics = read_rollout_metrics(tmp_path)[3]
    assert metrics["group_all_correct_frac"] == 0.5
    assert metrics["group_all_wrong_frac"] == 0.5
    assert metrics["tool_error_rate"] == 0.125
    assert metrics["invalid_format_rate"] == 0.5
