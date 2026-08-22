from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.analyze_e4 import analyze, candidate_reasons, code_hash, is_trivial_code


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": "sample-1",
        "output": (
            '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(42)"}}</tool_call>'
            "user<tool_response>42</tool_response>assistant answer 42"
        ),
        "score": 0.3,
        "base_score": 0.1,
        "exec_success_fraction": 1.0,
        "exec_bonus": 0.2,
        "budget_penalty": 0.0,
        "n_tool_calls": 1.0,
        "n_tool_success": 1.0,
        "acc": 0.0,
        "invalid": 0.0,
        "tool_parse_errors": 0.0,
        "truncated": 0.0,
    }
    row.update(overrides)
    return row


def _write_run(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "predictions/train").mkdir(parents=True)
    (run_dir / "predictions/validation").mkdir(parents=True)
    (run_dir / "predictions/train/1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "resolved_config.yaml").write_text(
        "reward:\n  custom_reward_function:\n    reward_kwargs:\n      lambda_exec: 0.2\n"
        "      lambda_budget: 0.0\n      budget: 3\n",
        encoding="utf-8",
    )
    return run_dir


def test_code_normalization_and_trivial_detection() -> None:
    assert code_hash("print(42)") == code_hash("print( 42 )")
    assert is_trivial_code("print(42)")
    assert is_trivial_code("pass")
    assert not is_trivial_code("x = 40\nx + 2")


def test_candidate_rules_cover_full_bonus_repeat_and_unused() -> None:
    output = (
        '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(42)"}}</tool_call>'
        "<tool_response>42</tool_response>"
        '<tool_call>{"name":"code_interpreter","arguments":{"code":"print(42)"}}</tool_call>'
        "<tool_response>42</tool_response>assistant unrelated conclusion"
    )
    reasons, hashes = candidate_reasons(_row(output=output, n_tool_calls=2.0), False)
    assert "wrong_answer_with_full_exec_bonus" in reasons
    assert "shaping_reward_without_answer_credit" in reasons
    assert "repeated_normalized_code" in reasons
    assert "trivial_or_noop_code" in reasons
    assert "tool_result_not_used" in reasons
    assert len(hashes) == 2


def test_analyze_writes_full_candidate_and_aggregate(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [_row()])
    summary = analyze(run_dir)
    assert summary["candidate_count"] == 1
    assert summary["aggregate"]["train"]["mean_base_score"] == pytest.approx(0.1)
    candidate = json.loads((run_dir / "analysis/e4_hacking_candidates.jsonl").read_text())
    assert candidate["trajectory"]["output"] == _row()["output"]


def test_analyze_rejects_breakdown_mismatch(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [_row(score=0.9)])
    with pytest.raises(ValueError, match="does not add up"):
        analyze(run_dir)
