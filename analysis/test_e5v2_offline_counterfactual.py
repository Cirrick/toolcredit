"""Step-0 tests for the E5-v2 offline counterfactual (plan §4, §7 Fixture F/H on pure functions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.e5v2_offline_counterfactual import analyze, trajectory_counterfactual
from rl.custom.turn_advantage_v2 import conserved_turn_corrections, v2_turn_conditions


def _turn(index: int, start: int, end: int, status: str, s_t: int, text: str, correction: float = 0.0) -> dict:
    return {
        "turn_index": index,
        "adoption_status": status,
        "s_t": s_t,
        "correction": correction,
        "policy_token_span": [start, end],
        "assistant_text": text,
    }


def _call(code: str) -> str:
    return '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": ' + json.dumps(code) + "}}\n</tool_call>"


def test_fixture_f_hand_computed_conservation() -> None:
    corrections, s_tilde, gated = conserved_turn_corrections(
        [1, 0, 1, 0], [10, 30, 20, 100], [True, True, True, False], beta=0.5, informative_group=True
    )
    assert s_tilde == pytest.approx(0.5)
    assert corrections == pytest.approx([0.25, -0.25, 0.25, 0.0])
    assert sum(c * n for c, n in zip(corrections, [10, 30, 20, 100])) == pytest.approx(0.0, abs=1e-12)
    assert gated is False


def test_gating_single_call_zero_variance_and_uniform_s() -> None:
    assert conserved_turn_corrections([1, 0], [5, 5], [True, False], beta=0.5, informative_group=True) == (
        [0.0, 0.0],
        None,
        True,
    )
    assert conserved_turn_corrections([1, 0, 0], [5, 5, 5], [True, True, False], beta=0.5, informative_group=False) == (
        [0.0, 0.0, 0.0],
        None,
        True,
    )
    uniform, s_tilde, gated = conserved_turn_corrections(
        [1, 1, 0], [7, 9, 3], [True, True, False], beta=0.5, informative_group=True
    )
    assert uniform == [0.0, 0.0, 0.0] and s_tilde == 1.0 and gated is False


def test_fixture_h_repeat_and_boxed_conditions() -> None:
    codes = [["x = 1\nprint(x)"], ["# comment\ny=1 ;  print( y )"], ["print(2)"], []]
    texts = ["call one", "call two", "So \\boxed{2} " , "final"]
    results = v2_turn_conditions(codes, texts, [1, 1, 1, 0], [True, True, True, False])
    assert [item["repeat_of_turn"] for item in results] == [None, None, None, None]
    assert results[1]["s_t"] == 1  # renamed variable is a different AST, not a repeat
    same = v2_turn_conditions(
        [["x = 1\nprint(x)"], ["x=1\n\n# note\nprint(x)"], ["print(3)"]],
        ["a", "b \\boxed{1}", "c"],
        [1, 1, 1],
        [True, True, True],
    )
    assert same[1]["repeat_of_turn"] == 0 and same[1]["s_t"] == 0 and same[1]["s_t_v1"] == 1
    assert same[2]["after_boxed"] is True and same[2]["s_t"] == 0 and same[2]["s_t_v1"] == 1
    assert same[0]["s_t"] == 1


def test_trajectory_counterfactual_uses_v1_status_partition() -> None:
    audit = {
        "turns": [
            _turn(0, 0, 10, "adopted", 1, "a " + _call("print(1)"), 0.3),
            _turn(1, 12, 42, "execution_failed", 0, "b " + _call("print(1/0)"), -0.2),
            _turn(2, 44, 64, "adopted", 1, "c " + _call("print(1)"), 0.3),
            _turn(3, 66, 166, "no_tool", 0, "\\boxed{1}", -0.4),
        ]
    }
    result = trajectory_counterfactual(audit, informative_group=True)
    assert result["eligible_turn_count"] == 3
    assert result["s_t_v2"] == [1, 0, 0, 0]
    assert result["repeat_of_turn"] == [None, None, 0, None]
    assert result["conservation_residual"] == pytest.approx(0.0, abs=1e-9)
    assert result["treated"] is True
    gated = trajectory_counterfactual(audit, informative_group=False)
    assert gated["treated"] is False and gated["treated_if_ungated"] is True


def test_analyze_synthetic_run(tmp_path: Path) -> None:
    train = tmp_path / "predictions/train"
    train.mkdir(parents=True)

    def row(step: int, uid: str, score: float, turns: list[dict]) -> str:
        return json.dumps({"step": step, "score": score, "turn_credit_audit": {"prompt_uid": uid, "turns": turns}})

    two_calls = [
        _turn(0, 0, 10, "adopted", 1, _call("print(1)")),
        _turn(1, 12, 20, "adopted", 1, _call("print(2)")),
        _turn(2, 22, 30, "no_tool", 0, "\\boxed{2}"),
    ]
    mixed = [
        _turn(0, 0, 10, "adopted", 1, _call("print(1)")),
        _turn(1, 12, 20, "execution_failed", 0, _call("print(1/0)")),
        _turn(2, 22, 30, "no_tool", 0, "\\boxed{2}"),
    ]
    single = [_turn(0, 0, 10, "adopted", 1, _call("print(1)")), _turn(1, 12, 20, "no_tool", 0, "\\boxed{1}")]
    lines = [
        row(1, "g1", 1.1, mixed),  # informative group, treated
        row(1, "g1", 0.1, two_calls),  # uniform s -> not treated
        row(1, "g2", 0.0, mixed),  # zero-variance group -> gated
        row(1, "g2", 0.0, single),
    ]
    (train / "1.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = analyze(tmp_path, ["1-1"])
    window = report["windows"]["1-1"]
    assert window["trajectories"] == 4 and window["groups"] == 2
    assert window["informative_group_fraction"] == 0.5
    assert window["multi_call_trajectory_fraction"] == 0.75
    assert window["treated_fraction_ungated"] == 0.5
    assert window["treated_fraction_v2"] == 0.25
    assert window["zero_variance_trajectories"] == 2
    assert window["max_abs_conservation_residual"] == 0.0
