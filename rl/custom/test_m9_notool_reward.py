"""M9 / E3-NoTool — plans/M9.md §6 fixture K: the E3 reward degrades correctly without tools.

E3-NoTool reuses ``rl/custom/reward.py:compute_score`` unchanged.  With no agent-loop tool
metadata the counts default to zero, the four-call truncation rule can never fire, and
``format_ok`` collapses to "a parseable \\boxed{} is present".
"""

from __future__ import annotations

from typing import Any

import pytest

from rl.custom.reward import compute_score

DATA_SOURCE = "toolcredit_math"
# The two keys veRL's reward manager always injects, even with no agent-loop tool fields.
VERL_INJECTED: dict[str, Any] = {"num_turns": 2, "rollout_reward_scores": {}}


def _score(solution: str, gold: str = "42", extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    return compute_score(
        data_source=DATA_SOURCE,
        solution_str=solution,
        ground_truth=gold,
        extra_info=extra_info,
    )


@pytest.mark.parametrize("extra_info", [None, {}, dict(VERL_INJECTED), {"id": "math_train_000001"}])
def test_fixture_k_tool_fields_default_to_zero(extra_info: dict[str, Any] | None) -> None:
    result = _score("The answer is \\boxed{42}.", extra_info=extra_info)
    assert result["n_tool_calls"] == 0.0
    assert result["truncated"] == 0.0
    assert result["tool_error_rate"] == 0.0
    assert result["tool_parse_errors"] == 0.0


def test_fixture_k_correct_boxed_answer_scores_answer_plus_format() -> None:
    result = _score("Reasoning in words only. \\boxed{42}")
    assert result["score"] == pytest.approx(1.1)
    assert result["acc"] == 1.0
    assert result["format_ok"] == 1.0
    assert result["invalid"] == 0.0


def test_fixture_k_missing_boxed_scores_zero() -> None:
    result = _score("The answer is 42, but I never boxed it.")
    assert result["score"] == 0.0
    assert result["acc"] == 0.0
    assert result["format_ok"] == 0.0
    assert result["truncated"] == 0.0


def test_fixture_k_boxed_but_wrong_keeps_only_the_format_term() -> None:
    result = _score("A confident mistake: \\boxed{41}")
    assert result["score"] == pytest.approx(0.1)
    assert result["acc"] == 0.0
    assert result["format_ok"] == 1.0


def test_fixture_k_length_truncation_is_not_reported_as_tool_truncation() -> None:
    """``truncated`` is the four-call budget flag; with zero calls it can never fire."""
    verbose = ("I will keep restating the problem. " * 400) + "and then I run out of budget"
    result = _score(verbose)
    assert result["truncated"] == 0.0
    assert result["score"] == 0.0
    assert result["n_tool_calls"] == 0.0


def test_fixture_k_no_shaping_components_are_emitted() -> None:
    result = _score("\\boxed{42}")
    for shaping_key in ("base_score", "exec_success_fraction", "exec_bonus", "budget_penalty", "n_tool_success"):
        assert shaping_key not in result
    assert set(result) == {
        "score",
        "acc",
        "format_ok",
        "invalid",
        "n_tool_calls",
        "tool_error_rate",
        "tool_parse_errors",
        "truncated",
        "verify_method",
        "sample_id",
    }


def test_fixture_k_tir_path_is_unchanged_by_m9() -> None:
    """Regression: with tool metadata present the scalar still matches the E3 semantics."""
    tir = compute_score(
        data_source=DATA_SOURCE,
        solution_str="used the sandbox then answered \\boxed{42}",
        ground_truth="42",
        extra_info={"tool_call_counts": 2, "tool_success_count": 2, "tool_error_count": 0},
    )
    assert tir["score"] == pytest.approx(1.1)
    assert tir["n_tool_calls"] == 2.0
    exhausted = compute_score(
        data_source=DATA_SOURCE,
        solution_str="four calls and no boxed answer",
        ground_truth="42",
        extra_info={"tool_call_counts": 4, "tool_success_count": 4, "tool_error_count": 0},
    )
    assert exhausted["truncated"] == 1.0
    assert exhausted["score"] == 0.0
