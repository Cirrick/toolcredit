"""Exact pass@k and behavior denominator fixtures."""

from __future__ import annotations

import pytest

from eval.metrics import pass_at_k, pass_metrics, tool_behavior_metrics


@pytest.mark.parametrize(
    ("c", "expected"),
    [
        (0, {1: 0.0, 2: 0.0, 4: 0.0}),
        (1, {1: 0.25, 2: 0.5, 4: 1.0}),
        (2, {1: 0.5, 2: 5 / 6, 4: 1.0}),
        (3, {1: 0.75, 2: 1.0, 4: 1.0}),
        (4, {1: 1.0, 2: 1.0, 4: 1.0}),
    ],
)
def test_unbiased_pass_at_k_n4(c: int, expected: dict[int, float]) -> None:
    assert {k: pass_at_k(4, c, k) for k in (1, 2, 4)} == expected


def test_pass_at_k_boundaries_fail_closed() -> None:
    with pytest.raises(ValueError):
        pass_at_k(4, -1, 1)
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)
    with pytest.raises(ValueError):
        pass_at_k(4, 2, 5)


def _row(question: str, index: int, correct: bool) -> dict:
    return {
        "checkpoint_role": "raw_base_model",
        "evaluation_setting": "sampled",
        "source": "MATH500",
        "question_id": question,
        "sample_index": index,
        "correct": correct,
        "invalid": False,
        "truncated": False,
        "reward": {"n_tool_calls": 0.0, "tool_error_rate": 0.0, "tool_parse_errors": 0.0},
    }


def test_pass_metrics_average_question_estimators() -> None:
    rows = [_row("q1", i, i < 2) for i in range(4)] + [_row("q2", i, False) for i in range(4)]
    metrics = pass_metrics(rows)["raw_base_model:sampled:MATH500"]
    assert metrics["pass@1"]["value"] == 0.25
    assert metrics["pass@2"]["value"] == pytest.approx(5 / 12)
    assert metrics["pass@4"]["value"] == 0.5
    behavior = tool_behavior_metrics(rows)
    assert behavior["tool_call_rate"] == {"numerator": 0, "denominator": 8, "value": 0.0, "zero_denominator": False}
    assert behavior["parser_error_recovery"]["value"] is None
