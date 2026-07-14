"""Reward/verifier test suite (M2, PLAN §6.2–6.3). Run: python -m pytest rewards/ -v"""

import pytest

from rewards.composite_reward import RewardConfig, compute_reward
from rewards.format_reward import format_ok
from rewards.verifier import extract_boxed, verify_answer

# ---------- boxed extraction ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is \\boxed{42}.", "42"),
        ("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}"),
        ("nested \\boxed{\\sqrt{x^{2}}} ok", "\\sqrt{x^{2}}"),
        ("two: \\boxed{1} then \\boxed{2}", "2"),  # last one wins
        ("no box here", None),
        ("\\boxed{unbalanced", None),
    ],
)
def test_extract_boxed(text: str, expected: str | None) -> None:
    assert extract_boxed(text) == expected


# ---------- equivalence judgements ----------


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        ("\\boxed{\\frac{3}{4}}", "0.75"),  # fraction vs decimal
        ("\\boxed{3/4}", "\\frac{3}{4}"),  # plain vs latex fraction
        ("\\boxed{\\sqrt{8}}", "2\\sqrt{2}"),  # radical normalization
        ("\\boxed{(3, \\frac{\\pi}{2})}", "\\left( 3, \\frac{\\pi}{2} \\right)"),  # coordinate pair
        ("\\boxed{[2, 5)}", "[2,5)"),  # interval
        ("\\boxed{-\\frac{1}{2}}", "-0.5"),  # leading minus
        ("\\boxed{2\\pi}", "2\\pi"),  # pi expression
        ("\\boxed{x=3}", "3"),  # equation vs value
        ("gold text …\\boxed{0}", "0"),  # M1 probe row math_train_000000 style
    ],
)
def test_equivalent(pred: str, gold: str) -> None:
    r = verify_answer(pred, gold)
    assert r["correct"], r
    assert not r["invalid"]


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        ("\\boxed{3}", "4"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{3}"),
        ("\\boxed{50\\%}", "50"),  # 50% = 0.5 ≠ 50 (math-verify semantics)
        ("\\boxed{(1, 2)}", "(2, 1)"),
    ],
)
def test_not_equivalent(pred: str, gold: str) -> None:
    r = verify_answer(pred, gold)
    assert not r["correct"], r


# ---------- strict vs lenient extraction (anti-hacking regime) ----------


def test_strict_no_boxed_is_wrong_not_invalid() -> None:
    r = verify_answer("The answer is 42.", "42", strict_boxed=True)
    assert not r["correct"]
    assert r["method"] == "no_boxed"
    assert not r["invalid"]  # well-defined outcome, not a machinery failure


def test_lenient_extracts_without_boxed() -> None:
    r = verify_answer("So the final answer is $42$.", "42", strict_boxed=False)
    assert r["correct"]


def test_strict_blocks_candidate_spray() -> None:
    """Anti-hacking: multiple candidates without boxed must NOT score under training rules."""
    spray = "It could be 41, or 42, maybe 43."
    assert not verify_answer(spray, "42", strict_boxed=True)["correct"]


def test_garbage_does_not_crash() -> None:
    r = verify_answer("\\boxed{\\@#$%^&}", "42")
    assert not r["correct"]  # judged wrong or invalid — but never raises


# ---------- format reward ----------


def test_format_ok() -> None:
    assert format_ok("answer \\boxed{1}", all_tool_calls_parsed=True)
    assert not format_ok("answer 1", all_tool_calls_parsed=True)
    assert not format_ok("answer \\boxed{1}", all_tool_calls_parsed=False)


# ---------- composite reward ----------


def test_composite_correct_plus_format() -> None:
    r = compute_reward("\\boxed{42}", "42")
    assert r["reward"] == pytest.approx(1.1)
    assert r["answer_correct"] and r["format_ok"]


def test_composite_format_only() -> None:
    r = compute_reward("\\boxed{41}", "42")
    assert r["reward"] == pytest.approx(0.1)


def test_composite_truncated_is_zero() -> None:
    r = compute_reward("\\boxed{42}", "42", truncated=True)
    assert r["reward"] == 0.0
    assert r["verify_method"] == "truncated"


def test_composite_e4_exec_bonus() -> None:
    cfg = RewardConfig(lambda_exec=0.2)
    r = compute_reward("\\boxed{42}", "42", n_tool_calls=4, n_tool_success=3, config=cfg)
    assert r["reward"] == pytest.approx(1.1 + 0.2 * 0.75)


def test_composite_e4_budget_penalty() -> None:
    cfg = RewardConfig(lambda_budget=0.1, budget=3)
    r = compute_reward("\\boxed{42}", "42", n_tool_calls=5, n_tool_success=5, config=cfg)
    assert r["reward"] == pytest.approx(1.1 - 0.1 * 2)


def test_composite_defaults_are_sparse() -> None:
    """Default config must stay outcome+format only (E5 experimental precondition)."""
    cfg = RewardConfig()
    assert cfg.lambda_exec == 0.0 and cfg.lambda_budget == 0.0
