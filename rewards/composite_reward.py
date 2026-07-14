"""Composite reward (M2, PLAN §6.3).

    r = w_answer · answer_correct
      + w_format · format_ok
      + lambda_exec · frac_successful_tool_calls    (E4 only, default 0)
      - lambda_budget · max(0, n_tool_calls - budget)  (E4 only, default 0)

Truncated trajectories (max_turns exhausted without a final answer) score 0 on the
outcome term by protocol (PLAN §3.2); their format term is naturally 0 too (no boxed).
The answer term uses strict_boxed=True — training reward only ever reads \\boxed{}
(anti-hacking, see rewards/verifier.py docstring).
"""

from dataclasses import dataclass
from typing import TypedDict

from rewards.format_reward import format_ok
from rewards.verifier import verify_answer


@dataclass(frozen=True)
class RewardConfig:
    w_answer: float = 1.0
    w_format: float = 0.1
    lambda_exec: float = 0.0  # E4: 0.2
    lambda_budget: float = 0.0  # E4: 0.1
    budget: int = 3


class RewardBreakdown(TypedDict):
    reward: float
    answer_correct: bool
    format_ok: bool
    verify_method: str
    invalid: bool


def compute_reward(
    final_text: str,
    gold: str,
    n_tool_calls: int = 0,
    n_tool_success: int = 0,
    all_tool_calls_parsed: bool = True,
    truncated: bool = False,
    config: RewardConfig = RewardConfig(),
) -> RewardBreakdown:
    if truncated:
        return RewardBreakdown(
            reward=0.0, answer_correct=False, format_ok=False, verify_method="truncated", invalid=False
        )

    verdict = verify_answer(final_text, gold, strict_boxed=True)
    fmt = format_ok(final_text, all_tool_calls_parsed)
    exec_frac = (n_tool_success / n_tool_calls) if n_tool_calls > 0 else 0.0
    reward = (
        config.w_answer * float(verdict["correct"])
        + config.w_format * float(fmt)
        + config.lambda_exec * exec_frac
        - config.lambda_budget * max(0, n_tool_calls - config.budget)
    )
    return RewardBreakdown(
        reward=round(reward, 6),
        answer_correct=verdict["correct"],
        format_ok=fmt,
        verify_method=verdict["method"],
        invalid=verdict["invalid"],
    )
