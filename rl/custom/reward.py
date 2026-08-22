"""veRL custom reward entry point for M4 / E3.

The scalar is delegated to the M2 composite reward with E4 shaping disabled.
The returned components are persisted by veRL in rollout/validation JSONL so
invalid parsing, tool errors, and group variance remain observable.
"""

from __future__ import annotations

from typing import Any

from rewards.composite_reward import RewardConfig, compute_reward

MAX_TOOL_CALLS = 4


def _as_nonnegative_int(extra_info: dict[str, Any], key: str) -> int:
    value = extra_info.get(key, 0)
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{key} must be nonnegative, got {parsed}")
    return parsed


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    lambda_exec: float | None = None,
    lambda_budget: float | None = None,
    budget: int | None = None,
    **_: Any,
) -> dict[str, float | str]:
    """Return the E3 scalar reward and auditable numeric components."""
    if data_source not in {"toolcredit_math", "toolcredit_math500"}:
        raise ValueError(f"unexpected data_source for ToolCredit reward: {data_source!r}")
    info = dict(extra_info or {})
    n_tool_calls = _as_nonnegative_int(info, "tool_call_counts")
    n_tool_success = _as_nonnegative_int(info, "tool_success_count")
    n_tool_errors = _as_nonnegative_int(info, "tool_error_count")
    n_parse_errors = _as_nonnegative_int(info, "tool_parse_error_count")
    if n_tool_success + n_tool_errors > n_tool_calls:
        raise ValueError("tool success/error counts exceed total calls")

    # Four calls without a boxed final answer means the trajectory exhausted the
    # PLAN §3.2 tool budget. Length-only truncation still receives zero through
    # the strict boxed outcome/format terms and remains visible in response clips.
    truncated = n_tool_calls >= MAX_TOOL_CALLS and "\\boxed{" not in solution_str
    shaping_enabled = any(value is not None for value in (lambda_exec, lambda_budget, budget))
    reward_config = RewardConfig(
        lambda_exec=0.0 if lambda_exec is None else lambda_exec,
        lambda_budget=0.0 if lambda_budget is None else lambda_budget,
        budget=3 if budget is None else budget,
    )
    result = compute_reward(
        final_text=solution_str,
        gold=ground_truth,
        n_tool_calls=n_tool_calls,
        n_tool_success=n_tool_success,
        all_tool_calls_parsed=n_parse_errors == 0,
        truncated=truncated,
        config=reward_config,
    )
    tool_error_rate = n_tool_errors / n_tool_calls if n_tool_calls else 0.0
    sample_id = str(info.get("id", "unknown"))
    output: dict[str, float | str] = {
        "score": result["reward"],
        "acc": float(result["answer_correct"]),
        "format_ok": float(result["format_ok"]),
        "invalid": float(result["invalid"] or n_parse_errors > 0),
        "n_tool_calls": float(n_tool_calls),
        "tool_error_rate": tool_error_rate,
        "tool_parse_errors": float(n_parse_errors),
        "truncated": float(truncated),
        "verify_method": result["verify_method"],
        "sample_id": sample_id,
    }
    audit_keys = (
        "original_policy_token_count",
        "original_tool_return_token_count",
        "nomask_loss_token_count",
    )
    present_audit_keys = [key for key in audit_keys if key in info]
    if present_audit_keys and len(present_audit_keys) != len(audit_keys):
        raise ValueError("incomplete E6 mask audit fields")
    for key in present_audit_keys:
        output[key] = float(_as_nonnegative_int(info, key))
    if present_audit_keys:
        expected = output["original_policy_token_count"] + output["original_tool_return_token_count"]
        if output["nomask_loss_token_count"] != expected:
            raise ValueError("E6 mask audit token counts do not add up")
    if shaping_enabled:
        output.update(
            {
                "base_score": result["base_score"],
                "exec_success_fraction": result["exec_success_fraction"],
                "exec_bonus": result["exec_bonus"],
                "budget_penalty": result["budget_penalty"],
                "n_tool_success": float(n_tool_success),
            }
        )
    return output
