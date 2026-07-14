"""Format reward component (M2, PLAN §6.3): boxed present + tool calls parseable."""

from rewards.verifier import extract_boxed


def format_ok(final_text: str, all_tool_calls_parsed: bool = True) -> bool:
    """True iff the response ends with a parseable \\boxed{} answer and every tool
    call in the trajectory was syntactically parseable (caller supplies the flag
    from the rollout, since parsing happens in the agent loop)."""
    return extract_boxed(final_text) is not None and all_tool_calls_parsed
