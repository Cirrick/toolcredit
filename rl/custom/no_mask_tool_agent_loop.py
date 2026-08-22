"""E6 agent-loop adapter that exposes tool-return tokens to the PPO loss."""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

from rl.custom.tool_agent_loop import ToolCreditAgentLoop

MASK_AUDIT_FIELDS = (
    "original_policy_token_count",
    "original_tool_return_token_count",
    "nomask_loss_token_count",
)


def expose_tool_return_tokens(output: AgentLoopOutput) -> None:
    """Set every real response token to loss weight one and record the delta."""
    if len(output.response_ids) != len(output.response_mask):
        raise ValueError("response IDs and mask must have identical lengths")
    if any(value not in (0, 1) for value in output.response_mask):
        raise ValueError("response mask must be binary before applying E6")

    policy_tokens = sum(output.response_mask)
    tool_tokens = len(output.response_mask) - policy_tokens
    output.extra_fields.update(
        {
            "original_policy_token_count": policy_tokens,
            "original_tool_return_token_count": tool_tokens,
            "nomask_loss_token_count": len(output.response_mask),
        }
    )
    output.response_mask = [1] * len(output.response_mask)


class ToolCreditNoMaskAgentLoop(ToolCreditAgentLoop):
    """Keep E3 generation unchanged while unmasking tool/environment returns."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        output = await super().run(sampling_params, **kwargs)
        expose_tool_return_tokens(output)
        return output
