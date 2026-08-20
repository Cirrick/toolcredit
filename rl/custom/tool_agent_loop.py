"""ToolAgentLoop adapter that keeps per-trajectory tool metadata rectangular."""

from __future__ import annotations

import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, FunctionCall, ToolAgentLoop

TOOL_COUNT_FIELDS = (
    "tool_call_counts",
    "tool_success_count",
    "tool_error_count",
    "tool_parse_error_count",
)


def ensure_tool_count_fields(extra_fields: dict[str, Any]) -> None:
    """Add zero counts for trajectories that finish without calling a tool."""
    for field in TOOL_COUNT_FIELDS:
        extra_fields.setdefault(field, 0)


def count_hermes_parse_errors(text: str, parsed_count: int) -> int:
    """Count malformed or incomplete Hermes tool-call blocks."""
    pattern = r"<tool_call>.*?</tool_call>"
    complete_count = len(re.findall(pattern, text, flags=re.DOTALL))
    remainder = re.sub(pattern, "", text, flags=re.DOTALL)
    candidate_count = complete_count + remainder.count("<tool_call>") + remainder.count("</tool_call>")
    return max(0, candidate_count - parsed_count)


class ToolCreditAgentLoop(ToolAgentLoop):
    """Use veRL's native loop while guaranteeing aligned tool-count columns."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        output = await super().run(sampling_params, **kwargs)
        ensure_tool_count_fields(output.extra_fields)
        return output

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        hit_length = not ignore_termination and len(agent_data.response_mask) >= self.response_length
        hit_assistant_budget = bool(
            self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns
        )
        hit_user_budget = bool(self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        parser_ran = not (hit_length or hit_assistant_budget or hit_user_budget)
        if parser_ran and self.tool_parser_name == "hermes":
            text = await self.loop.run_in_executor(None, self.tokenizer.decode, agent_data.response_ids)
            parse_errors = count_hermes_parse_errors(text, len(agent_data.tool_calls))
            if parse_errors:
                extras = agent_data.extra_fields
                extras["tool_parse_error_count"] = int(extras.get("tool_parse_error_count", 0)) + parse_errors
        return state

    async def _call_tool(
        self,
        tool_call: FunctionCall,
        tools_kwargs: dict[str, Any],
        agent_data: AgentData,
    ) -> tuple[Any, float, dict[str, Any]]:
        """Make dispatch/argument/execution failures visible in trajectory counts."""
        extras = agent_data.extra_fields
        calls_before = int(extras.get("tool_call_counts", 0))
        errors_before = int(extras.get("tool_error_count", 0))
        result = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        if not result[2]:
            if int(extras.get("tool_call_counts", 0)) == calls_before:
                extras["tool_call_counts"] = calls_before + 1
            if int(extras.get("tool_error_count", 0)) == errors_before:
                extras["tool_error_count"] = errors_before + 1
        return result
