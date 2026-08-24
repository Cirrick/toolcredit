"""E5-only agent loop that records lossless assistant-turn metadata."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, FunctionCall

from rl.custom.tool_agent_loop import ToolCreditAgentLoop
from rl.custom.turn_advantage import (
    ADOPTION_VERSION,
    LEDGER_KEY,
    LEDGER_SCHEMA_VERSION,
    classify_adoption,
    decision_to_dict,
    normalize_visible_text,
)


def _serialize_call(call: FunctionCall) -> dict[str, str]:
    return {"name": str(call.name), "arguments": str(call.arguments)}


def _tool_codes(turns: list[dict[str, Any]]) -> list[str]:
    codes: list[str] = []
    for turn in turns:
        for call in turn.get("parsed_calls", []):
            if call.get("name") != "code_interpreter":
                continue
            try:
                arguments = json.loads(call.get("arguments", ""))
            except (json.JSONDecodeError, TypeError):
                continue
            code = arguments.get("code") if isinstance(arguments, dict) else None
            if isinstance(code, str):
                codes.append(code)
    return codes


def _execution_status(
    call: FunctionCall, active_tools: dict[str, Any], visible_text: str | None, metrics: dict[str, Any]
) -> tuple[int, int, str]:
    if call.name not in active_tools:
        return 0, 0, "unknown_tool"
    try:
        arguments = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        return 0, 0, "invalid_arguments_json"
    if not isinstance(arguments, dict):
        return 0, 0, "invalid_arguments_schema"

    executed = 1
    if float(metrics.get("tool_success", 0.0) or 0.0) == 1.0:
        return executed, 1, "ok"
    lowered = (visible_text or "").casefold()
    if lowered.startswith("timeout:"):
        status = "timeout"
    elif lowered.startswith("error: code_interpreter expects"):
        status = "invalid_arguments_schema"
    elif lowered.startswith("error executing tool"):
        status = "execution_exception"
    elif lowered.startswith("error:"):
        status = "sandbox_error"
    else:
        status = "execution_error"
    return executed, 0, status


class TurnCreditAgentLoop(ToolCreditAgentLoop):
    """Preserve E3 behavior while adding E5's structured per-turn ledger."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        self._turn_credit_trajectory_uid = uuid4().hex
        output = await super().run(sampling_params, **kwargs)
        ledger = output.extra_fields.get(LEDGER_KEY)
        if not isinstance(ledger, dict):
            raise RuntimeError("E5 agent loop did not produce a turn-credit ledger")
        self._finalize_ledger(output, ledger)
        output.extra_fields[LEDGER_KEY] = ledger
        return output

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        ledger = self._ledger(agent_data)
        start = len(agent_data.response_mask)
        parse_errors_before = int(agent_data.extra_fields.get("tool_parse_error_count", 0))
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        end = len(agent_data.response_mask)
        parse_errors_after = int(agent_data.extra_fields.get("tool_parse_error_count", 0))
        parsed_calls = list(agent_data.tool_calls) if state == AgentState.PROCESSING_TOOLS else []
        ledger["assistant_turns"].append(
            {
                "turn_index": len(ledger["assistant_turns"]),
                "policy_token_start": start,
                "policy_token_end": end,
                "parsed_call_count": len(parsed_calls),
                "parsed_calls": [_serialize_call(call) for call in parsed_calls],
                "executed_call_count": 0,
                "parser_error_count": parse_errors_after - parse_errors_before,
                "execution_success": 0,
                "execution_status": "no_tool" if not parsed_calls else "pending",
                "visible_tool_response_text": None,
                "raw_sandbox_stdout": None,
                "tool_token_start": None,
                "tool_token_end": None,
                "adopted": 0,
                "adoption_status": "no_tool" if not parsed_calls else "pending",
                "adoption_evidence": None,
                "s_t": 0,
            }
        )
        if state == AgentState.TERMINATED:
            if not ignore_termination and end >= self.response_length:
                ledger["trajectory_truncated"] = True
                ledger["termination_reason"] = "response_length"
            elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                ledger["trajectory_truncated"] = True
                ledger["termination_reason"] = "assistant_turn_budget"
            elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
                ledger["trajectory_truncated"] = True
                ledger["termination_reason"] = "tool_turn_budget"
            else:
                ledger["termination_reason"] = "final_answer"
        return state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        ledger = self._ledger(agent_data)
        if not ledger["assistant_turns"]:
            raise RuntimeError("tool processing occurred before an assistant turn was recorded")
        turn = ledger["assistant_turns"][-1]
        start = len(agent_data.response_mask)
        state = await super()._handle_processing_tools_state(agent_data)
        end = len(agent_data.response_mask)
        pending = turn.pop("_pending_visible_tool_response_text", None)
        if end > start:
            turn["tool_token_start"] = start
            turn["tool_token_end"] = end
            turn["visible_tool_response_text"] = pending
        else:
            turn["unencoded_tool_response_text"] = pending
            turn["visible_tool_response_text"] = None
            ledger["trajectory_truncated"] = True
            ledger["termination_reason"] = "tool_response_would_exceed_response_length"
        return state

    async def _call_tool(
        self,
        tool_call: FunctionCall,
        tools_kwargs: dict[str, Any],
        agent_data: AgentData,
    ) -> tuple[Any, float, dict[str, Any]]:
        result = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        ledger = self._ledger(agent_data)
        if not ledger["assistant_turns"]:
            raise RuntimeError("tool call occurred before an assistant turn was recorded")
        turn = ledger["assistant_turns"][-1]
        visible_text = result[0].text
        metrics = result[2] if isinstance(result[2], dict) else {}
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        executed, success, status = _execution_status(tool_call, active_tools, visible_text, metrics)
        turn["executed_call_count"] = executed
        turn["execution_success"] = success
        turn["execution_status"] = status
        turn["_pending_visible_tool_response_text"] = visible_text
        return result

    def _ledger(self, agent_data: AgentData) -> dict[str, Any]:
        ledger = agent_data.extra_fields.get(LEDGER_KEY)
        if ledger is None:
            ledger = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "adoption_version": ADOPTION_VERSION,
                "trajectory_uid": self._turn_credit_trajectory_uid,
                "response_length_unpadded": None,
                "trajectory_truncated": False,
                "termination_reason": None,
                "assistant_turns": [],
            }
            agent_data.extra_fields[LEDGER_KEY] = ledger
        if not isinstance(ledger, dict):
            raise TypeError("turn-credit ledger must be a dictionary")
        return ledger

    def _finalize_ledger(self, output: AgentLoopOutput, ledger: dict[str, Any]) -> None:
        response_length = len(output.response_ids)
        if len(output.response_mask) != response_length:
            raise ValueError("response ids/mask length mismatch before turn-credit finalization")
        ledger["response_length_unpadded"] = response_length
        turns = ledger["assistant_turns"]

        for turn in turns:
            start = int(turn["policy_token_start"])
            end = min(int(turn["policy_token_end"]), response_length)
            if not 0 <= start < end <= response_length:
                raise ValueError(f"assistant span was truncated away: [{start}, {end})")
            turn["policy_token_end"] = end
            if output.response_mask[start:end] != [1] * (end - start):
                raise ValueError(f"assistant span {turn['turn_index']} is not fully policy-masked")
            turn["assistant_text"] = self.tokenizer.decode(
                output.response_ids[start:end], skip_special_tokens=True
            )
            tool_start = turn.get("tool_token_start")
            tool_end = turn.get("tool_token_end")
            if tool_start is not None:
                tool_start = int(tool_start)
                tool_end = min(int(tool_end), response_length)
                if not 0 <= tool_start < tool_end <= response_length:
                    raise ValueError(f"invalid encoded tool span: [{tool_start}, {tool_end})")
                turn["tool_token_end"] = tool_end
                if output.response_mask[tool_start:tool_end] != [0] * (tool_end - tool_start):
                    raise ValueError(f"tool span {turn['turn_index']} is not fully masked zero")
                decoded_tool = self.tokenizer.decode(
                    output.response_ids[tool_start:tool_end], skip_special_tokens=True
                )
                visible = turn.get("visible_tool_response_text")
                if visible and normalize_visible_text(visible) not in normalize_visible_text(decoded_tool):
                    raise ValueError("visible tool text does not match its encoded rollout span")

        for index, turn in enumerate(turns):
            parsed_count = int(turn.get("parsed_call_count", 0))
            parser_errors = int(turn.get("parser_error_count", 0))
            if parsed_count == 0:
                turn["adoption_status"] = "no_tool"
                continue
            if parsed_count != 1 or parser_errors:
                turn["adoption_status"] = "ambiguous_multiple_calls" if parsed_count != 1 else "parser_error"
                continue
            if int(turn.get("execution_success", 0)) != 1:
                turn["adoption_status"] = "execution_failed"
                continue
            later_turns = turns[index + 1 :]
            decision = classify_adoption(
                turn.get("visible_tool_response_text"),
                [str(item.get("assistant_text", "")) for item in later_turns],
                _tool_codes(later_turns),
            )
            if ledger.get("trajectory_truncated") and decision.status in {"not_found", "no_output"}:
                decision_dict = decision_to_dict(decision)
                decision_dict.update({"status": "unobservable_truncated", "s_t": 0})
            else:
                decision_dict = decision_to_dict(decision)
            turn["adoption_status"] = decision_dict["status"]
            turn["adoption_evidence"] = decision_dict["evidence"]
            turn["adopted"] = int(decision_dict["s_t"])
            turn["s_t"] = int(decision_dict["s_t"])

        if not turns:
            raise ValueError("E5 trajectory contains no assistant turns")
