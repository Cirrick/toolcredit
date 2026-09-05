"""M8 / E5-v2 agent loop: the frozen E5 v1 ledger plus plan §3.2 conditions 3 and 4.

``rl/custom/turn_credit_agent_loop.py`` is hashed by the M6/M7 freeze closures and is not
modified; this subclass only swaps the ledger version strings and appends the v2 fields
after the v1 finalization has run.  Generation, tool execution, masking and the v1
adoption heuristic are byte-for-byte the frozen implementation.
"""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData

from rl.custom.turn_advantage import LEDGER_KEY
from rl.custom.turn_advantage_v2 import (
    ADOPTION_VERSION_V2,
    LEDGER_SCHEMA_VERSION_V2,
    v2_turn_conditions,
)
from rl.custom.turn_credit_agent_loop import TurnCreditAgentLoop, _tool_codes


class TurnCreditV2AgentLoop(TurnCreditAgentLoop):
    """v1 structured ledger, then ``s_t_v1``/``repeat_of_turn``/``after_boxed``/``credit_eligible``/``s_t``."""

    def _ledger(self, agent_data: AgentData) -> dict[str, Any]:
        ledger = agent_data.extra_fields.get(LEDGER_KEY)
        if ledger is None:
            ledger = {
                "schema_version": LEDGER_SCHEMA_VERSION_V2,
                "adoption_version": ADOPTION_VERSION_V2,
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
        super()._finalize_ledger(output, ledger)
        if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION_V2:
            raise ValueError("E5-v2 loop finalized a non-v2 ledger")
        turns = ledger["assistant_turns"]
        codes_by_turn = [_tool_codes([turn]) for turn in turns]
        texts = [str(turn.get("assistant_text", "")) for turn in turns]
        eligible = [
            int(turn.get("parsed_call_count", 0)) == 1 and int(turn.get("parser_error_count", 0)) == 0
            for turn in turns
        ]
        s_t_v1 = [int(turn["s_t"]) for turn in turns]
        conditions = v2_turn_conditions(codes_by_turn, texts, s_t_v1, eligible)
        for turn, item in zip(turns, conditions, strict=True):
            if item["s_t_v1"] == 1 and not item["credit_eligible"]:
                raise ValueError("v1 adoption marked an ineligible turn as s_t=1")
            turn.update(item)
            if item["s_t_v1"] == 1 and item["s_t"] == 0:
                turn["adoption_status_v2"] = (
                    "repeated_code" if item["repeat_of_turn"] is not None else "after_boxed"
                )
            else:
                turn["adoption_status_v2"] = str(turn.get("adoption_status"))
