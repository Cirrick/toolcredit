"""Real veRL state-machine fixture for E5 structured turn metadata."""

from __future__ import annotations

import asyncio

import pytest

from rl.custom.test_masking import (
    ASSISTANT_TURNS,
    MODEL_PATH,
    QUESTION,
    TOOL_OUTPUT_1,
    TOOL_OUTPUT_2,
    _build_loop,
)
from rl.custom.turn_advantage import LEDGER_KEY


def _successful_function_tool():
    from transformers.utils import get_json_schema
    from verl.tools.function_tool import FunctionTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    outputs = iter((TOOL_OUTPUT_1, TOOL_OUTPUT_2))

    async def code_interpreter(code: str):
        """Execute deterministic fixture code.

        Args:
            code: Python source.
        """
        del code
        return next(outputs), 0.0, {"tool_success": 1.0, "tool_error": 0.0}

    return FunctionTool(
        name="code_interpreter",
        fn=code_interpreter,
        tool_schema=OpenAIFunctionToolSchema(**get_json_schema(code_interpreter)),
        is_async=True,
    )


@pytest.fixture(scope="module")
def turn_credit_rollout():
    from transformers import AutoTokenizer

    from rl.custom.turn_credit_agent_loop import TurnCreditAgentLoop

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    loop = _build_loop(tokenizer, loop_class=TurnCreditAgentLoop)
    tool = _successful_function_tool()
    loop.tools = {"code_interpreter": tool}
    loop.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    output = asyncio.run(loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": QUESTION}]))
    return output


def test_structured_turn_spans_cover_exact_policy_mask(turn_credit_rollout) -> None:
    output = turn_credit_rollout
    ledger = output.extra_fields[LEDGER_KEY]
    assert ledger["response_length_unpadded"] == len(output.response_ids)
    assert len(ledger["assistant_turns"]) == len(ASSISTANT_TURNS)
    covered = [False] * len(output.response_ids)
    for turn in ledger["assistant_turns"]:
        start, end = turn["policy_token_start"], turn["policy_token_end"]
        assert output.response_mask[start:end] == [1] * (end - start)
        for index in range(start, end):
            assert not covered[index]
            covered[index] = True
        if turn["tool_token_start"] is not None:
            tool_start, tool_end = turn["tool_token_start"], turn["tool_token_end"]
            assert output.response_mask[tool_start:tool_end] == [0] * (tool_end - tool_start)
    assert covered == [bool(value) for value in output.response_mask]


def test_visible_tool_text_and_adoption_are_auditable(turn_credit_rollout) -> None:
    turns = turn_credit_rollout.extra_fields[LEDGER_KEY]["assistant_turns"]
    assert [turn["visible_tool_response_text"] for turn in turns[:2]] == [TOOL_OUTPUT_1, TOOL_OUTPUT_2]
    assert [turn["raw_sandbox_stdout"] for turn in turns] == [None, None, None]
    assert [turn["execution_success"] for turn in turns] == [1, 1, 0]
    assert [turn["adoption_status"] for turn in turns] == ["adopted", "adopted", "no_tool"]
    assert [turn["s_t"] for turn in turns] == [1, 1, 0]
