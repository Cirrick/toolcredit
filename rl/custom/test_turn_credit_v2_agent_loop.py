"""Real veRL state-machine fixture for the E5-v2 ledger (plans/M8.md §7 Fixture H, runtime)."""

from __future__ import annotations

import asyncio

from rl.custom.test_masking import MODEL_PATH, QUESTION, FakeServer, _build_loop
from rl.custom.turn_advantage import LEDGER_KEY
from rl.custom.turn_advantage_v2 import ADOPTION_VERSION_V2, LEDGER_SCHEMA_VERSION_V2

REPEAT_TURNS = [
    'Compute.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}\n</tool_call>',
    'Again.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "# same\\nprint( 2 + 2 )"}}\n</tool_call>',
    "So the answer is \\boxed{4}. Let me verify.\n"
    '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(4)"}}\n</tool_call>',
    "Confirmed: \\boxed{4}.",
]


def _v2_loop(turns: list[str]):
    from transformers import AutoTokenizer
    from transformers.utils import get_json_schema
    from verl.tools.function_tool import FunctionTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    from rl.custom.turn_credit_v2_agent_loop import TurnCreditV2AgentLoop

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    loop = _build_loop(tokenizer, loop_class=TurnCreditV2AgentLoop)
    loop.server_manager = FakeServer(tokenizer, turns)

    async def code_interpreter(code: str):
        """Always-successful fixture tool.

        Args:
            code: Python source.
        """
        del code
        return "4\n", 0.0, {"tool_success": 1.0, "tool_error": 0.0}

    tool = FunctionTool(
        name="code_interpreter",
        fn=code_interpreter,
        tool_schema=OpenAIFunctionToolSchema(**get_json_schema(code_interpreter)),
        is_async=True,
    )
    loop.tools = {"code_interpreter": tool}
    loop.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    return loop


def _run(turns: list[str]):
    loop = _v2_loop(turns)
    return asyncio.run(loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": QUESTION}]))


def test_v2_loop_is_v1_plus_repeat_and_boxed_conditions() -> None:
    output = _run(REPEAT_TURNS)
    ledger = output.extra_fields[LEDGER_KEY]
    assert ledger["schema_version"] == LEDGER_SCHEMA_VERSION_V2
    assert ledger["adoption_version"] == ADOPTION_VERSION_V2
    turns = ledger["assistant_turns"]
    assert len(turns) == 4
    assert [turn["credit_eligible"] for turn in turns] == [True, True, True, False]
    assert [turn["s_t_v1"] for turn in turns] == [1, 1, 1, 0]
    assert [turn["repeat_of_turn"] for turn in turns] == [None, 0, None, None]
    assert [turn["after_boxed"] for turn in turns] == [False, False, False, True]
    assert [turn["s_t"] for turn in turns] == [1, 0, 1, 0]
    assert [turn["adoption_status_v2"] for turn in turns] == ["adopted", "repeated_code", "adopted", "no_tool"]
    covered = [False] * len(output.response_ids)
    for turn in turns:
        assert output.response_mask[turn["policy_token_start"] : turn["policy_token_end"]] == [1] * (
            turn["policy_token_end"] - turn["policy_token_start"]
        )
        for index in range(turn["policy_token_start"], turn["policy_token_end"]):
            covered[index] = True
    assert covered == [bool(value) for value in output.response_mask]


def test_v2_loop_boxed_before_call_zeroes_later_success() -> None:
    single = _run(
        [
            'Compute.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}\n</tool_call>',
            "So \\boxed{4}.",
        ]
    )
    assert [turn["s_t"] for turn in single.extra_fields[LEDGER_KEY]["assistant_turns"]] == [1, 0]
    boxed_first = _run(
        [
            "The answer is \\boxed{4}. Checking.\n"
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}\n</tool_call>',
            'Once more.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1+3)"}}\n</tool_call>',
            "Confirmed \\boxed{4}.",
        ]
    )
    turns = boxed_first.extra_fields[LEDGER_KEY]["assistant_turns"]
    assert [turn["s_t_v1"] for turn in turns] == [1, 1, 0]
    assert [turn["after_boxed"] for turn in turns] == [False, True, True]
    assert [turn["repeat_of_turn"] for turn in turns] == [None, None, None]
    assert [turn["s_t"] for turn in turns] == [1, 0, 0]
    assert turns[1]["adoption_status_v2"] == "after_boxed"
