"""Loss-masking unit test against verl 0.8.0 ToolAgentLoop (M2, PLAN §6.4).

The mask under test is built in
`verl/experimental/agent_loop/tool_agent_loop.py` (delta-based tokenization):
  - `_handle_generating_state`:       response_mask += [1] * len(model_tokens)
  - `_handle_processing_tools_state`: response_mask += [0] * len(tool_tokens)
and padding zeros come from `_agent_loop_postprocess`:
  response_mask = pad(mask) * response_attention_mask.

Strategy: drive the REAL ToolAgentLoop state machine with the real Qwen3-1.7B
tokenizer + real hermes ToolParser + a real FunctionTool, mocking only the LLM
server (scripted token outputs for a 2-tool-call trajectory), then assert on
the mask/token alignment. verl code is exercised, never modified (禁止事项 #1).

Run: python -m pytest rl/custom/test_masking.py -v
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

MODEL_PATH = os.path.expanduser("~/Qwen/Qwen3-1.7B")
QUESTION = "What is 2+2? Output the final answer within \\boxed{}."
TOOL_OUTPUT_1 = "SANDBOX_OUTPUT_FIRST 4\n"
TOOL_OUTPUT_2 = "SANDBOX_OUTPUT_SECOND 4\n"
ASSISTANT_TURNS = [
    'Let me compute this.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}\n</tool_call>',
    'Double-checking.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1+3)"}}\n</tool_call>',
    "Both runs agree. The final answer is \\boxed{4}.",
]


class FakeServer:
    """Replays scripted assistant token outputs in order."""

    def __init__(self, tokenizer, texts: list[str]):
        eos = tokenizer.eos_token_id
        self.outputs = [tokenizer.encode(t, add_special_tokens=False) + [eos] for t in texts]
        self.calls = 0

    async def generate(self, **kwargs):
        from verl.workers.rollout.replica import TokenOutput

        out = self.outputs[self.calls]
        self.calls += 1
        return TokenOutput(token_ids=out, stop_reason="completed", num_preempted=0)


def _sandbox_outputs():
    outputs = [TOOL_OUTPUT_1, TOOL_OUTPUT_2]
    state = {"n": 0}

    def code_interpreter(code: str) -> str:
        """Execute the code in the sandbox.

        Args:
            code: The code to be executed.

        Returns:
            The output of the code execution.
        """
        out = outputs[state["n"]]
        state["n"] += 1
        return out

    return code_interpreter


def _build_loop(tokenizer, response_length: int = 2048, loop_class=None):
    """Assemble a ToolAgentLoop without the trainer-level dependency injection."""
    from transformers.utils import get_json_schema

    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.function_tool import FunctionTool
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.utils.chat_template import initialize_system_prompt

    fn = _sandbox_outputs()
    tool = FunctionTool(
        name="code_interpreter",
        fn=fn,
        tool_schema=OpenAIFunctionToolSchema(**get_json_schema(fn)),
        is_async=False,
    )

    loop_class = loop_class or ToolAgentLoop
    loop = loop_class.__new__(loop_class)
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.apply_chat_template_kwargs = {"enable_thinking": False}
    loop.mm_processor_kwargs = {}
    loop.system_prompt = initialize_system_prompt(tokenizer, enable_thinking=False)
    loop.loop = asyncio.get_event_loop()
    loop.server_manager = FakeServer(tokenizer, ASSISTANT_TURNS)
    loop.tools = {"code_interpreter": tool}
    loop.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    loop.tool_parser = ToolParser.get_tool_parser("hermes", tokenizer)
    loop.tool_parser_name = "hermes"
    loop.max_user_turns = 8
    loop.max_assistant_turns = 8
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 1024
    loop.tool_response_truncate_side = "middle"
    loop.prompt_length = 1024
    loop.response_length = response_length
    loop.rollout_config = SimpleNamespace(prompt_length=1024, response_length=response_length)
    return loop


@pytest.fixture(scope="module")
def rollout():
    from transformers import AutoTokenizer

    from rl.custom.tool_agent_loop import ToolCreditAgentLoop

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    loop = _build_loop(tokenizer, loop_class=ToolCreditAgentLoop)
    output = asyncio.get_event_loop().run_until_complete(
        loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": QUESTION}])
    )
    return tokenizer, output


@pytest.fixture(scope="module")
def nomask_rollout():
    from transformers import AutoTokenizer

    from rl.custom.no_mask_tool_agent_loop import ToolCreditNoMaskAgentLoop

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    loop = _build_loop(tokenizer, loop_class=ToolCreditNoMaskAgentLoop)
    output = asyncio.get_event_loop().run_until_complete(
        loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": QUESTION}])
    )
    return tokenizer, output


def test_mask_alignment(rollout) -> None:
    _, output = rollout
    assert len(output.response_mask) == len(output.response_ids)
    assert set(output.response_mask) == {0, 1}  # both segments present
    assert output.num_turns == 6  # 3 assistant + 2 tool + 1 initial user


def test_model_tokens_masked_one(rollout) -> None:
    """mask==1 tokens must decode to exactly the scripted model outputs."""
    tokenizer, output = rollout
    gen_ids = [t for t, m in zip(output.response_ids, output.response_mask) if m == 1]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    for turn in ASSISTANT_TURNS:
        assert turn in gen_text
    assert "SANDBOX_OUTPUT" not in gen_text  # no tool output leaks into loss


def test_tool_tokens_masked_zero(rollout) -> None:
    """mask==0 tokens must contain both tool outputs and no model text."""
    tokenizer, output = rollout
    tool_ids = [t for t, m in zip(output.response_ids, output.response_mask) if m == 0]
    tool_text = tokenizer.decode(tool_ids, skip_special_tokens=True)
    assert TOOL_OUTPUT_1.strip() in tool_text
    assert TOOL_OUTPUT_2.strip() in tool_text
    assert "\\boxed{4}" not in tool_text
    assert "Double-checking" not in tool_text


def test_prompt_not_in_response(rollout) -> None:
    """Prompt tokens live outside response_ids/response_mask entirely."""
    tokenizer, output = rollout
    prompt_text = tokenizer.decode(output.prompt_ids, skip_special_tokens=True)
    response_text = tokenizer.decode(output.response_ids, skip_special_tokens=True)
    assert QUESTION in prompt_text
    assert QUESTION not in response_text


def test_padding_masked_zero(rollout) -> None:
    """Replicates _agent_loop_postprocess: padded response_mask must be 0 on padding.

    verl pads the mask list with tokenizer.pad (which inserts pad_token_id, NOT 0!)
    and relies on multiplying by the response attention mask to zero the padding —
    this test pins that contract."""
    tokenizer, output = rollout
    max_len = len(output.response_ids) + 17
    tokenizer.padding_side = "right"
    padded_mask = tokenizer.pad(
        {"input_ids": output.response_mask}, padding="max_length", max_length=max_len, return_tensors="pt"
    )["input_ids"]
    padded_resp = tokenizer.pad(
        {"input_ids": output.response_ids},
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if padded_mask.dim() == 1:  # verl's _pad_token_ids unsqueezes 1-D results the same way
        padded_mask = padded_mask.unsqueeze(0)
        padded_resp = {k: v.unsqueeze(0) for k, v in padded_resp.items()}
    final_mask = padded_mask * padded_resp["attention_mask"]
    assert final_mask.shape[-1] == max_len
    assert final_mask[0, -17:].sum() == 0  # padding contributes zero loss weight
    assert final_mask[0, : len(output.response_mask)].tolist() == output.response_mask


def test_e6_changes_only_tool_return_mask_and_adds_audit(rollout, nomask_rollout) -> None:
    _, baseline = rollout
    _, treatment = nomask_rollout
    assert treatment.response_ids == baseline.response_ids
    assert treatment.prompt_ids == baseline.prompt_ids
    assert treatment.num_turns == baseline.num_turns
    # Timing fields are measured independently in the two deterministic replays.
    assert treatment.metrics.num_preempted == baseline.metrics.num_preempted
    assert treatment.extra_fields["tool_call_counts"] == baseline.extra_fields["tool_call_counts"]
    assert treatment.extra_fields["tool_success_count"] == baseline.extra_fields["tool_success_count"]
    assert treatment.extra_fields["tool_error_count"] == baseline.extra_fields["tool_error_count"]
    assert treatment.extra_fields["tool_parse_error_count"] == baseline.extra_fields["tool_parse_error_count"]
    assert baseline.response_mask.count(0) > 0
    assert treatment.response_mask == [1] * len(treatment.response_ids)
    assert treatment.extra_fields["original_policy_token_count"] == sum(baseline.response_mask)
    assert treatment.extra_fields["original_tool_return_token_count"] == baseline.response_mask.count(0)
    assert treatment.extra_fields["nomask_loss_token_count"] == len(treatment.response_ids)


def test_e6_padding_remains_outside_loss(nomask_rollout) -> None:
    tokenizer, output = nomask_rollout
    max_len = len(output.response_ids) + 11
    tokenizer.padding_side = "right"
    padded_mask = tokenizer.pad(
        {"input_ids": output.response_mask}, padding="max_length", max_length=max_len, return_tensors="pt"
    )["input_ids"]
    padded_resp = tokenizer.pad(
        {"input_ids": output.response_ids},
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if padded_mask.dim() == 1:
        padded_mask = padded_mask.unsqueeze(0)
        padded_resp = {key: value.unsqueeze(0) for key, value in padded_resp.items()}
    final_mask = padded_mask * padded_resp["attention_mask"]
    assert final_mask[0, : len(output.response_ids)].tolist() == [1] * len(output.response_ids)
    assert final_mask[0, -11:].sum() == 0


def test_truncation_keeps_alignment() -> None:
    """With a tiny response budget the loop truncates; ids and mask stay aligned."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    loop = _build_loop(tokenizer, response_length=48)
    output = asyncio.get_event_loop().run_until_complete(
        loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": QUESTION}])
    )
    assert len(output.response_ids) <= 48
    assert len(output.response_mask) == len(output.response_ids)
