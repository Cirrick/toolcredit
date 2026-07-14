"""M3: turn distilled traces into SFT training examples via the REAL verl agent loop.

Instead of re-implementing delta tokenization (and risking drift from what M4's
rollout produces), each trace is replayed through verl's ToolAgentLoop with a
fake server that emits the teacher's recorded assistant turns and a fake tool
that emits the recorded sandbox outputs. The loop then builds prompt_ids /
response_ids / response_mask exactly as it will during RL training
(same machinery M2's rl/custom/test_masking.py validates).

Training example: input_ids = prompt + response; labels = response tokens where
mask==1, everything else -100 (prompt, tool returns; padding is added by the
collator in train_sft.py).
"""

import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any

MODEL_PATH = os.path.expanduser("~/Qwen/Qwen3-1.7B")


def assistant_text_from_message(msg: dict[str, Any]) -> str:
    """Reconstruct the hermes-format generation text from an OpenAI-style message."""
    parts = [msg.get("content") or ""]
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        args = fn["arguments"]
        args_obj = json.loads(args) if isinstance(args, str) else args
        call = json.dumps({"name": fn["name"], "arguments": args_obj}, ensure_ascii=False)
        parts.append(f"<tool_call>\n{call}\n</tool_call>")
    return "\n".join(p for p in parts if p)


class _ReplayServer:
    """Feeds the recorded assistant turns back through the agent loop."""

    def __init__(self, tokenizer: Any, texts: list[str]):
        eos = tokenizer.eos_token_id
        self.outputs = [tokenizer.encode(t, add_special_tokens=False) + [eos] for t in texts]
        self.calls = 0

    async def generate(self, **kwargs):
        from verl.workers.rollout.replica import TokenOutput

        out = self.outputs[self.calls]
        self.calls += 1
        return TokenOutput(token_ids=out, stop_reason="completed", num_preempted=0)


def _build_replay_loop(tokenizer: Any, assistant_texts: list[str], tool_outputs: list[str], response_length: int):
    from transformers.utils import get_json_schema

    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.function_tool import FunctionTool
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.utils.chat_template import initialize_system_prompt

    state = {"n": 0}

    def code_interpreter(code: str) -> str:
        """Execute the code in the sandbox.

        Args:
            code: The code to be executed.

        Returns:
            The output of the code execution.
        """
        out = tool_outputs[state["n"]]
        state["n"] += 1
        return out

    tool = FunctionTool(
        name="code_interpreter",
        fn=code_interpreter,
        tool_schema=OpenAIFunctionToolSchema(**get_json_schema(code_interpreter)),
        is_async=False,
    )
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.apply_chat_template_kwargs = {"enable_thinking": False}
    loop.mm_processor_kwargs = {}
    loop.system_prompt = initialize_system_prompt(tokenizer, enable_thinking=False)
    loop.loop = asyncio.get_event_loop()
    loop.server_manager = _ReplayServer(tokenizer, assistant_texts)
    loop.tools = {"code_interpreter": tool}
    loop.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    loop.tool_parser = ToolParser.get_tool_parser("hermes", tokenizer)
    loop.tool_parser_name = "hermes"
    loop.max_user_turns = 16
    loop.max_assistant_turns = 16
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 100_000  # replay verbatim; truncation happened at generation
    loop.tool_response_truncate_side = "middle"
    loop.prompt_length = 1024
    loop.response_length = response_length
    loop.rollout_config = SimpleNamespace(prompt_length=1024, response_length=response_length)
    loop._replay_state = state  # exposes tool-output queue consumption for alignment asserts
    return loop


def trace_to_example(trace: dict[str, Any], tokenizer: Any, response_length: int = 3072) -> dict[str, list[int]]:
    """Replay one trace through ToolAgentLoop; return input_ids/labels/attention_mask."""
    messages = trace["messages"]
    assert messages[0]["role"] == "user", "trace must start with the user turn"
    assistant_texts = [assistant_text_from_message(m) for m in messages if m["role"] == "assistant"]
    tool_outputs = [m["content"] for m in messages if m["role"] == "tool"]

    loop = _build_replay_loop(tokenizer, assistant_texts, tool_outputs, response_length)
    output = asyncio.get_event_loop().run_until_complete(
        loop.run(sampling_params={}, raw_prompt=[messages[0]])
    )
    assert loop.server_manager.calls == len(assistant_texts), (
        f"replay consumed {loop.server_manager.calls}/{len(assistant_texts)} assistant turns "
        f"(trace {trace.get('id')}: tool-call parse drift?)"
    )
    assert loop._replay_state["n"] == len(tool_outputs), (
        f"replay consumed {loop._replay_state['n']}/{len(tool_outputs)} tool outputs "
        f"(trace {trace.get('id')}: misaligned tool replay — malformed args upstream?)"
    )
    input_ids = list(output.prompt_ids) + list(output.response_ids)
    labels = [-100] * len(output.prompt_ids) + [
        tok if m == 1 else -100 for tok, m in zip(output.response_ids, output.response_mask)
    ]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
