"""Print one real, model-generated veRL ToolAgentLoop trajectory.

This is a read-only debugging path for the model/checkpoint: it runs no trainer,
optimizer, GRPO grouping, or advantage computation.  It reuses the M2 pattern of
driving veRL's actual ToolAgentLoop directly, but replaces M2's scripted server
with local Hugging Face generation and executes tool calls through env/sandbox.py.

Run from the repository root:
  /home/jovyan/.conda/envs/toolcredit/bin/python scripts/debug_one_tool_agent_rollout.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from env.sandbox import prepare_tool_code, run_python  # noqa: E402
from rewards.verifier import verify_answer  # noqa: E402

DEFAULT_MODEL = PROJECT_DIR / "sft/checkpoints/qwen3-1.7b-sft"
QUESTION = (
    "Evaluate $\\left\\lceil3\\left(6-\\frac12\\right)\\right\\rceil$. "
    "Let's think step by step and output the final answer within \\boxed{}."
)
GOLD = "17"


@dataclass
class ToolRecord:
    code: str
    executed_code: str
    observation: str
    status: str


class LocalModelServer:
    """Small TokenOutput adapter: a real model replaces M2's FakeServer."""

    def __init__(self, model_path: str, tokenizer: Any, max_new_tokens: int):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        self.model.eval()
        self.turn_token_ids: list[list[int]] = []

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], **_: Any):
        from verl.workers.rollout.replica import TokenOutput

        eos_ids = list(
            dict.fromkeys(
                [self.tokenizer.eos_token_id] + list(sampling_params.get("stop_token_ids") or [])
            )
        )
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            generated = self.model.generate(
                inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        token_ids = generated[0, inputs.shape[1] :].tolist()
        self.turn_token_ids.append(token_ids)
        return TokenOutput(token_ids=token_ids, stop_reason="completed", num_preempted=0)


def make_tool(records: list[ToolRecord]):
    from verl.tools.function_tool import FunctionTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    def code_interpreter(code: str) -> str:
        """Execute Python code in the ToolCredit sandbox.

        Args:
            code: The Python source code to execute.

        Returns:
            The captured standard output and standard error.
        """
        executed_code = prepare_tool_code(code)
        result = run_python(executed_code)
        observation = result["stdout"] + result["stderr"]
        records.append(ToolRecord(code, executed_code, observation, result["status"]))
        return observation

    return FunctionTool(
        name="code_interpreter",
        fn=code_interpreter,
        tool_schema=OpenAIFunctionToolSchema(**get_json_schema(code_interpreter)),
        is_async=False,
    )


def build_loop(
    tokenizer: Any,
    server: LocalModelServer,
    records: list[ToolRecord],
    response_length: int,
    event_loop: asyncio.AbstractEventLoop,
):
    """Use the same direct ToolAgentLoop assembly exercised by M2's mask test."""
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.utils.chat_template import initialize_system_prompt

    tool = make_tool(records)
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.apply_chat_template_kwargs = {"enable_thinking": False}
    loop.mm_processor_kwargs = {}
    loop.system_prompt = initialize_system_prompt(tokenizer, enable_thinking=False)
    loop.loop = event_loop
    loop.server_manager = server
    loop.tools = {"code_interpreter": tool}
    loop.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    loop.tool_parser = ToolParser.get_tool_parser("hermes", tokenizer)
    loop.tool_parser_name = "hermes"
    loop.max_user_turns = 4
    loop.max_assistant_turns = 5
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 1024
    loop.tool_response_truncate_side = "middle"
    loop.prompt_length = 1024
    loop.response_length = response_length
    loop.rollout_config = SimpleNamespace(prompt_length=1024, response_length=response_length)
    return loop


def token_rows(output: Any, server: LocalModelServer, tokenizer: Any) -> list[dict[str, Any]]:
    """Assign each response token to an assistant action or following observation."""
    rows: list[dict[str, Any]] = []
    assistant_turn = 1
    for index, (token_id, mask) in enumerate(zip(output.response_ids, output.response_mask, strict=True)):
        if index and mask == 1 and output.response_mask[index - 1] == 0:
            assistant_turn += 1
        source = "MODEL_ACTION" if mask == 1 else "ENV_OBSERVATION"
        rows.append(
            {
                "index": index,
                "token": tokenizer.convert_ids_to_tokens(token_id),
                "id": token_id,
                "source": source,
                "mask": mask,
                "policy_loss": "YES" if mask == 1 else "NO",
                "turn": f"A{assistant_turn}" if mask == 1 else f"after A{assistant_turn}",
            }
        )
    expected_model_ids = [token for turn in server.turn_token_ids for token in turn]
    actual_model_ids = [row["id"] for row in rows if row["mask"] == 1]
    assert actual_model_ids == expected_model_ids, "model-token recording drifted from veRL response_mask"
    return rows


def render(
    model_path: str,
    tokenizer: Any,
    server: LocalModelServer,
    records: list[ToolRecord],
    output: Any,
) -> str:
    assistant_turns = [
        tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for ids in server.turn_token_ids
    ]
    final_answer = assistant_turns[-1] if assistant_turns else ""
    verdict = verify_answer(final_answer, GOLD, strict_boxed=True)
    rows = token_rows(output, server, tokenizer)

    lines = [
        "ONE REAL veRL ToolAgentLoop ROLLOUT",
        "=" * 80,
        f"Model: {model_path}",
        "Backend: local Hugging Face generation adapter (greedy); veRL ToolAgentLoop is real",
        "Training/GRPO/advantage computation: NOT RUN",
        "",
        "1. ORIGINAL USER PROMPT",
        QUESTION,
        "",
        "2-4. ASSISTANT TURNS, TOOL CALLS, AND ENVIRONMENT OBSERVATIONS",
    ]
    record_index = 0
    for turn_index, text in enumerate(assistant_turns, start=1):
        lines.extend(["", f"--- Assistant turn A{turn_index} [MODEL ACTION] ---", text])
        if turn_index < len(assistant_turns) and record_index < len(records):
            record = records[record_index]
            record_index += 1
            lines.extend(
                [
                    "",
                    f"--- Tool call after A{turn_index} ---",
                    "tool: code_interpreter",
                    "exact model-generated Python code:",
                    record.code,
                    "code actually executed after ToolCredit prepare_tool_code:",
                    record.executed_code,
                    f"status: {record.status}",
                    f"--- Environment observation after A{turn_index} [MASKED OUT] ---",
                    record.observation if record.observation else "<empty>",
                ]
            )

    lines.extend(
        [
            "",
            "5. FINAL ANSWER",
            final_answer,
            "",
            "6. VERIFIER REWARD",
            f"reward: {float(verdict['correct']):.1f}",
            f"verdict: {json.dumps(verdict, ensure_ascii=False)}",
            "",
            "7-8. TOKENIZED RESPONSE AND LOSS MASK",
            "Legend: MODEL_ACTION + mask=1 => participates in policy loss; "
            "ENV_OBSERVATION + mask=0 => masked out.",
            "Prompt tokens are outside response_ids and therefore do not appear in this table.",
            f"prompt_tokens={len(output.prompt_ids)} response_tokens={len(output.response_ids)} "
            f"model_action_tokens={sum(output.response_mask)} "
            f"masked_environment_tokens={len(output.response_mask) - sum(output.response_mask)}",
            "",
            "idx\ttoken_text\ttoken_id\tsource\tresponse/loss_mask\tpolicy_loss\tassistant_turn",
        ]
    )
    for row in rows:
        token_text = json.dumps(row["token"], ensure_ascii=False)
        lines.append(
            f"{row['index']}\t{token_text}\t{row['id']}\t{row['source']}\t"
            f"{row['mask']}\t{row['policy_loss']}\t{row['turn']}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--max-new-tokens-per-turn", type=int, default=256)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--output", default="debug_outputs/one_tool_agent_rollout.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records: list[ToolRecord] = []
    server = LocalModelServer(args.model, tokenizer, args.max_new_tokens_per_turn)
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    loop = build_loop(tokenizer, server, records, args.response_length, event_loop)
    try:
        output = event_loop.run_until_complete(
            loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": QUESTION}])
        )
    finally:
        event_loop.close()
    text = render(args.model, tokenizer, server, records, output)
    output_path = PROJECT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved full output to: {output_path}")


if __name__ == "__main__":
    main()
