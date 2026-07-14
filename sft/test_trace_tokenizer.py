"""Tests for the trace->SFT-example replay tokenizer (M3). Run: python -m pytest sft/ -v"""

import json
import os

import pytest

from sft.trace_tokenizer import MODEL_PATH, assistant_text_from_message, trace_to_example

PROBE_TIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "probe", "trajectories_tir.jsonl")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH)


def _load_real_trace(min_tool_calls: int) -> dict:
    with open(PROBE_TIR, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["n_tool_calls"] >= min_tool_calls and not r["truncated"] and r["messages"]:
                return r
    pytest.skip("no suitable probe trajectory found")


def test_assistant_text_roundtrip() -> None:
    msg = {
        "role": "assistant",
        "content": "Let me compute.",
        "tool_calls": [{
            "id": "x", "type": "function",
            "function": {"name": "code_interpreter", "arguments": json.dumps({"code": "print(1)"})},
        }],
    }
    text = assistant_text_from_message(msg)
    assert text.startswith("Let me compute.")
    assert '<tool_call>\n{"name": "code_interpreter"' in text and text.endswith("</tool_call>")


def test_real_trace_single_call(tokenizer) -> None:
    trace = _load_real_trace(min_tool_calls=1)
    ex = trace_to_example(trace, tokenizer)
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    supervised = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    text = tokenizer.decode(supervised, skip_special_tokens=True)
    final = trace["messages"][-1]["content"]
    assert final[-40:] in text  # final answer text is supervised
    # tool returns live in the MASKED part of the response region (between two
    # supervised runs), wrapped in <tool_response> markers by the chat template
    start = next(i for i, l in enumerate(ex["labels"]) if l != -100)
    masked_resp = tokenizer.decode(
        [t for t, l in zip(ex["input_ids"][start:], ex["labels"][start:]) if l == -100]
    )
    assert "<tool_response>" in masked_resp
    first_tool = next(m for m in trace["messages"] if m["role"] == "tool")
    if first_tool["content"].strip():
        assert first_tool["content"].strip()[:20] in masked_resp


def test_real_trace_multi_call(tokenizer) -> None:
    trace = _load_real_trace(min_tool_calls=2)
    ex = trace_to_example(trace, tokenizer)
    labels = ex["labels"]
    # question (prompt) fully masked
    q30 = trace["messages"][0]["content"][:30]
    prompt_text = tokenizer.decode(
        [t for t, l in zip(ex["input_ids"], labels) if l == -100], skip_special_tokens=True
    )
    assert q30 in prompt_text
    # mask alternates: some -100 runs inside the response region (tool returns)
    resp_labels = labels[next(i for i, l in enumerate(labels) if l != -100):]
    assert any(l == -100 for l in resp_labels), "expected masked tool-return segment inside response"
