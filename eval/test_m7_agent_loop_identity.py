"""Behavioral identity gate for M7's metadata-only loop."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

from rl.custom.test_masking import MODEL_PATH, QUESTION, _build_loop
from rl.custom.turn_advantage import LEDGER_KEY


@contextmanager
def _deterministic_timer(name, metrics):
    del name, metrics
    yield


def test_m7_metadata_loop_is_behaviorally_identical(monkeypatch) -> None:
    from transformers import AutoTokenizer

    import verl.experimental.agent_loop.tool_agent_loop as native_module
    from eval.generate import M7MetadataAgentLoop
    from rl.custom.tool_agent_loop import ToolCreditAgentLoop

    # Timing is nondeterministic audit telemetry, so suppress it in both otherwise
    # identical real state-machine replays and compare every output behavior field.
    monkeypatch.setattr(native_module, "simple_timer", _deterministic_timer)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    baseline_loop = _build_loop(tokenizer, loop_class=ToolCreditAgentLoop)
    metadata_loop = _build_loop(tokenizer, loop_class=M7MetadataAgentLoop)
    baseline = asyncio.run(
        baseline_loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": QUESTION}])
    )
    treatment = asyncio.run(
        metadata_loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": QUESTION}])
    )

    assert M7MetadataAgentLoop.__dict__.keys().isdisjoint(
        {"run", "_handle_generating_state", "_handle_processing_tools_state", "_call_tool"}
    )
    for field in (
        "prompt_ids", "response_ids", "response_mask", "multi_modal_data", "mm_processor_kwargs",
        "response_logprobs", "num_turns", "routed_experts",
    ):
        assert getattr(treatment, field) == getattr(baseline, field), field
    assert treatment.metrics == baseline.metrics
    baseline_extras = dict(baseline.extra_fields)
    treatment_extras = dict(treatment.extra_fields)
    ledger = treatment_extras.pop(LEDGER_KEY)
    assert baseline_extras == treatment_extras
    assert LEDGER_KEY not in baseline.extra_fields
    assert ledger["response_length_unpadded"] == len(treatment.response_ids)
    assert ledger["assistant_turns"]
