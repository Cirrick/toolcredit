"""Deterministic M6 Tier-A fixtures; no GPU or model weights required."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.protocol import DataProto
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.core_algos import AdvantageEstimator

from rl.custom.turn_advantage import (
    ADOPTION_VERSION,
    AUDIT_KEY,
    LEDGER_KEY,
    LEDGER_SCHEMA_VERSION,
    classify_adoption,
    compute_turn_credit_data,
    truncate_tool_response_text,
)


def _ledger(trajectory_uid: str, s_values: list[int]) -> dict:
    turns = []
    cursor = 0
    for index, s_t in enumerate(s_values):
        start, end = cursor, cursor + 2
        tool_start = tool_end = None
        if index < len(s_values) - 1:
            tool_start, tool_end = end, end + 1
            cursor = tool_end
        else:
            cursor = end
        turns.append(
            {
                "turn_index": index,
                "policy_token_start": start,
                "policy_token_end": end,
                "tool_token_start": tool_start,
                "tool_token_end": tool_end,
                "execution_success": s_t,
                "execution_status": "ok" if s_t else "execution_error",
                "adopted": s_t,
                "adoption_status": "adopted" if s_t else "execution_failed",
                "adoption_evidence": None,
                "s_t": s_t,
                "visible_tool_response_text": "42" if s_t else "error",
            }
        )
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "adoption_version": ADOPTION_VERSION,
        "trajectory_uid": trajectory_uid,
        "response_length_unpadded": cursor,
        "trajectory_truncated": False,
        "termination_reason": "final_answer",
        "assistant_turns": turns,
    }


def _batch(ledgers: list[dict], uids: list[str], rewards: list[float]) -> DataProto:
    width = max(int(ledger["response_length_unpadded"]) for ledger in ledgers) + 2
    masks = torch.zeros((len(ledgers), width), dtype=torch.long)
    token_rewards = torch.zeros((len(ledgers), width), dtype=torch.float32)
    for row, (ledger, reward) in enumerate(zip(ledgers, rewards, strict=True)):
        for turn in ledger["assistant_turns"]:
            masks[row, turn["policy_token_start"] : turn["policy_token_end"]] = 1
        last_policy = int(ledger["assistant_turns"][-1]["policy_token_end"]) - 1
        token_rewards[row, last_policy] = reward
    batch = TensorDict(
        {
            "responses": torch.arange(width).repeat(len(ledgers), 1),
            "response_mask": masks,
            "token_level_rewards": token_rewards.clone(),
            "token_level_scores": token_rewards.clone(),
        },
        batch_size=len(ledgers),
    )
    ledger_array = np.empty(len(ledgers), dtype=object)
    ledger_array[:] = ledgers
    return DataProto(
        batch=batch,
        non_tensor_batch={"uid": np.array(uids, dtype=object), LEDGER_KEY: ledger_array},
    )


def _native(data: DataProto, *args, **kwargs) -> DataProto:
    return ray_trainer.compute_advantage(data, *args, **kwargs)


def _constant_native(data: DataProto, *args, **kwargs) -> DataProto:
    del args, kwargs
    mask = data.batch["response_mask"]
    data.batch["advantages"] = torch.zeros_like(mask, dtype=torch.float32)
    data.batch["returns"] = torch.zeros_like(mask, dtype=torch.float32)
    return data


def test_fixture_a_crossed_good_bad_turns_and_token_mapping() -> None:
    data = _batch(
        [_ledger("r1", [1, 0]), _ledger("r2", [0, 1])],
        ["group", "group"],
        [1.0, 0.0],
    )
    result = compute_turn_credit_data(
        _native,
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config={},
        beta=0.5,
        enabled=True,
        proof_id="fixture-a",
    )
    native = (0.5 / (0.5**0.5 + 1e-6))
    expected = [[native + 0.25, native - 0.25], [-native - 0.25, -native + 0.25]]
    for row, row_expected in enumerate(expected):
        audit = result.non_tensor_batch[AUDIT_KEY][row]
        assert [turn["s_bar_t"] for turn in audit["turns"]] == [0.5, 0.5]
        for turn, value in zip(audit["turns"], row_expected, strict=True):
            start, end = turn["policy_token_span"]
            assert torch.allclose(result.batch["advantages"][row, start:end], torch.tensor(value))
            assert turn["A_turn"] == pytest.approx(value)
    assert torch.all(result.batch["advantages"][~result.batch["response_mask"].bool()] == 0)
    assert torch.equal(result.batch["turn_ids"].ge(0), result.batch["response_mask"].bool())


def test_fixture_b_disabled_is_exact_native_identity() -> None:
    data = _batch([_ledger("r1", [1]), _ledger("r2", [0])], ["g", "g"], [1.0, 0.0])
    expected = _native(
        deepcopy(data),
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config={},
    )
    actual = compute_turn_credit_data(
        _native,
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config={},
        beta=0.5,
        enabled=False,
        proof_id="disabled",
    )
    assert torch.equal(actual.batch["advantages"], expected.batch["advantages"])
    assert torch.equal(actual.batch["returns"], expected.batch["returns"])
    assert set(actual.batch.keys()) == set(expected.batch.keys())
    assert set(actual.non_tensor_batch) == set(expected.non_tensor_batch)


def test_fixture_c_variable_turn_denominators() -> None:
    data = _batch(
        [_ledger("r2", [1, 0]), _ledger("r3", [0, 1, 1]), _ledger("r4", [1, 1, 0, 1])],
        ["g", "g", "g"],
        [0.0, 0.0, 0.0],
    )
    result = compute_turn_credit_data(
        _constant_native,
        data,
        num_repeat=3,
        beta=0.5,
        enabled=True,
        proof_id="fixture-c",
    )
    audits = result.non_tensor_batch[AUDIT_KEY]
    assert [turn["position_denominator"] for turn in audits[0]["turns"]] == [3, 3]
    assert [turn["position_denominator"] for turn in audits[1]["turns"]] == [3, 3, 2]
    assert [turn["position_denominator"] for turn in audits[2]["turns"]] == [3, 3, 2, 1]
    assert [turn["s_bar_t"] for turn in audits[2]["turns"]] == pytest.approx([2 / 3, 2 / 3, 0.5, 1.0])
    assert audits[2]["turns"][-1]["correction"] == 0.0


def test_fixture_d_success_but_unused_and_visible_only() -> None:
    unused = classify_adoption("result=42", ["The tool returned 42."], [])
    assert unused.status == "mentioned_only"
    assert unused.s_t == 0
    raw_only = classify_adoption("visible result 7", ["Therefore the answer is 42."], [])
    assert raw_only.status == "not_found"
    assert raw_only.s_t == 0
    adopted = classify_adoption("42", ["Therefore the final answer is \\boxed{42}."], [])
    assert adopted.status == "adopted"
    assert adopted.s_t == 1


@pytest.mark.parametrize(
    ("visible", "later"),
    [
        ("-12", "Thus x = -12."),
        ("1/3", "Therefore \\boxed{\\frac{1}{3}}."),
        ("1.2e3", "Hence the answer is 1200."),
        ("0.3333333", "So x = 0.3333334."),
    ],
)
def test_adoption_numeric_forms(visible: str, later: str) -> None:
    assert classify_adoption(visible, [later]).s_t == 1


def test_fixture_e_tool_text_truncation_and_alignment_failures() -> None:
    raw = "A" * 256 + "HIDDEN42" + "B" * 256
    visible = truncate_tool_response_text(raw, 512, "middle")
    assert visible == raw[:256] + "...(truncated)..." + raw[-256:]
    assert len(visible) == 529
    assert classify_adoption(visible, ["Therefore x = 42."]).s_t == 0

    bad = _ledger("bad", [1])
    bad["assistant_turns"][0]["policy_token_end"] += 1
    data = _batch([bad, _ledger("other", [0])], ["g", "g"], [1.0, 0.0])
    with pytest.raises(ValueError, match="policy.*span"):
        compute_turn_credit_data(
            _constant_native,
            data,
            num_repeat=2,
            beta=0.5,
            enabled=True,
            proof_id="bad-span",
        )


def test_duplicate_and_incomplete_uid_groups_fail_closed() -> None:
    duplicate = _batch([_ledger("same", [1]), _ledger("same", [0])], ["g", "g"], [1.0, 0.0])
    with pytest.raises(ValueError, match="duplicate trajectory_uid"):
        compute_turn_credit_data(
            _constant_native,
            duplicate,
            num_repeat=2,
            beta=0.5,
            enabled=True,
            proof_id="duplicate",
        )
    incomplete = _batch([_ledger("one", [1])], ["g"], [1.0])
    with pytest.raises(ValueError, match="num_repeat=2"):
        compute_turn_credit_data(
            _constant_native,
            incomplete,
            num_repeat=2,
            beta=0.5,
            enabled=True,
            proof_id="incomplete",
        )
