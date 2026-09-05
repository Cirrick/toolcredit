"""M8 / E5-v2 fixtures F–I (plans/M8.md §7); no GPU or model weights required.

Reuses the frozen v1 test helpers (batch construction, native GRPO call) without
modifying `rl/custom/test_turn_advantage.py`.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator

from rl.custom.test_turn_advantage import _batch, _constant_native, _ledger, _native
from rl.custom.turn_advantage import AUDIT_KEY
from rl.custom.turn_advantage_v2 import (
    ADOPTION_VERSION_V2,
    LEDGER_SCHEMA_VERSION_V2,
    VARIANT_CONSERVED_V2,
    compute_conserved_turn_credit_data,
    conserved_turn_corrections,
    make_conserved_compute_advantage_wrapper,
    v2_turn_conditions,
)


def _ledger_v2(
    trajectory_uid: str,
    s_values: list[int],
    lengths: list[int],
    *,
    eligible: list[bool] | None = None,
    s_v1: list[int] | None = None,
    repeats: list[int | None] | None = None,
    after_boxed: list[bool] | None = None,
) -> dict:
    """Tool turns (all but the last) followed by a final no-tool turn, with explicit token counts."""
    eligible = eligible if eligible is not None else [index < len(s_values) - 1 for index in range(len(s_values))]
    s_v1 = s_v1 if s_v1 is not None else list(s_values)
    repeats = repeats if repeats is not None else [None] * len(s_values)
    after_boxed = after_boxed if after_boxed is not None else [False] * len(s_values)
    turns = []
    cursor = 0
    for index, (s_t, n_t) in enumerate(zip(s_values, lengths, strict=True)):
        start, end = cursor, cursor + n_t
        tool_start = tool_end = None
        if eligible[index]:
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
                "parsed_call_count": int(eligible[index]),
                "parser_error_count": 0,
                "execution_success": int(eligible[index]),
                "execution_status": "ok" if eligible[index] else "no_tool",
                "adopted": s_v1[index],
                "adoption_status": "adopted" if s_v1[index] else ("not_found" if eligible[index] else "no_tool"),
                "adoption_evidence": None,
                "credit_eligible": eligible[index],
                "s_t_v1": s_v1[index],
                "repeat_of_turn": repeats[index],
                "after_boxed": after_boxed[index],
                "s_t": s_t,
                "visible_tool_response_text": "42" if eligible[index] else None,
            }
        )
    return {
        "schema_version": LEDGER_SCHEMA_VERSION_V2,
        "adoption_version": ADOPTION_VERSION_V2,
        "trajectory_uid": trajectory_uid,
        "response_length_unpadded": cursor,
        "trajectory_truncated": False,
        "termination_reason": "final_answer",
        "assistant_turns": turns,
    }


def _run_v2(data: DataProto, num_repeat: int) -> DataProto:
    return compute_conserved_turn_credit_data(
        _native,
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=True,
        config={},
        beta=0.5,
        enabled=True,
        proof_id="fixture-v2",
    )


def _native_copy(data: DataProto, num_repeat: int) -> DataProto:
    return _native(
        deepcopy(data),
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=True,
        config={},
    )


def test_pure_formula_fixture_f_and_gates() -> None:
    corrections, s_tilde, gated = conserved_turn_corrections(
        [1, 0, 1, 0], [10, 30, 20, 100], [True, True, True, False], beta=0.5, informative_group=True
    )
    assert s_tilde == pytest.approx(0.5) and gated is False
    assert corrections == pytest.approx([0.25, -0.25, 0.25, 0.0])
    assert sum(c * n for c, n in zip(corrections, [10, 30, 20, 100])) == pytest.approx(0.0, abs=1e-12)
    assert conserved_turn_corrections([1, 0], [5, 5], [True, False], beta=0.5, informative_group=True) == ([0.0, 0.0], None, True)
    assert conserved_turn_corrections([1, 0, 0], [5, 5, 5], [True, True, False], beta=0.5, informative_group=False) == ([0.0] * 3, None, True)
    uniform, s_tilde, gated = conserved_turn_corrections([1, 1, 0], [7, 9, 3], [True, True, False], beta=0.5, informative_group=True)
    assert uniform == [0.0, 0.0, 0.0] and s_tilde == 1.0 and gated is False


def test_pure_conditions_fixture_h_repeat_and_boxed() -> None:
    renamed = v2_turn_conditions(
        [["x = 1\nprint(x)"], ["# comment\ny=1 ;  print( y )"], []], ["a", "b", "\\boxed{1}"], [1, 1, 0], [True, True, False]
    )
    assert [item["repeat_of_turn"] for item in renamed] == [None, None, None]
    assert [item["s_t"] for item in renamed] == [1, 1, 0]
    same = v2_turn_conditions(
        [["x = 1\nprint(x)"], ["x=1\n\n# note\nprint(x)"], ["print(3)"]],
        ["a", "b \\boxed{1}", "c"],
        [1, 1, 1],
        [True, True, True],
    )
    assert same[1]["repeat_of_turn"] == 0 and same[1]["s_t"] == 0 and same[1]["s_t_v1"] == 1
    assert same[2]["after_boxed"] is True and same[2]["s_t"] == 0 and same[2]["s_t_v1"] == 1
    assert same[0]["s_t"] == 1


def test_fixture_f_conserved_hand_computed_redistribution() -> None:
    treated = _ledger_v2("r1", [1, 0, 1, 0], [10, 30, 20, 100])
    other = _ledger_v2("r2", [1, 0], [5, 5])
    data = _batch([treated, other], ["g", "g"], [1.0, 0.0])
    expected_native = _native_copy(data, 2)
    result = _run_v2(data, 2)
    native_scalar = float(expected_native.batch["advantages"][0][0])
    audit = result.non_tensor_batch[AUDIT_KEY][0]
    assert audit["variant"] == VARIANT_CONSERVED_V2 and audit["treated"] is True and audit["gated"] is False
    assert [turn["s_tilde"] for turn in audit["turns"]] == pytest.approx([0.5] * 4)
    assert [turn["correction"] for turn in audit["turns"]] == pytest.approx([0.25, -0.25, 0.25, 0.0])
    assert audit["conservation_residual"] == pytest.approx(0.0, abs=1e-9)
    for turn, value in zip(audit["turns"], [0.25, -0.25, 0.25, 0.0], strict=True):
        start, end = turn["policy_token_span"]
        assert torch.allclose(result.batch["advantages"][0, start:end], torch.tensor(native_scalar + value))
    mask = result.batch["response_mask"][0].bool()
    assert float(result.batch["advantages"][0][mask].mean()) == pytest.approx(native_scalar, abs=1e-6)
    assert torch.equal(result.batch["advantages"][1], expected_native.batch["advantages"][1])
    assert result.non_tensor_batch[AUDIT_KEY][1]["gated"] is True
    assert torch.all(result.batch["advantages"][~result.batch["response_mask"].bool()] == 0)
    assert torch.equal(result.batch["turn_ids"].ge(0), result.batch["response_mask"].bool())


def test_fixture_g_e3_identity_single_call_zero_variance_uniform_s() -> None:
    single = _batch([_ledger_v2("a", [1, 0], [4, 6]), _ledger_v2("b", [0, 0], [3, 3])], ["g", "g"], [1.0, 0.0])
    expected = _native_copy(single, 2)
    result = _run_v2(single, 2)
    assert torch.equal(result.batch["advantages"], expected.batch["advantages"])
    assert torch.equal(result.batch["returns"], expected.batch["returns"])
    zero = _batch(
        [_ledger_v2("c", [1, 0, 1, 0], [10, 30, 20, 100]), _ledger_v2("d", [0, 1, 0], [5, 5, 5])], ["g", "g"], [1.0, 1.0]
    )
    expected = _native_copy(zero, 2)
    result = _run_v2(zero, 2)
    assert torch.equal(result.batch["advantages"], expected.batch["advantages"])
    assert all(audit["gated"] and not audit["informative_group"] for audit in result.non_tensor_batch[AUDIT_KEY])
    uniform = _batch(
        [_ledger_v2("e", [1, 1, 1, 0], [10, 30, 20, 100]), _ledger_v2("f", [0, 0, 0], [5, 5, 5])], ["g", "g"], [1.0, 0.0]
    )
    expected = _native_copy(uniform, 2)
    result = _run_v2(uniform, 2)
    assert torch.equal(result.batch["advantages"], expected.batch["advantages"])
    assert result.non_tensor_batch[AUDIT_KEY][0]["treated"] is False
    assert result.non_tensor_batch[AUDIT_KEY][0]["gated"] is False


def test_fixture_h_v2_conditions_zero_out_repeats_and_post_boxed_calls() -> None:
    ledger = _ledger_v2(
        "h", [1, 0, 0, 0], [10, 10, 10, 10], s_v1=[1, 1, 1, 0], repeats=[None, 0, None, None], after_boxed=[False, False, True, True]
    )
    data = _batch([ledger, _ledger_v2("h2", [0, 0], [5, 5])], ["g", "g"], [1.0, 0.0])
    result = _run_v2(data, 2)
    audit = result.non_tensor_batch[AUDIT_KEY][0]
    assert [turn["s_t_v1"] for turn in audit["turns"]] == [1, 1, 1, 0]
    assert [turn["s_t"] for turn in audit["turns"]] == [1, 0, 0, 0]
    assert audit["turns"][0]["s_tilde"] == pytest.approx(1 / 3)
    assert [turn["correction"] for turn in audit["turns"]] == pytest.approx([1 / 3, -1 / 6, -1 / 6, 0.0])
    assert audit["conservation_residual"] == pytest.approx(0.0, abs=1e-9)


def test_fixture_i_mixed_batch_only_informative_groups_receive_credit() -> None:
    data = _batch(
        [
            _ledger_v2("a1", [1, 0, 0], [10, 10, 10]),
            _ledger_v2("a2", [0, 0], [5, 5]),
            _ledger_v2("b1", [1, 0, 0], [10, 10, 10]),
            _ledger_v2("b2", [0, 1, 0], [10, 10, 10]),
        ],
        ["A", "A", "B", "B"],
        [1.0, 0.0, 0.5, 0.5],
    )
    expected = _native_copy(data, 2)
    result = _run_v2(data, 2)
    audits = result.non_tensor_batch[AUDIT_KEY]
    assert [audit["treated"] for audit in audits] == [True, False, False, False]
    assert [audit["informative_group"] for audit in audits] == [True, True, False, False]
    for row in (1, 2, 3):
        assert torch.equal(result.batch["advantages"][row], expected.batch["advantages"][row])
    assert not torch.equal(result.batch["advantages"][0], expected.batch["advantages"][0])


def test_v2_fails_closed_on_beta_and_v1_schema() -> None:
    data = _batch([_ledger_v2("x", [1, 0, 0], [5, 5, 5]), _ledger_v2("y", [0, 0], [5, 5])], ["g", "g"], [1.0, 0.0])
    with pytest.raises(ValueError, match="frozen"):
        compute_conserved_turn_credit_data(_constant_native, deepcopy(data), num_repeat=2, beta=0.4, enabled=True, proof_id="bad")
    v1_schema = _batch([_ledger("p", [1]), _ledger("q", [0])], ["g", "g"], [1.0, 0.0])
    with pytest.raises(ValueError, match="schema"):
        compute_conserved_turn_credit_data(_constant_native, v1_schema, num_repeat=2, beta=0.5, enabled=True, proof_id="bad")
    wrapped = make_conserved_compute_advantage_wrapper(_constant_native, beta=0.5, enabled=True, proof_id="p")
    assert wrapped.__toolcredit_variant__ == VARIANT_CONSERVED_V2
    assert wrapped.__toolcredit_wrapper_proof__ == "p"
