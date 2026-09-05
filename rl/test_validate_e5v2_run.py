"""E5-v2 audit validator accepts runtime output and rejects tampering."""

from __future__ import annotations

from copy import deepcopy

import pytest
from verl.trainer.ppo.core_algos import AdvantageEstimator

from rl.custom.test_turn_advantage import _batch, _native
from rl.custom.test_turn_advantage_v2 import _ledger_v2
from rl.custom.turn_advantage import AUDIT_KEY
from rl.custom.turn_advantage_v2 import compute_conserved_turn_credit_data
from rl.validate_e5v2_run import validate_audit_v2


def _audits():
    data = _batch(
        [
            _ledger_v2("a1", [1, 0, 0], [10, 20, 10]),
            _ledger_v2("a2", [0, 0], [5, 5]),
            _ledger_v2("b1", [1, 0, 0], [10, 10, 10]),
            _ledger_v2("b2", [0, 1, 0], [10, 10, 10]),
        ],
        ["A", "A", "B", "B"],
        [1.0, 0.0, 0.5, 0.5],
    )
    result = compute_conserved_turn_credit_data(
        _native,
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config={},
        beta=0.5,
        enabled=True,
        proof_id="e5v2-wrapper-v1:test",
    )
    return list(result.non_tensor_batch[AUDIT_KEY])


def test_runtime_audits_pass_validator() -> None:
    for audit in _audits():
        validate_audit_v2(audit)


def test_validator_rejects_tampered_correction_and_flags() -> None:
    treated = _audits()[0]
    bad = deepcopy(treated)
    bad["turns"][0]["correction"] += 0.01
    bad["turns"][0]["A_turn"] += 0.01
    with pytest.raises(ValueError, match="formula"):
        validate_audit_v2(bad)
    bad = deepcopy(treated)
    bad["turns"][2]["s_t"] = 1
    with pytest.raises(ValueError):
        validate_audit_v2(bad)
    bad = deepcopy(treated)
    bad["gated"] = True
    with pytest.raises(ValueError, match="gated"):
        validate_audit_v2(bad)
    bad = deepcopy(treated)
    bad["wrapper_proof"] = "e5-wrapper-v1:old"
    with pytest.raises(ValueError, match="proof"):
        validate_audit_v2(bad)
