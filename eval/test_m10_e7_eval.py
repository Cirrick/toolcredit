"""M10 / protocol v6 evaluator fixtures (plans/M10.md §7, §9).

Covered here, all without a GPU or any generated trajectory:

* the 7,600-key universe and its disjointness from every earlier protocol's universe;
* the v6 raw-row validator: accepts a frozen-shaped row, rejects role leakage, seed drift,
  hash tampering, out-of-setting sample indices and infrastructure failures;
* the two compute axes: the paired question-level AUC bootstrap and the fixed-100 panel guard;
* the plan §9 four-cell matrix, cell by cell, including the "CI crosses zero" downgrades;
* the v5 -> v6 semantic diff: accepts the real additive shape, rejects each way of smuggling a
  semantic change through it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from eval import m10_e7_eval as m10
from eval.build_diagnostic_freeze_v6 import DYNAMIC_FILTERING_SEMANTICS, PAIRINGS
from eval.checkpoints_v6 import E7_ROLE_NAMES, ROLE_EQUAL_COMPUTE, ROLE_STEP_200
from eval.generate import derive_sample_seed, expected_keys, load_panel_questions
from eval.verify_diagnostic_freeze_v6_semantics import (
    EXPECTED_CHANGED_TOP_LEVEL,
    FROZEN_SEMANTIC_FIELDS,
    semantic_diff,
)

E3_ROLE = "e3_grpo_baseline_step_200"


# --------------------------------------------------------------------------- #
# key universe
# --------------------------------------------------------------------------- #


def test_key_universe_is_7600_and_disjoint_from_the_frozen_universes() -> None:
    keys = m10.expected_keys_v6()
    assert len(keys) == 7600 == len(set(keys))
    assert {key[0] for key in keys} == set(E7_ROLE_NAMES)
    per_role = {role: sum(1 for key in keys if key[0] == role) for role in E7_ROLE_NAMES}
    assert per_role == {role: 3800 for role in E7_ROLE_NAMES}
    frozen = set(expected_keys())
    assert frozen.isdisjoint(set(keys))
    greedy = [key for key in keys if key[1] == "greedy"]
    assert len(greedy) == 2 * 760 and all(key[3] == 0 for key in greedy)
    with pytest.raises(ValueError, match="unknown role"):
        m10.expected_keys_v6(("not_a_role",))


def test_role_universe_covers_every_earlier_protocol_role() -> None:
    assert len(m10.ROLE_NAMES_V6) == len(set(m10.ROLE_NAMES_V6)) == 11
    for role in (E3_ROLE, "raw_base_model", "e5v2_conserved_credit_step_200", "e3notool_grpo_step_200", *E7_ROLE_NAMES):
        assert role in m10.ROLE_NAMES_V6


# --------------------------------------------------------------------------- #
# raw-row validation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def question() -> dict[str, Any]:
    return load_panel_questions()[0]


def _row(question: dict[str, Any], role: str = ROLE_STEP_200, setting: str = "greedy", index: int = 0) -> dict[str, Any]:
    response = "reasoning then \\boxed{2}"
    return {
        "protocol_version": m10.PROTOCOL_VERSION_V6,
        "schema_version": "toolcredit_m7_trajectory_schema_v1",
        "evaluator_version": m10.EVALUATOR_VERSION,
        "freeze_manifest_sha256": "0" * 64,
        "checkpoint_role": role,
        "checkpoint_identity": {"role": role},
        "evaluation_setting": setting,
        "question_id": question["question_id"],
        "source": question["source"],
        "split": question["split"],
        "stratum": question["stratum"],
        "question_hash": question["question_sha256_nfkc_whitespace"],
        "sample_index": index,
        "sample_seed": derive_sample_seed(str(question["question_id"]), index),
        "prompt": "q",
        "prompt_hash": hashlib.sha256(b"q").hexdigest(),
        "messages": [],
        "response": response,
        "response_token_count": 5,
        "termination_reason": "final_answer",
        "tool_metadata": {},
        "infrastructure_status": "ok",
        "raw_output_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
    }


def test_validator_accepts_both_e7_roles(question: dict[str, Any]) -> None:
    for role in E7_ROLE_NAMES:
        m10.validate_raw_generation_row_v6(_row(question, role=role), question)
    m10.validate_raw_generation_row_v6(_row(question, setting="sampled", index=3), question)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"checkpoint_role": "some_other_model"}, "unknown checkpoint role"),
        ({"infrastructure_status": "server_error"}, "infrastructure failures"),
        ({"sample_seed": 12345}, "sample seed drifted"),
        ({"raw_output_sha256": "0" * 64}, "response hash mismatch"),
        ({"question_hash": "0" * 64}, "question hash mismatch"),
        ({"sample_index": 7}, "outside its frozen setting"),
    ],
)
def test_validator_fails_closed(question: dict[str, Any], mutation: dict[str, Any], message: str) -> None:
    row = {**_row(question), **mutation}
    with pytest.raises(ValueError, match=message):
        m10.validate_raw_generation_row_v6(row, question)


def test_validator_requires_every_frozen_field(question: dict[str, Any]) -> None:
    row = _row(question)
    del row["termination_reason"]
    with pytest.raises(ValueError, match="missing fields"):
        m10.validate_raw_generation_row_v6(row, question)


def test_resume_shard_rejects_duplicates_and_foreign_questions(tmp_path: Path, question: dict[str, Any]) -> None:
    path = tmp_path / "shard.jsonl"
    row = _row(question)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate resume key"):
        m10.validated_resume_rows_v6(path, {question["question_id"]: question})
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown question ID"):
        m10.validated_resume_rows_v6(path, {})
    assert len(m10.validated_resume_rows_v6(path, {question["question_id"]: question})) == 1
    assert m10.validated_resume_rows_v6(tmp_path / "absent.jsonl", {}) == {}


# --------------------------------------------------------------------------- #
# compute axes
# --------------------------------------------------------------------------- #


def _curves(values: dict[int, float], n: int = 100) -> dict[str, dict[int, float]]:
    """A fixed-100 panel where exactly ``round(acc*100)`` questions are correct at each step."""
    curves: dict[str, dict[int, float]] = {f"q{i:03d}": {} for i in range(n)}
    for step, accuracy in values.items():
        correct = round(accuracy * n)
        for i, qid in enumerate(sorted(curves)):
            curves[qid][step] = 1.0 if i < correct else 0.0
    return curves


def test_auc_delta_bootstrap_is_paired_and_brackets_the_point_estimate() -> None:
    flat = {step: 0.60 for step in m10.VALIDATION_STEPS}
    better = {step: 0.60 + 0.10 * (step / 200) for step in m10.VALIDATION_STEPS}
    result = m10.auc_delta_bootstrap(_curves(flat), _curves(better), replicates=200, seed=1)
    assert result["point_estimate_pt"] == pytest.approx(5.0, abs=0.5)
    lower, upper = result["ci_pt"]
    assert lower <= result["point_estimate_pt"] <= upper
    assert result["replicates"] == 200 and result["seed"] == 1
    identical = m10.auc_delta_bootstrap(_curves(flat), _curves(flat), replicates=100, seed=1)
    assert identical["point_estimate_pt"] == pytest.approx(0.0)
    assert identical["ci_pt"] == [pytest.approx(0.0), pytest.approx(0.0)]


def test_auc_bootstrap_requires_the_same_panel() -> None:
    a = _curves({step: 0.6 for step in m10.VALIDATION_STEPS})
    b = _curves({step: 0.6 for step in m10.VALIDATION_STEPS})
    b["extra"] = dict(b["q000"])
    with pytest.raises(ValueError, match="same fixed-100 panel"):
        m10.auc_delta_bootstrap(a, b, replicates=5)


def test_per_question_curves_reject_an_incomplete_panel(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "predictions/validation").mkdir(parents=True)
    for step in m10.VALIDATION_STEPS:
        rows = [{"sample_id": f"q{i:03d}", "acc": 1.0} for i in range(99)]
        (run / "predictions/validation" / f"{step}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="100 rows"):
        m10.per_question_validation_curves(run)


# --------------------------------------------------------------------------- #
# plan §9 four-cell matrix
# --------------------------------------------------------------------------- #


def _step(point_pt: float, lo: float, hi: float) -> dict[str, Any]:
    return {"point_estimate_pt": point_pt, "ci_pt": [lo, hi]}


def _compute(point: float, lo: float, hi: float) -> dict[str, Any]:
    return {"point_estimate": point, "lower": lo, "upper": hi}


@pytest.mark.parametrize(
    "step,compute,case",
    [
        (_step(4.0, 2.0, 6.0), _compute(0.005, -0.02, 0.03), "A"),
        (_step(4.0, 2.0, 6.0), _compute(0.035, 0.01, 0.06), "B"),
        (_step(0.5, -1.5, 2.5), _compute(0.005, -0.02, 0.03), "C"),
        (_step(4.0, 2.0, 6.0), _compute(-0.04, -0.07, -0.01), "D"),
        (_step(-4.0, -6.0, -2.0), _compute(0.005, -0.02, 0.03), "outside_preregistered_table"),
        # A 3pt effect whose CI crosses zero is "unchanged", not "positive".
        (_step(3.0, -0.5, 6.5), _compute(0.005, -0.02, 0.03), "C"),
        # A significant but sub-threshold effect is also "unchanged".
        (_step(1.5, 1.0, 2.0), _compute(0.005, -0.02, 0.03), "C"),
    ],
)
def test_preregistered_matrix_cell_by_cell(step: dict[str, Any], compute: dict[str, Any], case: str) -> None:
    result = m10.classify_preregistered_case(step, compute)
    assert result["case"] == case
    assert result["threshold_pt"] == 2.0
    assert result["preregistered_writeup"]
    assert "both axes" in result["guardrail"]


def test_matrix_reports_both_directions_and_the_compute_axis_dominates() -> None:
    result = m10.classify_preregistered_case(_step(9.0, 7.0, 11.0), _compute(-0.05, -0.08, -0.02))
    assert result["case"] == "D"
    assert result["delta_step_direction"] == "positive" and result["delta_compute_direction"] == "negative"
    assert result["delta_compute_pt"] == pytest.approx(-5.0)


# --------------------------------------------------------------------------- #
# v5 -> v6 semantic diff
# --------------------------------------------------------------------------- #


def _protocols() -> tuple[dict[str, Any], dict[str, Any]]:
    v5: dict[str, Any] = {
        "protocol_version": "toolcredit_diagnostic_v5",
        "schema_version": "toolcredit_diagnostic_schema_v5",
        "frozen_at": "2026-09-07T00:00:00Z",
        "git_head_at_freeze": "abc",
        "supersedes": {"protocol_version": "toolcredit_diagnostic_v4"},
        "checkpoints": {E3_ROLE: {"state": "validated", "role": E3_ROLE}},
        "checkpoint_locator_gate": {"rule": "base rule", "materialization_manifests": {E3_ROLE: "eval/materialization/e3.json"}},
        "evaluator": {"version": "m7", "m9_extension": {"new_roles": []}},
        **{field: {"frozen": field} for field in FROZEN_SEMANTIC_FIELDS},
    }
    v6 = copy.deepcopy(v5)
    v6.update(
        {
            "protocol_version": "toolcredit_diagnostic_v6",
            "schema_version": "toolcredit_diagnostic_schema_v6",
            "frozen_at": "2026-09-10T00:00:00Z",
            "git_head_at_freeze": "def",
            "supersedes": {"protocol_version": "toolcredit_diagnostic_v5"},
        }
    )
    for role in E7_ROLE_NAMES:
        v6["checkpoints"][role] = {
            "state": "validated",
            "role": role,
            "generation_mode": "tir",
            "weight_role": role,
            "run_name": "e7_dynamic_filtering_20260909_200152",
            "source_locator": f"/runs/e7/{role}",
            "dynamic_filtering": {"effective_step": 93 if role == ROLE_EQUAL_COMPUTE else 200, "cumulative_trajectories": 102400},
        }
        v6["checkpoint_locator_gate"]["materialization_manifests"][role] = f"eval/materialization/{role}.json"
    v6["checkpoint_locator_gate"]["rule"] = "base rule; v6 additionally requires ..."
    v6["evaluator"]["m10_extension"] = {
        "new_roles": list(E7_ROLE_NAMES),
        "source_run": "e7_dynamic_filtering_20260909_200152",
        "dynamic_filtering": copy.deepcopy(DYNAMIC_FILTERING_SEMANTICS),
        "equal_compute_step": 93,
        "equal_compute_locator": "rl/runs/e7/checkpoints/equal_compute_step_93",
        "expected_workload": {"total_new_trajectories": 7600, "trajectories_per_role": 3800},
        "pairings": [dict(item) for item in PAIRINGS],
        "interpretation_matrix": {
            "delta_step": "auc",
            "delta_compute": "paired",
            "cases": {"A": "a", "B": "b", "C": "c", "D": "d"},
        },
    }
    return v5, v6


def test_semantic_diff_accepts_the_declared_additive_shape() -> None:
    v5, v6 = _protocols()
    report = semantic_diff(v5, v6)
    assert set(report["changed_top_level_fields"]) == EXPECTED_CHANGED_TOP_LEVEL
    assert report["equal_compute_step"] == 93
    assert "generation_mode" in report["unchanged_frozen_semantic_fields"]


@pytest.mark.parametrize(
    "mutate,message",
    [
        # A frozen field that moves shows up in the top-level diff first; the FROZEN_SEMANTIC_FIELDS
        # loop is the second, redundant net behind that check.
        (lambda v5, v6: v6.update({"generation": {"changed": True}}), "unexpected v5->v6 semantic diff"),
        (lambda v5, v6: v6.update({"generation_mode": {"default": "notool"}}), "unexpected v5->v6 semantic diff"),
        (lambda v5, v6: v6["checkpoints"][E3_ROLE].update({"tampered": True}), "frozen v5 checkpoint role"),
        (lambda v5, v6: v6["checkpoints"][ROLE_STEP_200].update({"generation_mode": "notool"}), "wrong mode/weight"),
        (
            lambda v5, v6: v6["checkpoints"][ROLE_EQUAL_COMPUTE].update(
                {"source_locator": v6["checkpoints"][ROLE_STEP_200]["source_locator"]}
            ),
            "must not share one source checkpoint",
        ),
        (
            lambda v5, v6: v6["evaluator"]["m10_extension"]["dynamic_filtering"].update({"zero_variance_metric": "acc"}),
            "dynamic-filtering semantics drifted",
        ),
        (
            lambda v5, v6: v6["evaluator"]["m10_extension"]["expected_workload"].update({"total_new_trajectories": 15200}),
            "workload is not",
        ),
        (
            lambda v5, v6: v6["evaluator"]["m10_extension"].update({"equal_compute_step": 300}),
            "not a valid effective step",
        ),
        (
            lambda v5, v6: v6["evaluator"]["m10_extension"].update({"equal_compute_locator": "x/equal_compute_step_7"}),
            "does not match its declared step",
        ),
        (
            lambda v5, v6: v6["evaluator"]["m10_extension"].update(
                {"pairings": [{"before_role": E3_ROLE, "after_role": ROLE_STEP_200, "priority": "primary"}]}
            ),
            "pairings differ",
        ),
        (
            lambda v5, v6: v6["evaluator"]["m10_extension"]["interpretation_matrix"]["cases"].pop("D"),
            "four-cell table",
        ),
        (lambda v5, v6: v6["evaluator"].update({"version": "m8"}), "beyond the m10_extension block"),
        (lambda v5, v6: v6["checkpoint_locator_gate"].update({"rule": "unrelated rule"}), "does not extend"),
        (lambda v5, v6: v6["supersedes"].update({"protocol_version": "toolcredit_diagnostic_v3"}), "supersedes the wrong"),
    ],
)
def test_semantic_diff_rejects_every_smuggled_change(mutate: Any, message: str) -> None:
    v5, v6 = _protocols()
    mutate(v5, v6)
    with pytest.raises(ValueError, match=message):
        semantic_diff(v5, v6)


def test_a_frozen_field_set_back_to_its_v5_value_still_fails_the_top_level_check() -> None:
    """Making v6 supersede v4 makes ``supersedes`` equal to v5's, which removes it from the diff."""
    v5, v6 = _protocols()
    v6["supersedes"] = copy.deepcopy(v5["supersedes"])
    with pytest.raises(ValueError, match="unexpected v5->v6 semantic diff"):
        semantic_diff(v5, v6)


def test_pairings_declare_two_primaries_from_e3_to_the_two_e7_roles() -> None:
    primary = [item for item in PAIRINGS if item["priority"] == "primary"]
    assert len(primary) == 2
    assert {item["before_role"] for item in primary} == {E3_ROLE}
    assert {item["after_role"] for item in primary} == set(E7_ROLE_NAMES)
