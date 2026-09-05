"""M8 evaluator tests: extended role universe, frozen-equivalent scoring, statistics, v3->v4 diff."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eval.checkpoints_v4 import ROLE_V4
from eval.generate import ROLE_NAMES, expected_keys, load_panel_questions, validate_raw_generation_row
from eval.m8_e5v2_eval import (
    M7_CANONICAL_RUN,
    ROLE_NAMES_V4,
    classify_preregistered_case,
    expected_keys_v4,
    failure_category_migration_statistic,
    score_trajectory_v4,
    validate_raw_generation_row_v4,
)
from eval.verify_diagnostic_freeze_v4_semantics import semantic_diff

E3_RAW = M7_CANONICAL_RUN / "generation/e3_grpo_baseline_step_200/greedy/MATH500.jsonl"
E3_SCORED = M7_CANONICAL_RUN / "scored/e3_grpo_baseline_step_200/greedy/MATH500.jsonl"


def _first(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def test_v4_key_universe_is_3800_and_disjoint_from_frozen() -> None:
    keys = expected_keys_v4()
    assert len(keys) == 3800 and len(set(keys)) == 3800
    assert sum(key[1] == "greedy" for key in keys) == 760
    assert not set(keys) & set(expected_keys())
    assert ROLE_NAMES_V4 == (*ROLE_NAMES, ROLE_V4)


def test_v4_validator_accepts_new_role_and_frozen_validator_rejects_it() -> None:
    questions = {row["question_id"]: row for row in load_panel_questions()}
    raw = _first(E3_RAW)
    question = questions[raw["question_id"]]
    validate_raw_generation_row(raw, question)
    validate_raw_generation_row_v4(raw, question)
    relabeled = {**raw, "checkpoint_role": ROLE_V4}
    validate_raw_generation_row_v4(relabeled, question)
    with pytest.raises(ValueError, match="unknown checkpoint role"):
        validate_raw_generation_row(relabeled, question)
    tampered = {**raw, "response": raw["response"] + " "}
    with pytest.raises(ValueError, match="hash"):
        validate_raw_generation_row_v4(tampered, question)


def test_v4_scoring_reproduces_canonical_m7_scored_row() -> None:
    questions = {row["question_id"]: row for row in load_panel_questions()}
    raw = _first(E3_RAW)
    canonical = _first(E3_SCORED)
    assert canonical["question_id"] == raw["question_id"]
    scored = score_trajectory_v4(raw, questions[raw["question_id"]])
    for field in ("reward", "verifier_outcome", "correct", "format_ok", "invalid", "truncated", "budget_exhausted", "raw_evidence_pointer"):
        assert scored[field] == canonical[field], field


def test_migration_statistic_reproduces_m7_e3_to_e5_bootstrap_point() -> None:
    rows = [json.loads(line) for line in (M7_CANONICAL_RUN / "transitions/e3_to_e5.paired_rows.jsonl").read_text().splitlines() if line]
    greedy_aime24 = [row for row in rows if row.get("setting", "greedy") == "greedy" and row["source"] == "AIME2024" and row["key"][0].startswith("aime24")]
    reference = json.load((M7_CANONICAL_RUN / "transitions/e3_to_e5.bootstrap.json").open())["settings"]["greedy"]["AIME2024"]["failure_category_migration_rate"]["point_estimate"]
    assert failure_category_migration_statistic(greedy_aime24) == pytest.approx(reference)
    assert failure_category_migration_statistic([{"before_state": "correct", "after_state": "correct"}]) is None


def test_preregistered_case_classification() -> None:
    null_delta = {"point_estimate": -0.005, "lower": -0.03, "upper": 0.02}
    assert classify_preregistered_case(null_delta, 0.5, True, 0.10)["case"] == "A"
    assert classify_preregistered_case(null_delta, 2.5, True, 0.10)["case"] == "B"
    assert classify_preregistered_case(null_delta, 0.5, True, 0.05)["case"] == "A_untested_exposure_below_floor"
    positive = {"point_estimate": 0.03, "lower": 0.005, "upper": 0.055}
    assert classify_preregistered_case(positive, 1.0, True, 0.10)["case"] == "C"
    negative = {"point_estimate": -0.03, "lower": -0.055, "upper": -0.005}
    assert classify_preregistered_case(negative, 0.0, True, 0.10)["case"] == "outside_preregistered_table"


def _synthetic_v3() -> dict:
    frozen = {name: {"x": name} for name in ("bootstrap", "data_isolation", "file_sha256", "generation", "pairing_and_transitions", "panel", "persistence", "prompt", "protocol_change_rule", "taxonomy", "tool", "verifier")}
    return {
        **frozen,
        "protocol_version": "toolcredit_diagnostic_v3",
        "schema_version": "s3",
        "frozen_at": "t3",
        "git_head_at_freeze": "g3",
        "supersedes": {"protocol_version": "toolcredit_diagnostic_v2"},
        "checkpoints": {"e3_grpo_baseline_step_200": {"state": "validated"}, "e5_turn_credit_step_200": {"state": "validated"}},
        "checkpoint_locator_gate": {"rule": "base rule", "materialization_manifests": {"e3_grpo_baseline_step_200": "a", "e5_turn_credit_step_200": "b"}, "timing": "t"},
        "evaluator": {"version": "v1", "source_sha256": {}},
    }


def _synthetic_v4(v3: dict) -> dict:
    v4 = copy.deepcopy(v3)
    v4.update({"protocol_version": "toolcredit_diagnostic_v4", "schema_version": "s4", "frozen_at": "t4", "git_head_at_freeze": "g4", "supersedes": {"protocol_version": "toolcredit_diagnostic_v3"}})
    v4["checkpoints"][ROLE_V4] = {"state": "validated"}
    v4["checkpoint_locator_gate"]["materialization_manifests"][ROLE_V4] = "c"
    v4["checkpoint_locator_gate"]["rule"] = "base rule; plus v2"
    v4["evaluator"]["m8_extension"] = {
        "new_role": ROLE_V4,
        "expected_workload": {"total_new_trajectories": 3800},
        "pairings": [
            {"before_role": "e3_grpo_baseline_step_200", "after_role": ROLE_V4, "priority": "primary"},
            {"before_role": "e5_turn_credit_step_200", "after_role": ROLE_V4, "priority": "secondary"},
        ],
    }
    return v4


def test_semantic_diff_accepts_additive_v4_and_rejects_drift() -> None:
    v3 = _synthetic_v3()
    v4 = _synthetic_v4(v3)
    report = semantic_diff(v3, v4)
    assert "checkpoints" in report["changed_top_level_fields"]
    drifted = copy.deepcopy(v4)
    drifted["generation"] = {"x": "changed"}
    with pytest.raises(ValueError, match="generation"):
        semantic_diff(v3, drifted)
    altered_role = copy.deepcopy(v4)
    altered_role["checkpoints"]["e3_grpo_baseline_step_200"] = {"state": "swapped"}
    with pytest.raises(ValueError, match="frozen v3 checkpoint role"):
        semantic_diff(v3, altered_role)
    evaluator_drift = copy.deepcopy(v4)
    evaluator_drift["evaluator"]["version"] = "v9"
    with pytest.raises(ValueError, match="evaluator"):
        semantic_diff(v3, evaluator_drift)
