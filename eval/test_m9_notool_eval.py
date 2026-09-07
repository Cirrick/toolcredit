"""M9 evaluator tests: Fixture M (v5 scoring equivalence, notool raw-row validator), the
notool prompt identity with the E3-NoTool training rollouts, the v4->v5 semantic diff, the
§8 preregistered case reader and the notool mechanism metrics."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from eval.build_diagnostic_freeze_v5 import NOTOOL_MODE_SEMANTICS, PAIRINGS
from eval.checkpoints_v4 import ROLE_V4
from eval.checkpoints_v5 import E3_CONTROL_DIFF_PATHS, NOTOOL_ROLE_NAMES, NOTOOL_RUN, NOTOOL_WEIGHT_ROLES, ROLE_V5
from eval.generate import ROLE_NAMES, expected_keys, load_panel_questions, validate_raw_generation_row
from eval.m8_e5v2_eval import M7_CANONICAL_RUN, expected_keys_v4, validate_raw_generation_row_v4
from eval.m9_notool_eval import (
    M8_CANONICAL_RUN,
    NOTOOL_SCHEMA_VERSION,
    PROTOCOL_VERSION_V5,
    ROLE_NAMES_V5,
    SENSITIVITY_ROLES,
    classify_notool_termination,
    classify_preregistered_case_m9,
    expected_keys_v5,
    is_notool_role,
    notool_behavior,
    notool_prompt_ids,
    paired_delta_by_stratum,
    score_trajectory_v5,
    validate_raw_generation_row_v5,
)
from eval.verify_diagnostic_freeze_v5_semantics import semantic_diff
from rl.validate_e3notool_run import tool_tag_stats

E3_RAW = M7_CANONICAL_RUN / "generation/e3_grpo_baseline_step_200/greedy/MATH500.jsonl"
E3_SCORED = M7_CANONICAL_RUN / "scored/e3_grpo_baseline_step_200/greedy/MATH500.jsonl"
E5V2_RAW = M8_CANONICAL_RUN / f"generation/{ROLE_V4}/greedy/MATH500.jsonl"
E5V2_SCORED = M8_CANONICAL_RUN / f"scored/{ROLE_V4}/greedy/MATH500.jsonl"
SCORED_FIELDS = ("reward", "verifier_outcome", "correct", "format_ok", "invalid", "truncated", "budget_exhausted", "raw_evidence_pointer")


def _first(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _questions() -> dict[str, dict]:
    return {row["question_id"]: row for row in load_panel_questions()}


def _notool_row_from(raw: dict, role: str = ROLE_V5) -> dict:
    """Rebuild a raw TIR row as a well-formed notool row (same response text)."""
    response = str(raw["response"])
    stats = tool_tag_stats(response)
    row = {key: value for key, value in raw.items() if key not in {"tool_metadata", "messages", "response_mask"}}
    row.update(
        {
            "protocol_version": PROTOCOL_VERSION_V5,
            "schema_version": NOTOOL_SCHEMA_VERSION,
            "checkpoint_role": role,
            "generation_mode": "notool",
            "messages": [{"role": "assistant", "content": response}],
            "termination_reason": "final_answer",
            "tool_tag_emitted": bool(stats["tool_tag_emitted"]),
            "tool_tag_count": int(stats["tool_tag_count"]),
            "tool_metadata": {
                "tool_call_counts": 0,
                "tool_success_count": 0,
                "tool_error_count": 0,
                "tool_parse_error_count": 0,
                "num_turns": 2,
                "generation_mode": "notool",
                "sglang_finish_reason": {"type": "stop", "matched": 151645},
                "raw_output_token_count": int(raw["response_token_count"]),
                "tool_tag_stats": stats,
            },
        }
    )
    return row


# --------------------------------------------------------------------------- #
# key universe / role plumbing
# --------------------------------------------------------------------------- #


def test_v5_key_universe_is_15200_and_disjoint_from_frozen_universes() -> None:
    keys = expected_keys_v5()
    assert len(keys) == 15200 and len(set(keys)) == 15200
    assert sum(key[1] == "greedy" for key in keys) == 4 * 760
    assert not set(keys) & set(expected_keys())
    assert not set(keys) & set(expected_keys_v4())
    assert ROLE_NAMES_V5 == (*ROLE_NAMES, ROLE_V4, *NOTOOL_ROLE_NAMES)
    assert NOTOOL_ROLE_NAMES[-1] == ROLE_V5 and NOTOOL_WEIGHT_ROLES[ROLE_V5] == ROLE_V5
    assert {NOTOOL_WEIGHT_ROLES[role] for role in NOTOOL_ROLE_NAMES[:3]} == {"raw_base_model", "sft_start", "e3_grpo_baseline_step_200"}


def test_sensitivity_roles_are_notool_but_outside_the_protocol_universe() -> None:
    for role in SENSITIVITY_ROLES:
        assert is_notool_role(role) and role not in ROLE_NAMES_V5
        with pytest.raises(ValueError, match="unknown role"):
            expected_keys_v5((role,))


def test_control_diff_paths_match_the_launcher_gate() -> None:
    from rl.launch.e3notool_grpo import E3_DIFF_PATHS

    assert set(E3_CONTROL_DIFF_PATHS) == E3_DIFF_PATHS


# --------------------------------------------------------------------------- #
# Fixture M — v5 scoring equivalence on canonical raw rows + notool validator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw_path,scored_path", [(E3_RAW, E3_SCORED), (E5V2_RAW, E5V2_SCORED)])
def test_fixture_m_v5_scoring_reproduces_canonical_scored_rows(raw_path: Path, scored_path: Path) -> None:
    questions = _questions()
    raw_rows, scored_rows = _rows(raw_path)[:25], _rows(scored_path)[:25]
    for raw, canonical in zip(raw_rows, scored_rows):
        assert canonical["question_id"] == raw["question_id"]
        scored = score_trajectory_v5(raw, questions[raw["question_id"]])
        for field in SCORED_FIELDS:
            assert scored[field] == canonical[field], (raw["question_id"], field)


def test_fixture_m_validators_accept_frozen_rows_and_reject_role_leakage() -> None:
    questions = _questions()
    raw = _first(E3_RAW)
    question = questions[raw["question_id"]]
    validate_raw_generation_row(raw, question)
    validate_raw_generation_row_v4(raw, question)
    validate_raw_generation_row_v5(raw, question)
    relabeled = {**raw, "checkpoint_role": ROLE_V5}
    with pytest.raises(ValueError, match="unknown checkpoint role"):
        validate_raw_generation_row(relabeled, question)
    with pytest.raises(ValueError, match="unknown checkpoint role"):
        validate_raw_generation_row_v4(relabeled, question)
    tampered = {**raw, "response": raw["response"] + " "}
    with pytest.raises(ValueError, match="hash"):
        validate_raw_generation_row_v5(tampered, question)


def test_fixture_m_notool_validator_rejects_rows_with_tool_turns() -> None:
    questions = _questions()
    with_tool = next(row for row in _rows(E3_RAW) if int(row["tool_metadata"]["tool_call_counts"]) > 0)
    question = questions[with_tool["question_id"]]
    # A TIR row merely relabeled as the notool role: nonzero tool counts / tool turns / ledger.
    relabeled = {**with_tool, "checkpoint_role": ROLE_V5, "generation_mode": "notool", "schema_version": NOTOOL_SCHEMA_VERSION}
    with pytest.raises(ValueError, match="nonzero tool_call_counts"):
        validate_raw_generation_row_v5(relabeled, question)
    zeroed = copy.deepcopy(relabeled)
    zeroed["tool_metadata"] = {key: (0 if key in {"tool_call_counts", "tool_success_count", "tool_error_count", "tool_parse_error_count"} else value) for key, value in zeroed["tool_metadata"].items()}
    with pytest.raises(ValueError, match="ledger"):
        validate_raw_generation_row_v5(zeroed, question)
    zeroed["tool_metadata"].pop("turn_credit_ledger")
    with pytest.raises(ValueError, match="tool turn"):
        validate_raw_generation_row_v5(zeroed, question)
    # A well-formed notool row passes, and every tampering of its invariants is caught.
    good = _notool_row_from(with_tool)
    validate_raw_generation_row_v5(good, question)
    scored = score_trajectory_v5(good, question)
    assert scored["reward"]["n_tool_calls"] == 0.0 and scored["budget_exhausted"] is False and scored["truncated"] is False
    for mutate, message in (
        (lambda row: row.update(generation_mode="tir"), "generation_mode"),
        (lambda row: row.update(termination_reason="assistant_turn_budget"), "termination reason"),
        (lambda row: row.update(tool_tag_count=row["tool_tag_count"] + 1), "tool-tag"),
        (lambda row: row.update(messages=[{"role": "assistant", "content": "x"}]), "assistant message"),
        (lambda row: row.update(schema_version="toolcredit_m7_trajectory_schema_v1"), "schema version"),
    ):
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            validate_raw_generation_row_v5(bad, question)
    truncated = copy.deepcopy(good)
    truncated["termination_reason"] = "length"
    assert score_trajectory_v5(truncated, question)["truncated"] is True
    with pytest.raises(ValueError, match="unknown checkpoint role"):
        validate_raw_generation_row_v4(good, question)


def test_notool_validator_accepts_sensitivity_role_rows() -> None:
    questions = _questions()
    raw = _first(E3_RAW)
    row = _notool_row_from(raw, role=SENSITIVITY_ROLES[0])
    validate_raw_generation_row_v5(row, questions[raw["question_id"]])


# --------------------------------------------------------------------------- #
# notool prompt identity with the E3-NoTool training rollouts (Fixture J, evaluator side)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from eval.checkpoints import COMMON_TOKENIZER

    return AutoTokenizer.from_pretrained(str(COMMON_TOKENIZER))


def test_notool_prompt_reproduces_the_training_rollout_input(tokenizer) -> None:
    import pyarrow.parquet as pq

    from eval.checkpoints import ROOT

    train_row = json.loads((NOTOOL_RUN / "predictions/train/1.jsonl").read_text(encoding="utf-8").splitlines()[0])
    table = pq.read_table(ROOT / "rl/data/e3_train.parquet", columns=["prompt", "extra_info"])
    record = next(item for item in table.to_pylist() if item["extra_info"]["id"] == train_row["sample_id"])
    content = str(record["prompt"][0]["content"])
    ids = notool_prompt_ids(tokenizer, content)
    assert tokenizer.decode(ids, skip_special_tokens=True) == train_row["input"]
    direct = tokenizer.apply_chat_template([{"role": "user", "content": content}], tools=None, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    assert ids == list(direct)


def test_notool_panel_prompt_has_no_tool_schema_and_frozen_suffix(tokenizer) -> None:
    from eval.generate import validate_frozen_runtime_config

    config = validate_frozen_runtime_config()
    question = load_panel_questions()[0]
    prompt = str(question["question"]) + config["generation"]["prompt_suffix"]
    decoded = tokenizer.decode(notool_prompt_ids(tokenizer, prompt), skip_special_tokens=True)
    assert decoded.startswith("user\n") and decoded.endswith("assistant\n<think>\n\n</think>\n\n")
    assert prompt in decoded
    assert not any(marker in decoded for marker in ("<tools>", "code_interpreter", "<tool_call>", "# Tools", "system"))


def test_termination_classification() -> None:
    assert classify_notool_termination({"type": "stop", "matched": 151645}, 100, 3072) == "final_answer"
    assert classify_notool_termination({"type": "length", "length": 3072}, 3072, 3072) == "length"
    assert classify_notool_termination({"type": "stop"}, 3072, 3072) == "length"
    assert classify_notool_termination("stop", 10, 3072) == "final_answer"


# --------------------------------------------------------------------------- #
# §8 preregistered case reader, stratum decomposition, behavior metrics
# --------------------------------------------------------------------------- #


def test_preregistered_case_m9_reader() -> None:
    a = {"point_estimate": -0.04, "lower": -0.065, "upper": -0.015}
    assert classify_preregistered_case_m9(a, 0.01)["case"] == "A"
    assert classify_preregistered_case_m9(a, 0.05)["case"] == "A_indistinguishable_from_start_point_debt"
    assert classify_preregistered_case_m9(a, None)["case"] == "A"
    b = {"point_estimate": -0.005, "lower": -0.03, "upper": 0.02}
    assert classify_preregistered_case_m9(b, 0.05)["case"] == "B"
    c = {"point_estimate": 0.03, "lower": 0.005, "upper": 0.055}
    assert classify_preregistered_case_m9(c, 0.05)["case"] == "C"
    small = {"point_estimate": -0.015, "lower": -0.028, "upper": -0.002}
    assert classify_preregistered_case_m9(small, 0.0)["case"] == "outside_preregistered_table"
    assert classify_preregistered_case_m9({"point_estimate": None, "lower": None, "upper": None}, 0.0)["case"] == "undefined"


def _synthetic_scored(role: str, question_id: str, stratum: str, correct: bool, tokens: int = 100, tagged: bool = False, truncated: bool = False) -> dict:
    return {
        "checkpoint_role": role,
        "evaluation_setting": "greedy",
        "question_id": question_id,
        "source": "MATH500",
        "stratum": stratum,
        "correct": correct,
        "format_ok": True,
        "invalid": not correct,
        "truncated": truncated,
        "response_token_count": tokens,
        "tool_tag_emitted": tagged,
        "tool_tag_count": int(tagged),
        "reward": {"n_tool_calls": 0.0},
    }


def test_stratum_decomposition_and_notool_behavior() -> None:
    questions = load_panel_questions()
    before = [_synthetic_scored("e3", row["question_id"], row["stratum"], index % 2 == 0) for index, row in enumerate(questions)]
    after = [_synthetic_scored(ROLE_V5, row["question_id"], row["stratum"], index % 3 == 0) for index, row in enumerate(questions)]
    result = paired_delta_by_stratum(before, after)
    assert sum(item["denominator"] for item in result.values()) == 760
    assert set(result) == {row["stratum"] for row in questions}
    for item in result.values():
        assert item["after_correct"] - item["before_correct"] == item["fixed_failures"] - item["new_failures"]
    with pytest.raises(ValueError, match="complete paired greedy panel"):
        paired_delta_by_stratum(before[:-1], after)
    behavior = notool_behavior([_synthetic_scored(ROLE_V5, "q1", "s", True, 100), _synthetic_scored(ROLE_V5, "q2", "s", False, 300, tagged=True, truncated=True)])
    assert behavior["trajectory_count"] == 2
    assert behavior["response_tokens"]["mean"] == 200
    assert behavior["truncation_rate"]["value"] == 0.5 and behavior["tool_tag_emitted_rate"]["value"] == 0.5
    assert behavior["tool_calls"]["value"] == 0.0
    with pytest.raises(ValueError, match="tool calls"):
        notool_behavior([{**_synthetic_scored(ROLE_V5, "q1", "s", True), "reward": {"n_tool_calls": 1.0}}])


# --------------------------------------------------------------------------- #
# v4 -> v5 semantic diff
# --------------------------------------------------------------------------- #


def _synthetic_v4() -> dict:
    frozen = {name: {"x": name} for name in ("bootstrap", "data_isolation", "file_sha256", "pairing_and_transitions", "panel", "persistence", "prompt", "protocol_change_rule", "taxonomy", "tool", "verifier")}
    return {
        **frozen,
        "generation": {"max_response_tokens_total": 3072},
        "protocol_version": "toolcredit_diagnostic_v4",
        "schema_version": "s4",
        "frozen_at": "t4",
        "git_head_at_freeze": "g4",
        "supersedes": {"protocol_version": "toolcredit_diagnostic_v3"},
        "checkpoints": {role: {"state": "validated"} for role in (*ROLE_NAMES, ROLE_V4)},
        "checkpoint_locator_gate": {"rule": "base rule", "materialization_manifests": {"e3_grpo_baseline_step_200": "a", "e5_turn_credit_step_200": "b", ROLE_V4: "c"}, "timing": "t"},
        "evaluator": {"version": "v1", "source_sha256": {}, "m8_extension": {"new_role": ROLE_V4}},
    }


def _synthetic_v5(v4: dict) -> dict:
    v5 = copy.deepcopy(v4)
    v5.update({"protocol_version": "toolcredit_diagnostic_v5", "schema_version": "s5", "frozen_at": "t5", "git_head_at_freeze": "g5", "supersedes": {"protocol_version": "toolcredit_diagnostic_v4"}})
    v5["generation_mode"] = {
        "default": "tir",
        "rule": "r",
        "modes": {"tir": {"definition": "d"}, "notool": copy.deepcopy(NOTOOL_MODE_SEMANTICS)},
        "notool_roles": dict(NOTOOL_WEIGHT_ROLES),
    }
    for role in NOTOOL_ROLE_NAMES:
        v5["checkpoints"][role] = {"state": "validated", "generation_mode": "notool", "weight_role": NOTOOL_WEIGHT_ROLES[role]}
    v5["checkpoint_locator_gate"]["materialization_manifests"][ROLE_V5] = "d"
    v5["checkpoint_locator_gate"]["rule"] = "base rule; plus v5"
    v5["evaluator"]["m9_extension"] = {
        "new_roles": list(NOTOOL_ROLE_NAMES),
        "new_checkpoint_role": ROLE_V5,
        "expected_workload": {"total_new_trajectories": 15200, "trajectories_per_role": 3800},
        "pairings": [dict(item) for item in PAIRINGS],
    }
    return v5


def test_semantic_diff_accepts_additive_v5_and_rejects_drift() -> None:
    v4 = _synthetic_v4()
    v5 = _synthetic_v5(v4)
    report = semantic_diff(v4, v5)
    assert "generation_mode" in report["changed_top_level_fields"]
    cases = [
        ("generation", lambda d: d.update(generation={"max_response_tokens_total": 2048})),
        ("frozen v4 checkpoint role", lambda d: d["checkpoints"].update({"e3_grpo_baseline_step_200": {"state": "swapped"}})),
        ("frozen v4 checkpoint role", lambda d: d["checkpoints"]["e3_grpo_baseline_step_200"].update(generation_mode="tir")),
        ("wrong mode/weight", lambda d: d["checkpoints"][ROLE_V5].update(weight_role="sft_start")),
        ("evaluator", lambda d: d["evaluator"].update(version="v9")),
        ("pairings differ", lambda d: d["evaluator"]["m9_extension"]["pairings"].pop()),
        ("notool mode semantics", lambda d: d["generation_mode"]["modes"]["notool"].update(max_response_tokens_total=4096)),
        ("manifests", lambda d: d["checkpoint_locator_gate"]["materialization_manifests"].pop(ROLE_V5)),
    ]
    for message, mutate in cases:
        drifted = copy.deepcopy(v5)
        mutate(drifted)
        with pytest.raises(ValueError, match=message):
            semantic_diff(v4, drifted)
