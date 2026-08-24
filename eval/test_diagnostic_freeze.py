"""Identity and semantic gates for the M7 diagnostic freeze bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from eval.build_diagnostic_freeze import ROOT
from eval.build_diagnostic_freeze_v2 import closure_files, verify
from rl.launch.e5_turn_credit import _validate_checkpoint_locators, validate_freeze_bundle


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parquet_ids(relative: str) -> set[str]:
    table = pq.read_table(ROOT / relative, columns=["extra_info"])
    return {str(row["extra_info"]["id"]) for row in table.to_pylist()}


def test_freeze_closure_hashes_and_versions() -> None:
    verify()
    summary = validate_freeze_bundle(ROOT / "eval/diagnostic_freeze_v2.sha256")
    assert summary["panel_question_count"] == 760
    assert summary["closure_file_count"] == len(closure_files())
    assert summary["protocol_version"] == "toolcredit_diagnostic_v2"
    assert summary["taxonomy_version"] == "toolcredit_failure_taxonomy_v1"
    assert summary["checkpoint_roles"]["e3"]["state"] == "validated"
    assert summary["checkpoint_roles"]["e5"] == {
        "role": "e5_turn_credit_step_200",
        "state": "future",
        "locator": None,
    }


def test_panel_exact_primary_and_zero_train_id_overlap() -> None:
    rows = _rows(ROOT / "eval/diagnostic_panel_v1.jsonl")
    all_ids = {row["question_id"] for row in rows}
    primary_ids = {
        row["question_id"] for row in rows if "m6_primary_greedy" in row["panel_roles"]
    }
    assert len(rows) == len(all_ids) == 760
    assert primary_ids == _parquet_ids("rl/data/e3_val_math500_100.parquet")
    assert len(primary_ids) == 100
    assert not (all_ids & _parquet_ids("rl/data/e3_train.parquet"))
    assert all(row["split"] == "test" for row in rows)


def test_generation_tool_pairing_and_bootstrap_are_frozen_exactly() -> None:
    protocol = json.loads((ROOT / "eval/diagnostic_protocol_v2.json").read_text(encoding="utf-8"))
    generation = protocol["generation"]
    assert generation["greedy"] == {
        "do_sample": False,
        "n": 1,
        "temperature": 0.0,
        "top_k": -1,
        "top_p": 1.0,
    }
    assert generation["sampling"] == {
        "do_sample": True,
        "n": 4,
        "temperature": 0.6,
        "top_k": -1,
        "top_p": 1.0,
    }
    assert generation["max_response_tokens_total"] == 3072
    assert generation["max_user_tool_turns"] == 4
    assert generation["max_assistant_turns"] == 5
    assert generation["max_parallel_tool_calls"] == 1
    assert generation["base_seed"] == 42
    assert protocol["pairing_and_transitions"]["greedy_key"] == ["question_id"]
    assert protocol["pairing_and_transitions"]["sampling_key"] == [
        "question_id",
        "sample_index",
    ]
    assert protocol["bootstrap"]["replicates"] == 10000
    assert protocol["bootstrap"]["seed"] == 42

    truncation = protocol["tool"]["max_tool_response_length"]
    assert truncation["limit"] == 512
    assert truncation["operation"] == (
        "if len(text) > 512: text[:256] + marker + text[-256:]"
    )
    assert "before tokenization" in truncation["meaning"]
    assert "not a token budget" in truncation["meaning"]
    assert "post-processed/post-truncation" in truncation["visible_only_adoption"]
    assert "raw sandbox stdout is audit-only" in truncation["visible_only_adoption"]


def test_taxonomy_has_exact_preregistered_labels() -> None:
    taxonomy = json.loads((ROOT / "eval/diagnostic_taxonomy_v1.json").read_text(encoding="utf-8"))
    assert taxonomy["labels"] == [
        "tool_json_or_parser_failure",
        "no_tool_correct",
        "no_tool_wrong_tool_likely_helpful",
        "stateless_variable_reuse",
        "python_or_sandbox_execution_error",
        "no_recovery_after_tool_error",
        "tool_result_not_adopted_or_misread",
        "tool_correct_semantic_or_constraint_miss",
        "exact_to_low_precision_approximation",
        "boxed_or_verifier_format_error",
        "tool_overuse",
        "budget_exhaustion_or_truncation",
    ]
    assert len(taxonomy["primary_decision_tree"]) == 10


def test_checkpoint_locators_are_role_aware_and_fail_closed() -> None:
    protocol = json.loads((ROOT / "eval/diagnostic_protocol_v2.json").read_text(encoding="utf-8"))
    summary = _validate_checkpoint_locators(protocol)
    assert summary["raw"]["state"] == "validated"
    assert summary["sft"]["state"] == "validated"
    assert summary["e3"]["locator"].endswith(
        "rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200"
    )

    bad_path = json.loads(json.dumps(protocol))
    bad_path["checkpoints"]["e3"]["locator"] = (
        "rl/runs/e3_grpo_baseline_20260819_093123/global_step_200"
    )
    try:
        _validate_checkpoint_locators(bad_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("nonexistent E3 checkpoint locator was accepted")

    wrong_role = json.loads(json.dumps(protocol))
    wrong_role["checkpoints"]["e3"]["role"] = "sft_start"
    try:
        _validate_checkpoint_locators(wrong_role)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong checkpoint role was accepted")

    missing_locator = json.loads(json.dumps(protocol))
    missing_locator["checkpoints"]["e3"]["locator"] = None
    try:
        _validate_checkpoint_locators(missing_locator)
    except ValueError:
        pass
    else:
        raise AssertionError("null E3 checkpoint locator was accepted")
