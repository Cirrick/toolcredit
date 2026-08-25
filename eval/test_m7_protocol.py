"""CPU fixtures for frozen M7 generation, persistence, roles, and scoring boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import eval.checkpoints as checkpoint_module
from eval.checkpoints import ROLE_ORDER, ROLE_SOURCE_PATHS, validate_role_bindings
from eval.generate import (
    append_jsonl_fsync,
    derive_sample_seed,
    expected_keys,
    load_panel_questions,
    to_sglang_sampling_params,
    validate_frozen_runtime_config,
    validated_resume_rows,
)


def test_seed_formula_known_answers_and_cross_role_identity() -> None:
    expected = [40656745, 117193861, 900643002, 1238494141]
    assert [derive_sample_seed("math500_000000", index) for index in range(4)] == expected
    assert len({derive_sample_seed("math500_000000", index) for index in range(4)}) == 4


def test_panel_order_counts_and_exact_key_universe() -> None:
    rows = load_panel_questions()
    assert len(rows) == 760
    assert rows[0]["question_id"] == "math500_000000"
    keys = expected_keys()
    assert len(keys) == 15200
    assert len(set(keys)) == 15200
    assert sum(key[1] == "greedy" for key in keys) == 3040
    assert sum(key[1] == "sampled" for key in keys) == 12160
    same_pair = [key for key in keys if key[2] == "math500_000000" and key[3] == 2]
    assert len(same_pair) == 4
    assert {key[4] for key in same_pair} == {900643002}


def test_runtime_config_exact_and_diagnostic_priority() -> None:
    config = validate_frozen_runtime_config()
    assert config["generation"]["max_response_tokens_total"] == 3072
    assert config["generation"]["max_user_tool_turns"] == 4
    assert config["generation"]["max_assistant_turns"] == 5
    assert config["generation"]["max_parallel_tool_calls"] == 1
    assert config["generation"]["max_tool_response_length"] == 512
    assert config["diagnostic_evidence"]["greedy_paired_transitions"] == "primary_diagnostic"
    assert config["diagnostic_evidence"]["sampled_matched_index_transitions"] == "secondary_exploratory"


def test_pinned_sglang_request_translation_preserves_frozen_seed_and_controls() -> None:
    greedy = to_sglang_sampling_params(
        {"do_sample": False, "n": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 123, "max_new_tokens": 3072}
    )
    assert greedy == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "sampling_seed": 123,
        "max_new_tokens": 3072,
    }
    sampled = to_sglang_sampling_params(
        {"do_sample": True, "n": 4, "temperature": 0.6, "top_p": 1.0, "top_k": -1, "seed": 456, "stop_token_ids": [1, 2]}
    )
    assert sampled["sampling_seed"] == 456
    assert sampled["temperature"] == 0.6
    assert sampled["stop_token_ids"] == [1, 2]
    assert "do_sample" not in sampled and "n" not in sampled and "seed" not in sampled
    with pytest.raises(ValueError, match="unsupported"):
        to_sglang_sampling_params({"do_sample": False, "temperature": 0.0, "seed": 1, "bogus": 2})


def test_role_binding_rejects_null_swap_glob_and_latest() -> None:
    valid = {role: str(ROLE_SOURCE_PATHS[role]) for role in ROLE_ORDER}
    validate_role_bindings(valid)
    null = dict(valid)
    null["e5_turn_credit_step_200"] = None
    with pytest.raises(ValueError, match="cannot be null"):
        validate_role_bindings(null)
    swapped = dict(valid)
    swapped["e3_grpo_baseline_step_200"] = valid["e5_turn_credit_step_200"]
    with pytest.raises(ValueError, match="swapped or drifted"):
        validate_role_bindings(swapped)
    globbed = dict(valid)
    globbed["e3_grpo_baseline_step_200"] = "rl/runs/e3*/checkpoints/global_step_200"
    with pytest.raises(ValueError, match="glob/latest"):
        validate_role_bindings(globbed)
    latest = dict(valid)
    latest["e3_grpo_baseline_step_200"] = "rl/runs/e3/checkpoints/latest"
    with pytest.raises(ValueError, match="glob/latest"):
        validate_role_bindings(latest)


def _mini_e3_checkpoint(tmp_path: Path, monkeypatch) -> Path:
    run = tmp_path / "e3_fixture"
    checkpoint = run / "checkpoints/global_step_200"
    required = {
        "actor/model_world_size_1_rank_0.pt": b"model",
        "actor/optim_world_size_1_rank_0.pt": b"optim",
        "actor/extra_state_world_size_1_rank_0.pt": b"extra",
        "actor/fsdp_config.json": b"{}",
        "data.pt": b"data",
    }
    for relative, payload in required.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (run / "status.json").write_text('{"state":"completed"}\n', encoding="utf-8")
    (checkpoint.parent / "latest_checkpointed_iteration.txt").write_text("200\n", encoding="utf-8")
    config = {
        "actor_rollout_ref": {"model": {"path": str(checkpoint_module.SFT)}},
        "trainer": {"total_training_steps": 200},
    }
    config_path = run / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    role = "e3_grpo_baseline_step_200"
    monkeypatch.setitem(checkpoint_module.ROLE_SOURCE_PATHS, role, checkpoint)
    monkeypatch.setitem(
        checkpoint_module.SOURCE_REQUIRED_BYTES,
        role,
        {relative: len(payload) for relative, payload in required.items()},
    )
    monkeypatch.setitem(
        checkpoint_module.SOURCE_ACTOR_SHA256,
        role,
        checkpoint_module.sha256(checkpoint / "actor/model_world_size_1_rank_0.pt"),
    )
    monkeypatch.setitem(
        checkpoint_module.FROZEN_HASHES,
        "e3_resolved_config",
        checkpoint_module.sha256(config_path),
    )
    return checkpoint


def test_training_role_negative_status_tracker_start_missing_and_hash(tmp_path: Path, monkeypatch) -> None:
    role = "e3_grpo_baseline_step_200"
    checkpoint = _mini_e3_checkpoint(tmp_path, monkeypatch)
    assert checkpoint_module.validate_source_role(role, checkpoint)["state"] == "validated"

    status = checkpoint.parents[1] / "status.json"
    status.write_text('{"state":"running"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="status/tracker"):
        checkpoint_module.validate_source_role(role, checkpoint)
    status.write_text('{"state":"completed"}\n', encoding="utf-8")

    tracker = checkpoint.parent / "latest_checkpointed_iteration.txt"
    tracker.write_text("199\n", encoding="utf-8")
    with pytest.raises(ValueError, match="status/tracker"):
        checkpoint_module.validate_source_role(role, checkpoint)
    tracker.write_text("200\n", encoding="utf-8")

    config_path = checkpoint.parents[1] / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["actor_rollout_ref"]["model"]["path"] = str(tmp_path / "not_sft")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setitem(checkpoint_module.FROZEN_HASHES, "e3_resolved_config", checkpoint_module.sha256(config_path))
    with pytest.raises(ValueError, match="frozen SFT start"):
        checkpoint_module.validate_source_role(role, checkpoint)
    config["actor_rollout_ref"]["model"]["path"] = str(checkpoint_module.SFT)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setitem(checkpoint_module.FROZEN_HASHES, "e3_resolved_config", checkpoint_module.sha256(config_path))

    missing = checkpoint / "actor/extra_state_world_size_1_rank_0.pt"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        checkpoint_module.validate_source_role(role, checkpoint)
    missing.write_bytes(b"extra")

    actor = checkpoint / "actor/model_world_size_1_rank_0.pt"
    actor.write_bytes(b"MODEL")
    with pytest.raises(ValueError, match="source actor hash drifted"):
        checkpoint_module.validate_source_role(role, checkpoint)


def _raw_row(question: dict, response: str = "\\boxed{4}") -> dict:
    import hashlib

    seed = derive_sample_seed(question["question_id"], 0)
    return {
        "protocol_version": "toolcredit_diagnostic_v3",
        "schema_version": "toolcredit_m7_trajectory_schema_v1",
        "evaluator_version": "toolcredit_m7_evaluator_v1",
        "freeze_manifest_sha256": "a" * 64,
        "checkpoint_role": "raw_base_model",
        "checkpoint_identity": {},
        "evaluation_setting": "greedy",
        "question_id": question["question_id"],
        "source": question["source"],
        "split": question["split"],
        "stratum": question["stratum"],
        "question_hash": question["question_sha256_nfkc_whitespace"],
        "sample_index": 0,
        "sample_seed": seed,
        "prompt": question["question"],
        "prompt_hash": hashlib.sha256(question["question"].encode()).hexdigest(),
        "messages": [{"role": "assistant", "content": response}],
        "response": response,
        "response_token_count": 3,
        "termination_reason": "final_answer",
        "tool_metadata": {
            "tool_call_counts": 0,
            "tool_success_count": 0,
            "tool_error_count": 0,
            "tool_parse_error_count": 0,
        },
        "infrastructure_status": "ok",
        "raw_output_sha256": hashlib.sha256(response.encode()).hexdigest(),
    }


def test_atomic_resume_rejects_duplicate_conflict_and_partial(tmp_path: Path) -> None:
    question = load_panel_questions()[0]
    row = _raw_row(question)
    shard = tmp_path / "shard.jsonl"
    append_jsonl_fsync(shard, row)
    accepted = validated_resume_rows(shard, {question["question_id"]: question})
    assert len(accepted) == 1
    append_jsonl_fsync(shard, row)
    with pytest.raises(ValueError, match="duplicate resume key"):
        validated_resume_rows(shard, {question["question_id"]: question})
    conflict = dict(row)
    conflict["response"] = "different"
    conflict["raw_output_sha256"] = __import__("hashlib").sha256(b"different").hexdigest()
    conflict_path = tmp_path / "conflict.jsonl"
    append_jsonl_fsync(conflict_path, row)
    append_jsonl_fsync(conflict_path, conflict)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        validated_resume_rows(conflict_path, {question["question_id"]: question})
    partial = tmp_path / "partial.jsonl"
    partial.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="partial JSONL"):
        validated_resume_rows(partial, {question["question_id"]: question})
