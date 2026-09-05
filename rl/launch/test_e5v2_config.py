"""M8 / E5-v2 launcher gates: config diff (plan §5), frozen treatment, runtime hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.launch.e3_grpo_baseline import DEFAULT_CONFIG as E3_CONFIG
from rl.launch.e3_grpo_baseline import compose_config
from rl.launch.e5_turn_credit import compose_e5_config
from rl.launch.e5v2_conserved_credit import (
    DEFAULT_CONFIG,
    E3_CONTROL_DIFF_PATHS,
    METADATA_DIFF_PATHS,
    SMOKE_DIFF_PATHS,
    ConservedTurnCreditTaskRunner,
    compose_e5v2_config,
    config_diff_gate,
    diff_paths,
    disk_estimate,
    resume_disk_estimate,
    resolved,
    validate_e5v2_config,
)


def test_v1_to_v2_diff_is_only_turn_credit_and_metadata() -> None:
    gate = config_diff_gate("same_name")
    for path in gate["v1_to_v2"]:
        assert path.startswith("turn_credit.") or path in METADATA_DIFF_PATHS, path
    assert {"turn_credit.variant", "turn_credit.adoption_version", "turn_credit.heuristic_version"} <= set(gate["v1_to_v2"])
    assert set(gate["e3_to_v2"]) == E3_CONTROL_DIFF_PATHS
    v1 = resolved(compose_e5_config("same_name"))
    v2 = resolved(compose_e5v2_config("same_name"))
    for section in ("data", "algorithm", "reward"):
        assert v1[section] == v2[section]
    assert v1["actor_rollout_ref"]["actor"] == v2["actor_rollout_ref"]["actor"]
    assert v1["actor_rollout_ref"]["rollout"]["multi_turn"] == v2["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert v2["turn_credit"]["beta"] == 0.5 and v2["turn_credit"]["variant"] == "conserved_v2"
    e3 = resolved(compose_config(E3_CONFIG, "same_name"))
    assert e3["trainer"]["total_training_steps"] == v2["trainer"]["total_training_steps"] == 200


def test_smoke_only_changes_step_count_and_disables_checkpoints() -> None:
    formal = resolved(compose_e5v2_config("same_name"))
    smoke = resolved(compose_e5v2_config("same_name", smoke=True))
    assert diff_paths(formal, smoke) == SMOKE_DIFF_PATHS
    assert smoke["trainer"]["save_freq"] == -1 and smoke["trainer"]["total_training_steps"] == 5
    assert smoke["data"]["train_batch_size"] == 64 and smoke["actor_rollout_ref"]["rollout"]["n"] == 8
    assert Path(smoke["data"]["train_files"]).name == "e3_train.parquet"


def test_validate_rejects_variant_beta_and_loop_drift(tmp_path: Path) -> None:
    good = compose_e5v2_config("e5v2_validate")
    validate_e5v2_config(good, tmp_path / "missing_run")
    for field, value, message in (
        ("variant", "position_baseline_v1", "variant"),
        ("beta", 0.4, "beta"),
        ("adoption_version", "toolcredit_adoption_v1", "adoption"),
        ("gate_zero_variance", False, "gating"),
    ):
        config = compose_e5v2_config("e5v2_validate")
        setattr(config.turn_credit, field, value)
        with pytest.raises(ValueError, match=message):
            validate_e5v2_config(config, tmp_path / "missing_run")
    config = compose_e5v2_config("e5v2_validate")
    config.actor_rollout_ref.rollout.agent.default_agent_loop = "toolcredit_turn_credit_agent"
    with pytest.raises(ValueError, match="agent loop"):
        validate_e5v2_config(config, tmp_path / "missing_run")


def test_config_path_and_disk_estimate() -> None:
    assert DEFAULT_CONFIG.name == "e5v2_conserved_credit.yaml"
    estimate = disk_estimate(smoke=False, check=False)
    assert estimate["remaining_peak_write_gib"] >= 103
    assert estimate["disk_minimum_gib"] == pytest.approx(estimate["remaining_peak_write_gib"] + 30)


def test_task_runner_installs_v2_variant_and_restores(tmp_path: Path, monkeypatch) -> None:
    from verl.trainer.main_ppo import TaskRunner
    from verl.trainer.ppo import ray_trainer

    from rl.e5_recovery import validate_wrapper_proof_payload

    compute_before = ray_trainer.compute_advantage
    log_before = ray_trainer.RayPPOTrainer._log_rollout_data
    observed: dict[str, object] = {}

    def fake_native_run(self, config) -> None:
        del self, config
        observed["variant"] = getattr(ray_trainer.compute_advantage, "__toolcredit_variant__", None)
        observed["log_wrapped"] = ray_trainer.RayPPOTrainer._log_rollout_data is not log_before

    monkeypatch.setattr(TaskRunner, "run", fake_native_run)
    concrete = type("UnitTestE5V2TaskRunner", (ConservedTurnCreditTaskRunner, TaskRunner), {})
    config = compose_e5v2_config("wrapper_unit_v2")
    config.trainer.default_local_dir = str(tmp_path / "checkpoints")
    concrete().run(config)
    assert observed == {"variant": "conserved_v2", "log_wrapped": True}
    assert ray_trainer.compute_advantage is compute_before
    assert ray_trainer.RayPPOTrainer._log_rollout_data is log_before
    proof = json.loads((tmp_path / "wrapper_installation.json").read_text())
    assert proof["installed"] is True and proof["restored"] is True
    assert proof["proof_id"].startswith("e5v2-wrapper-v1:")
    assert proof["boundary_version"] == "e5v2_runtime_wrapper_v1"
    assert proof["variant"] == "conserved_v2" and proof["heuristic_version"] == "toolcredit_adoption_v2"
    validate_wrapper_proof_payload(proof)


def test_frozen_v1_files_are_untouched_by_m8() -> None:
    """Freeze v2/v3 closures hash the v1 modules; M8 must live in new files."""
    from eval.build_diagnostic_freeze_v2 import verify

    verify()


def test_resume_disk_estimate_measures_run_state(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "checkpoints/global_step_125").mkdir(parents=True)
    (run / "checkpoints/latest_checkpointed_iteration.txt").write_text("125\n")
    (run / "checkpoints/global_step_125/actor.pt").write_bytes(b"x" * 1024)
    (run / "predictions/train").mkdir(parents=True)
    for step in range(1, 148):
        (run / "predictions/train" / f"{step}.jsonl").write_bytes(b"y" * 100)
    (run / "predictions/validation").mkdir(parents=True)
    (run / "predictions/validation/0.jsonl").write_bytes(b"z" * 50)
    estimate = resume_disk_estimate(run, total_steps=200)
    assert estimate["checkpoint_step"] == 125 and estimate["remaining_steps"] == 75
    expected_bytes = 1024 + 75 * 100 + 3 * 50 + 22 * 100
    assert estimate["remaining_peak_write_bytes"] == expected_bytes
    assert estimate["disk_minimum_gib"] == pytest.approx(expected_bytes / 2**30 + 30, abs=0.01)
    assert estimate["mode"] == "resume"
