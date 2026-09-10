"""M10 / E7 launcher gates: config diff (plan §3.2), modes, segments, driver binding (STEP0 §4b)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from rl.custom.dynamic_filter_trainer import (
    CANDIDATES_DIRNAME,
    LEDGER_FILENAME,
    LEDGER_JSONL,
    STOP_REQUEST_FILENAME,
    TRAINER_CLASS_FQN,
    DynamicFilterRayPPOTrainer,
)
from rl.launch.e3_grpo_baseline import DEFAULT_CONFIG as E3_CONFIG
from rl.launch.e3_grpo_baseline import compose_config
from rl.launch.e5v2_conserved_credit import diff_paths, resolved
from rl.launch.e7_dynamic_filtering import (
    E3_DIFF_PATHS,
    M10_PINNED_VERL_HASHES,
    OFFLOAD_PATHS,
    SMOKE_DIFF_PATHS,
    DynamicFilteringTaskRunner,
    _final_status,
    archive_e7_evidence,
    compose_e7_config,
    config_diff_gate,
    disk_estimate,
    validate_e7_config,
    validate_segment_request,
    verl_source_hashes,
)


def test_e3_to_e7_diff_is_exactly_the_preregistered_set() -> None:
    gate = config_diff_gate("same_name")
    assert set(gate["e3_to_e7"]) == E3_DIFF_PATHS
    assert gate["filter_groups"] == {"enable": True, "metric": "score", "max_num_gen_batches": 4}
    e3 = resolved(compose_config(E3_CONFIG, "same_name"))
    e7 = resolved(compose_e7_config("same_name"))
    for section in ("data", "reward"):
        assert e3[section] == e7[section]
    algo_e3 = dict(e3["algorithm"])
    algo_e7 = dict(e7["algorithm"])
    algo_e7.pop("filter_groups")
    assert algo_e3 == algo_e7
    assert e3["actor_rollout_ref"]["rollout"]["multi_turn"] == e7["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert e3["actor_rollout_ref"]["rollout"]["agent"] == e7["actor_rollout_ref"]["rollout"]["agent"]
    for path in OFFLOAD_PATHS:
        node: Any = e7
        for key in path.split("."):
            node = node[key]
        assert node is False
    assert e7["trainer"]["toolcredit_trainer_class"] == TRAINER_CLASS_FQN
    assert e7["trainer"]["total_training_steps"] == 200 and e7["trainer"]["max_actor_ckpt_to_keep"] == 3


def test_modes_only_change_schedule_and_checkpoint_frequency() -> None:
    formal = resolved(compose_e7_config("same_name"))
    smoke = resolved(compose_e7_config("same_name", mode="smoke"))
    check = resolved(compose_e7_config("same_name", mode="resume_check"))
    assert diff_paths(formal, smoke) == SMOKE_DIFF_PATHS
    assert diff_paths(formal, check) == SMOKE_DIFF_PATHS
    assert smoke["trainer"]["save_freq"] == -1 and smoke["trainer"]["total_training_steps"] == 5
    assert check["trainer"]["save_freq"] == 2 and check["trainer"]["total_training_steps"] == 2
    assert smoke["data"]["train_batch_size"] == 64 and smoke["actor_rollout_ref"]["rollout"]["n"] == 8
    assert Path(smoke["data"]["train_files"]).name == "e3_train.parquet"
    with pytest.raises(ValueError):
        compose_e7_config("same_name", mode="bogus")


def test_validate_rejects_treatment_and_offload_drift(tmp_path: Path) -> None:
    good = compose_e7_config("e7_validate")
    validate_e7_config(good, tmp_path / "missing_run")
    cases = [
        (("algorithm", "filter_groups", "enable"), False, "enable"),
        (("algorithm", "filter_groups", "metric"), "acc", "score"),
        (("algorithm", "filter_groups", "max_num_gen_batches"), 3, "four"),
        (("actor_rollout_ref", "actor", "fsdp_config", "param_offload"), True, "offload"),
        (("actor_rollout_ref", "ref", "fsdp_config", "param_offload"), True, "offload"),
        (("trainer", "toolcredit_trainer_class"), "x.y.Z", "marker"),
        (("actor_rollout_ref", "rollout", "n"), 4, "n must"),
        (("algorithm", "norm_adv_by_std_in_grpo"), False, "std"),
    ]
    for path, value, message in cases:
        config = compose_e7_config("e7_validate")
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            validate_e7_config(config, tmp_path / "missing_run")


def test_pinned_sources_and_disk_estimate() -> None:
    hashes = verl_source_hashes()
    assert set(hashes) == set(M10_PINNED_VERL_HASHES)
    assert "verl/utils/checkpoint/checkpoint_manager.py" in hashes
    formal = disk_estimate("formal", check=False)
    smoke = disk_estimate("smoke", check=False)
    check = disk_estimate("resume_check", check=False)
    assert formal["remaining_peak_write_gib"] > 4 * formal["e3_checkpoint_gib"]
    assert formal["disk_minimum_gib"] == pytest.approx(formal["remaining_peak_write_gib"] + 30, abs=0.01)
    assert smoke["remaining_peak_write_gib"] < 5
    assert check["remaining_peak_write_gib"] > 2 * formal["e3_checkpoint_gib"]


def _run_dir(tmp_path: Path, tracker: int | None, total: int = 200, save_freq: int = 25, train_steps: int = 0) -> tuple[Path, Any]:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "predictions/train").mkdir(parents=True)
    for step in range(1, train_steps + 1):
        (run / "predictions/train" / f"{step}.jsonl").write_text("{}\n")
    if tracker is not None:
        (run / "checkpoints/latest_checkpointed_iteration.txt").write_text(str(tracker))
        (run / "checkpoints" / f"global_step_{tracker}").mkdir()
        (run / "checkpoints" / f"global_step_{tracker}" / LEDGER_FILENAME).write_text("{}")
    config = OmegaConf.create({"trainer": {"total_training_steps": total, "save_freq": save_freq}})
    return run, config


def test_final_status_distinguishes_segments_from_completion(tmp_path: Path) -> None:
    run, config = _run_dir(tmp_path, tracker=50)
    assert _final_status(run, config, 50) == ("segment_completed", {"segment_stop_step": 50, "stopped_by_request": False})
    with pytest.raises(RuntimeError):
        _final_status(run, config, 100)
    (run / STOP_REQUEST_FILENAME).write_text("{}")
    assert _final_status(run, config, 100)[0] == "segment_completed"
    run2, config2 = _run_dir(tmp_path / "b", tracker=200)
    assert _final_status(run2, config2, 200)[0] == "completed"
    assert _final_status(run2, config2, None)[0] == "completed"
    smoke_run, smoke_config = _run_dir(tmp_path / "c", tracker=None, total=5, save_freq=-1, train_steps=5)
    assert _final_status(smoke_run, smoke_config, None) == ("completed", {"final_step": 5, "checkpoint": None})
    short_run, short_config = _run_dir(tmp_path / "d", tracker=None, total=5, save_freq=-1, train_steps=3)
    with pytest.raises(RuntimeError):
        _final_status(short_run, short_config, None)


def test_segment_request_validation(tmp_path: Path) -> None:
    run, config = _run_dir(tmp_path, tracker=50)
    validate_segment_request(run, config, resume=True, stop_at_step=100)
    with pytest.raises(ValueError, match="beyond"):
        validate_segment_request(run, config, resume=True, stop_at_step=50)
    with pytest.raises(ValueError, match="within"):
        validate_segment_request(run, config, resume=False, stop_at_step=250)
    smoke = OmegaConf.create({"trainer": {"total_training_steps": 5, "save_freq": -1}})
    with pytest.raises(ValueError, match="save_freq"):
        validate_segment_request(run, smoke, resume=False, stop_at_step=3)
    (run / "checkpoints/global_step_50" / LEDGER_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="ledger"):
        validate_segment_request(run, config, resume=True, stop_at_step=100)


def test_archive_e7_evidence_copies_ledger_and_newer_candidates(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "predictions" / CANDIDATES_DIRNAME).mkdir(parents=True)
    for step in (49, 50, 51, 52):
        (run / "predictions" / CANDIDATES_DIRNAME / f"{step}.jsonl").write_text("{}\n")
    (run / "predictions" / CANDIDATES_DIRNAME / "52.underfilled.jsonl").write_text("{}\n")
    (run / LEDGER_JSONL).write_text("{}\n")
    (run / STOP_REQUEST_FILENAME).write_text("{}\n")
    archive = run / "recovery/resume_from_50_x"
    meta = archive_e7_evidence(run, archive, 50)
    assert meta["e7_archived_candidate_steps"] == [51, 52, 52]
    assert LEDGER_JSONL in meta["e7_archived_files"]
    assert meta["e7_consumed_stop_request"] and not (run / STOP_REQUEST_FILENAME).exists()
    assert (archive / f"{STOP_REQUEST_FILENAME}.consumed").exists()
    assert sorted(p.name for p in (archive / CANDIDATES_DIRNAME).iterdir()) == ["51.jsonl", "52.jsonl", "52.underfilled.jsonl"]


def test_driver_binding_is_installed_during_run_and_restored_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """STEP0 §4 option (b): ``main_ppo.RayPPOTrainer`` resolves to the E7 trainer only inside ``run``."""
    from verl.trainer import main_ppo
    from verl.trainer.main_ppo import TaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    observed: dict[str, Any] = {}

    def fake_run(self: Any, config: Any) -> None:
        observed["bound"] = main_ppo.RayPPOTrainer
        observed["stop"] = DynamicFilterRayPPOTrainer.E7_STOP_AT_STEP
        raise RuntimeError("stop before workers")

    monkeypatch.setattr(TaskRunner, "run", fake_run)
    concrete = type("E7DynamicFilteringTaskRunner", (DynamicFilteringTaskRunner, TaskRunner), {"E7_STOP_AT_STEP": 50})
    config = OmegaConf.create({"trainer": {"default_local_dir": str(tmp_path / "run/checkpoints")}})
    runner = concrete()
    with pytest.raises(RuntimeError, match="stop before workers"):
        runner.run(config)
    assert observed["bound"] is DynamicFilterRayPPOTrainer and observed["stop"] == 50
    assert main_ppo.RayPPOTrainer is RayPPOTrainer
    proof = json.loads((tmp_path / "run/e7_boundary_installation.json").read_text())
    assert proof["installed"] and proof["restored"] and proof["bound_to"] == TRAINER_CLASS_FQN and proof["stop_at_step"] == 50
    assert set(proof["verl_source_hashes"]) == set(M10_PINNED_VERL_HASHES)
    DynamicFilterRayPPOTrainer.E7_STOP_AT_STEP = None
    # Stacking is refused.
    monkeypatch.setattr(main_ppo, "RayPPOTrainer", DynamicFilterRayPPOTrainer)
    with pytest.raises(RuntimeError, match="stack"):
        concrete().run(config)
