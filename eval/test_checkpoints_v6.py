"""M10 / protocol v6 role validation: fail-closed paths over a synthetic E7 run (no GPU, no merge).

Every check in ``eval/checkpoints_v6.py`` is exercised against a directory tree shaped exactly
like a finished E7 run, so the real materialization step later only has to succeed, not to be
debugged.  The equal-compute step ``N`` is deliberately not 200 and not a multiple of 25 here,
proving the locator is discovered rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval import checkpoints_v6 as v6
from rl.launch.e7_dynamic_filtering import E3_DIFF_PATHS

EQUAL_STEP = 93
FORMAL = v6.FORMAL_STEPS
PER_UPDATE = v6.TRAJECTORIES_PER_UPDATE
E3_TOTAL = v6.E3_TOTAL_TRAJECTORIES


def _config_yaml(sft: Path) -> str:
    return f"""
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
  filter_groups:
    enable: true
    metric: score
    max_num_gen_batches: 4
data:
  train_batch_size: 64
actor_rollout_ref:
  model:
    path: {sft}
  actor:
    fsdp_config:
      param_offload: false
      optimizer_offload: false
  ref:
    fsdp_config:
      param_offload: false
  rollout:
    n: 8
    multi_turn:
      enable: true
trainer:
  total_training_steps: {FORMAL}
  toolcredit_trainer_class: {v6.TRAINER_CLASS_FQN}
"""


def _ledger_row(step: int, cumulative: int, equal: bool = False) -> dict[str, Any]:
    return {
        "effective_step": step,
        "generation_batches_used": 2,
        "cumulative_trajectories": cumulative,
        "equal_compute_checkpoint": equal,
    }


def _checkpoint(path: Path, step: int, cumulative: int, *, equal: bool) -> None:
    (path / "actor").mkdir(parents=True)
    for name in (
        "model_world_size_1_rank_0.pt",
        "optim_world_size_1_rank_0.pt",
        "extra_state_world_size_1_rank_0.pt",
    ):
        (path / "actor" / name).write_bytes(b"weights")
    (path / "actor/fsdp_config.json").write_text("{}", encoding="utf-8")
    (path / "data.pt").write_bytes(b"dl")
    ledger = {
        "schema_version": "toolcredit_e7_ledger_v1",
        "effective_step": step,
        "cumulative_trajectories": cumulative,
        "equal_compute_saved": True,
        "equal_compute_step": EQUAL_STEP,
    }
    if not equal and step == FORMAL:
        ledger["equal_compute_saved"] = True
    (path / "e7_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")


@pytest.fixture
def e7_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run = tmp_path / "rl/runs/e7_dynamic_filtering_20260909_200152"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "analysis").mkdir()
    (run / "recovery/resume_from_50_x").mkdir(parents=True)
    (run / "recovery/resume_from_50_x/recovery.json").write_text("{}", encoding="utf-8")

    # Cumulative trajectories grow by two chunks per step and cross E3's total exactly at EQUAL_STEP.
    rows = []
    cumulative = 0
    for step in range(1, FORMAL + 1):
        cumulative += PER_UPDATE * 2
        if step == EQUAL_STEP:
            cumulative = max(cumulative, E3_TOTAL)
        rows.append(_ledger_row(step, cumulative, equal=step == EQUAL_STEP))
    # Make the step before EQUAL_STEP sit strictly below the threshold.
    rows[EQUAL_STEP - 2]["cumulative_trajectories"] = E3_TOTAL - PER_UPDATE
    rows[EQUAL_STEP - 1]["cumulative_trajectories"] = E3_TOTAL
    (run / "e7_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    _checkpoint(checkpoints / f"global_step_{FORMAL}", FORMAL, rows[-1]["cumulative_trajectories"], equal=False)
    _checkpoint(checkpoints / f"equal_compute_step_{EQUAL_STEP}", EQUAL_STEP, E3_TOTAL, equal=True)
    (checkpoints / "latest_checkpointed_iteration.txt").write_text(str(FORMAL), encoding="utf-8")
    (run / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    sft = tmp_path / "sft/checkpoints/qwen3-1.7b-sft"
    sft.mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(_config_yaml(sft.resolve()), encoding="utf-8")
    gate = {
        "gate": "formal_completion",
        "run_name": run.name,
        "status": "completed",
        "stop_step": FORMAL,
        "ledger": {
            "steps": FORMAL,
            "equal_compute_step": EQUAL_STEP,
            "max_full_batch_streak": 3,
            "mean_generation_batches": 2.4,
            "zero_std_frac_mean": 0.41,
        },
        "candidates": {"steps_checked": FORMAL},
        "equal_compute": {"present": True, "step": EQUAL_STEP},
        "final_checkpoint": {"step": FORMAL},
    }
    (run / "analysis/formal_completion_gate.json").write_text(json.dumps(gate), encoding="utf-8")

    monkeypatch.setattr(v6, "E7_RUN", run)
    monkeypatch.setattr(v6, "E7_CHECKPOINTS", checkpoints)
    monkeypatch.setattr(v6, "E7_LEDGER", run / "e7_ledger.jsonl")
    monkeypatch.setattr(v6, "E7_COMPLETION_GATE", run / "analysis/formal_completion_gate.json")
    monkeypatch.setattr(v6, "SFT", sft)
    return run


def test_diff_paths_literal_equals_the_launcher_gate() -> None:
    assert set(v6.E3_CONTROL_DIFF_PATHS) == E3_DIFF_PATHS


def test_equal_compute_step_is_discovered_not_assumed(e7_run: Path) -> None:
    assert v6.equal_compute_step() == EQUAL_STEP
    assert EQUAL_STEP != FORMAL and EQUAL_STEP % 25 != 0
    assert v6.source_locator(v6.ROLE_EQUAL_COMPUTE).name == f"equal_compute_step_{EQUAL_STEP}"
    assert v6.source_locator(v6.ROLE_STEP_200).name == f"global_step_{FORMAL}"
    with pytest.raises(ValueError, match="unknown v6 role"):
        v6.source_locator("e3_grpo_baseline_step_200")


def test_both_roles_validate_and_report_the_frozen_facts(e7_run: Path) -> None:
    for role in v6.E7_ROLE_NAMES:
        summary = v6.validate_e7_source_role(role)
        assert summary["state"] == "validated"
        assert summary["generation_mode"] == "tir"
        assert summary["run_name"] == e7_run.name
        assert summary["dynamic_filtering"]["filter_groups"] == v6.FILTER_GROUPS
        assert summary["dynamic_filtering"]["equal_compute_step"] == EQUAL_STEP
        assert summary["recovery_events"] == ["resume_from_50_x"]
        assert "recovery_ledger_sha256" in summary
    step200 = v6.validate_e7_source_role(v6.ROLE_STEP_200)
    equal = v6.validate_e7_source_role(v6.ROLE_EQUAL_COMPUTE)
    assert step200["dynamic_filtering"]["effective_step"] == FORMAL
    assert equal["dynamic_filtering"]["effective_step"] == EQUAL_STEP
    assert equal["dynamic_filtering"]["cumulative_trajectories"] == E3_TOTAL
    assert step200["source_locator"] != equal["source_locator"]


def test_locator_swap_and_glob_fail_closed(e7_run: Path) -> None:
    equal_dir = e7_run / f"checkpoints/equal_compute_step_{EQUAL_STEP}"
    with pytest.raises(ValueError, match="explicit canonical path"):
        v6.validate_e7_source_role(v6.ROLE_STEP_200, equal_dir)
    with pytest.raises(ValueError, match="explicit canonical path"):
        v6.validate_e7_source_role(v6.ROLE_EQUAL_COMPUTE, e7_run / f"checkpoints/global_step_{FORMAL}")
    with pytest.raises(ValueError, match="unknown v6 role"):
        v6.validate_e7_source_role("e7_something_else")


def test_two_equal_compute_directories_fail_closed(e7_run: Path) -> None:
    (e7_run / "checkpoints/equal_compute_step_120").mkdir()
    with pytest.raises(ValueError, match="exactly one equal-compute"):
        v6.equal_compute_step()


def test_ledger_disagreeing_with_disk_fails_closed(e7_run: Path) -> None:
    rows = [json.loads(l) for l in (e7_run / "e7_ledger.jsonl").read_text().splitlines()]
    for row in rows:
        row["equal_compute_checkpoint"] = row["effective_step"] == EQUAL_STEP + 1
    (e7_run / "e7_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="run ledger flags"):
        v6.equal_compute_step()


def test_equal_compute_must_be_the_first_step_over_the_e3_total(e7_run: Path) -> None:
    rows = [json.loads(l) for l in (e7_run / "e7_ledger.jsonl").read_text().splitlines()]
    rows[EQUAL_STEP - 2]["cumulative_trajectories"] = E3_TOTAL + 1  # previous step already over
    (e7_run / "e7_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="not the first step"):
        v6.equal_compute_step()


def test_unfinished_run_and_wrong_tracker_fail_closed(e7_run: Path) -> None:
    (e7_run / "status.json").write_text(json.dumps({"state": "segment_completed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not completed"):
        v6.validate_e7_source_role(v6.ROLE_STEP_200)
    (e7_run / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    (e7_run / "checkpoints/latest_checkpointed_iteration.txt").write_text("150", encoding="utf-8")
    with pytest.raises(ValueError, match="tracker is at step"):
        v6.validate_e7_source_role(v6.ROLE_STEP_200)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("algorithm:\n  filter_groups:\n    enable: false", "filter_groups drifted"),
        ("algorithm:\n  filter_groups:\n    metric: acc", "filter_groups drifted"),
        ("actor_rollout_ref:\n  actor:\n    fsdp_config:\n      param_offload: true", "offload off"),
        ("actor_rollout_ref:\n  rollout:\n    multi_turn:\n      enable: false", "multi-turn tool loop"),
        ("trainer:\n  toolcredit_trainer_class: x.y.Z", "dynamic-filter trainer"),
        ("algorithm:\n  norm_adv_by_std_in_grpo: false", "std normalisation"),
        ("actor_rollout_ref:\n  rollout:\n    n: 4", "rollout n"),
    ],
)
def test_training_config_drift_fails_closed(e7_run: Path, mutation: str, message: str) -> None:
    import yaml

    config = yaml.safe_load((e7_run / "resolved_config.yaml").read_text(encoding="utf-8"))

    def deep_update(base: dict, patch: dict) -> dict:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base

    deep_update(config, yaml.safe_load(mutation))
    (e7_run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        v6.validate_e7_source_role(v6.ROLE_STEP_200)


def test_completion_gate_must_pass_and_bind_this_run(e7_run: Path) -> None:
    gate_path = e7_run / "analysis/formal_completion_gate.json"
    good = json.loads(gate_path.read_text(encoding="utf-8"))
    for mutation, message in (
        ({"run_name": "someone_else"}, "does not belong"),
        ({"status": "segment_completed"}, "did not cover"),
        ({"ledger": {**good["ledger"], "steps": 150}}, "does not span"),
        ({"candidates": {"steps_checked": 150}}, "did not verify"),
        ({"ledger": {**good["ledger"], "max_full_batch_streak": 10}}, "four-batch streak"),
        ({"equal_compute": {"present": False}}, "no equal-compute"),
        ({"final_checkpoint": {"step": 175}}, "does not bind"),
    ):
        gate_path.write_text(json.dumps({**good, **mutation}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            v6.validate_e7_source_role(v6.ROLE_STEP_200)
    gate_path.unlink()
    with pytest.raises(FileNotFoundError, match="completion gate has not been run"):
        v6.validate_e7_source_role(v6.ROLE_STEP_200)


def test_missing_checkpoint_files_fail_closed(e7_run: Path) -> None:
    (e7_run / f"checkpoints/global_step_{FORMAL}/data.pt").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        v6.validate_e7_source_role(v6.ROLE_STEP_200)


def test_materialize_refuses_to_overwrite(e7_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "eval/checkpoints/m10_e7_dynamic_filter_step_200_hf"
    target.mkdir(parents=True)
    manifest = tmp_path / "eval/materialization/e7_dynamic_filter_step_200.json"
    manifest.parent.mkdir(parents=True)
    monkeypatch.setitem(v6.MATERIALIZED_PATHS_V6, v6.ROLE_STEP_200, target)
    monkeypatch.setitem(v6.MANIFEST_PATHS_V6, v6.ROLE_STEP_200, manifest)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        v6.materialize_e7(v6.ROLE_STEP_200)
