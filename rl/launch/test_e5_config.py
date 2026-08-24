from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from rl.launch.e3_grpo_baseline import DEFAULT_CONFIG as E3_CONFIG
from rl.launch.e3_grpo_baseline import compose_config
from rl.launch.e5_turn_credit import (
    DEFAULT_CONFIG,
    TurnCreditTaskRunner,
    compose_e5_config,
    validate_e5_config,
)


def _diff_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in left.keys() | right.keys():
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_diff_paths(left[key], right[key], child))
        return paths
    return set() if left == right else {prefix}


def _resolved(config) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    assert isinstance(value, dict)
    return value


def test_e5_formal_diff_is_only_preregistered_treatment_and_metadata() -> None:
    e3 = _resolved(compose_config(E3_CONFIG, "same_name"))
    e5 = _resolved(compose_e5_config("same_name"))
    assert _diff_paths(e3, e5) == {
        "turn_credit",
        "actor_rollout_ref.rollout.agent.default_agent_loop",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
        "actor_rollout_ref.rollout.trace.project_name",
        "trainer.project_name",
    }
    assert e5["turn_credit"]["beta"] == 0.5
    assert e5["algorithm"]["adv_estimator"] == "grpo"


def test_e5_smoke_keeps_full_e3_experimental_controls() -> None:
    formal = _resolved(compose_e5_config("same_name"))
    smoke = _resolved(compose_e5_config("same_name", smoke=True))
    assert _diff_paths(formal, smoke) == {
        "trainer.total_training_steps",
        "trainer.save_freq",
        "trainer.test_freq",
    }
    assert smoke["data"]["train_batch_size"] == 64
    assert smoke["data"]["max_response_length"] == 3072
    assert smoke["actor_rollout_ref"]["rollout"]["n"] == 8
    assert Path(smoke["data"]["train_files"]).name == "e3_train.parquet"
    assert Path(smoke["data"]["val_files"]).name == "e3_val_math500_100.parquet"


def test_e5_config_validation_and_e3_runtime_identity(tmp_path: Path) -> None:
    from verl.trainer.ppo import ray_trainer

    compute_before = ray_trainer.compute_advantage
    log_before = ray_trainer.RayPPOTrainer._log_rollout_data
    config = compose_e5_config("e5_validate")
    validate_e5_config(config, tmp_path / "missing_run")
    assert ray_trainer.compute_advantage is compute_before
    assert ray_trainer.RayPPOTrainer._log_rollout_data is log_before
    assert not hasattr(ray_trainer.compute_advantage, "__toolcredit_wrapper_proof__")


def test_e5_config_path_is_independent() -> None:
    assert DEFAULT_CONFIG.name == "e5_turn_credit.yaml"


def test_task_runner_installs_and_restores_exact_runtime_hooks(tmp_path: Path, monkeypatch) -> None:
    from verl.trainer.main_ppo import TaskRunner
    from verl.trainer.ppo import ray_trainer

    compute_before = ray_trainer.compute_advantage
    log_before = ray_trainer.RayPPOTrainer._log_rollout_data
    observed: dict[str, bool] = {}

    def fake_native_run(self, config) -> None:
        del self, config
        observed["compute_wrapped"] = hasattr(
            ray_trainer.compute_advantage, "__toolcredit_wrapper_proof__"
        )
        observed["log_wrapped"] = ray_trainer.RayPPOTrainer._log_rollout_data is not log_before

    monkeypatch.setattr(TaskRunner, "run", fake_native_run)
    concrete = type("UnitTestE5TaskRunner", (TurnCreditTaskRunner, TaskRunner), {})
    config = compose_e5_config("wrapper_unit")
    config.trainer.default_local_dir = str(tmp_path / "checkpoints")
    concrete().run(config)

    assert observed == {"compute_wrapped": True, "log_wrapped": True}
    assert ray_trainer.compute_advantage is compute_before
    assert ray_trainer.RayPPOTrainer._log_rollout_data is log_before
    proof = json.loads((tmp_path / "wrapper_installation.json").read_text())
    assert proof["installed"] is True
    assert proof["restored"] is True
    assert proof["proof_id"].startswith("e5-wrapper-v1:")
