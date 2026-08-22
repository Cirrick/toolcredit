from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from rl.launch.e3_grpo_baseline import DEFAULT_CONFIG, compose_config

E6_CONFIG = Path(__file__).resolve().parents[1] / "configs/e6_nomask.yaml"
E4A_CONFIG = Path(__file__).resolve().parents[1] / "configs/e4a_exec_only.yaml"
E4B_CONFIG = Path(__file__).resolve().parents[1] / "configs/e4b_joint_shaping.yaml"


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


def _resolved(config_path: Path) -> dict[str, Any]:
    config = compose_config(config_path, "m5_config_diff")
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def test_e6_formal_config_diff_is_preregistered_minimum() -> None:
    e3 = _resolved(DEFAULT_CONFIG)
    e6 = _resolved(E6_CONFIG)
    assert _diff_paths(e3, e6) == {
        "actor_rollout_ref.rollout.agent.default_agent_loop",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
        "actor_rollout_ref.rollout.trace.project_name",
        "trainer.project_name",
        "trainer.total_training_steps",
    }
    assert e6["trainer"]["total_training_steps"] == 80
    assert e6["actor_rollout_ref"]["rollout"]["agent"]["default_agent_loop"] == (
        "toolcredit_nomask_agent"
    )


def test_e3_to_e4a_formal_config_diff_is_only_exec_reward_and_metadata() -> None:
    e3 = _resolved(DEFAULT_CONFIG)
    e4a = _resolved(E4A_CONFIG)
    assert _diff_paths(e3, e4a) == {
        "actor_rollout_ref.rollout.trace.project_name",
        "reward.custom_reward_function.reward_kwargs",
        "trainer.project_name",
    }
    assert e4a["trainer"]["total_training_steps"] == 200
    assert e4a["reward"]["custom_reward_function"]["reward_kwargs"] == {
        "lambda_exec": 0.2,
        "lambda_budget": 0.0,
        "budget": 3,
    }


def test_e4a_to_e4b_formal_diff_is_only_budget_penalty() -> None:
    e4a = _resolved(E4A_CONFIG)
    e4b = _resolved(E4B_CONFIG)
    assert _diff_paths(e4a, e4b) == {
        "reward.custom_reward_function.reward_kwargs.lambda_budget"
    }
    assert e4b["reward"]["custom_reward_function"]["reward_kwargs"] == {
        "lambda_exec": 0.2,
        "lambda_budget": 0.1,
        "budget": 3,
    }


def test_e4_smokes_keep_distinct_reward_kwargs() -> None:
    a = compose_config(E4A_CONFIG, "e4a_smoke_config", smoke=True)
    b = compose_config(E4B_CONFIG, "e4b_smoke_config", smoke=True)
    assert a.trainer.total_training_steps == b.trainer.total_training_steps == 5
    assert a.reward.custom_reward_function.reward_kwargs.lambda_exec == 0.2
    assert b.reward.custom_reward_function.reward_kwargs.lambda_exec == 0.2
    assert a.reward.custom_reward_function.reward_kwargs.lambda_budget == 0.0
    assert b.reward.custom_reward_function.reward_kwargs.lambda_budget == 0.1
