from __future__ import annotations

from pathlib import Path

import json
import os

import pytest
from omegaconf import OmegaConf

from rl.launch.e3_grpo_baseline import (
    DEFAULT_CONFIG,
    archive_interrupted_attempt,
    compose_config,
    validate_run_target,
)


def test_full_config_matches_approved_m4_controls() -> None:
    config = compose_config(DEFAULT_CONFIG, "e3_config_test")
    assert config.algorithm.adv_estimator == "grpo"
    assert config.data.train_batch_size == 64
    assert config.data.max_response_length == 3072
    assert config.data.filter_overlong_prompts is True
    assert config.data.dataloader_num_workers == 0
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.actor_rollout_ref.actor.optim.lr == 1.0e-6
    assert config.actor_rollout_ref.actor.clip_ratio_high == 0.28
    assert config.actor_rollout_ref.actor.kl_loss_coef == 0.001
    assert config.actor_rollout_ref.rollout.multi_turn.max_user_turns == 4
    assert config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns == 5
    assert config.actor_rollout_ref.rollout.agent.default_agent_loop == "toolcredit_agent"
    assert Path(config.actor_rollout_ref.rollout.agent.agent_loop_config_path).name == (
        "agent_loop_config.yaml"
    )
    assert config.trainer.total_training_steps == 200
    assert config.trainer.save_freq == config.trainer.test_freq == 25


def test_smoke_config_keeps_m4_tool_and_reward_wiring() -> None:
    full = compose_config(DEFAULT_CONFIG, "e3_full_test")
    smoke = compose_config(DEFAULT_CONFIG, "e3_smoke_test", smoke=True)
    assert smoke.trainer.total_training_steps == 5
    assert smoke.actor_rollout_ref.rollout.n == 4
    assert Path(smoke.actor_rollout_ref.rollout.multi_turn.tool_config_path) == Path(
        full.actor_rollout_ref.rollout.multi_turn.tool_config_path
    )
    assert Path(smoke.reward.custom_reward_function.path) == Path(full.reward.custom_reward_function.path)


def test_resume_requires_failed_run_with_matching_config(tmp_path: Path) -> None:
    config = compose_config(DEFAULT_CONFIG, "resume_test")
    run_dir = tmp_path / "resume_test"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints/latest_checkpointed_iteration.txt").write_text("25\n")
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    (run_dir / "status.json").write_text(json.dumps({"state": "failed"}))
    validate_run_target(config, run_dir, resume=True)

    (run_dir / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}))
    with pytest.raises(RuntimeError, match="cannot resume active"):
        validate_run_target(config, run_dir, resume=True)

    (run_dir / "status.json").write_text(json.dumps({"state": "running", "pid": 999999999}))
    validate_run_target(config, run_dir, resume=True)


def test_archive_interrupted_attempt_preserves_only_post_checkpoint_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "resume_test"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints/latest_checkpointed_iteration.txt").write_text("150\n")
    (run_dir / "checkpoints/global_step_150").mkdir()
    incomplete_checkpoint = run_dir / "checkpoints/global_step_175"
    incomplete_checkpoint.mkdir()
    (incomplete_checkpoint / "partial.pt").write_text("partial\n")
    train_dir = run_dir / "predictions/train"
    validation_dir = run_dir / "predictions/validation"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    for step in (149, 150, 151, 168):
        (train_dir / f"{step}.jsonl").write_text(f"step={step}\n")
    for step in (0, 25, 150):
        (validation_dir / f"{step}.jsonl").write_text(f"step={step}\n")
    (run_dir / "status.json").write_text(json.dumps({"state": "running"}))

    metadata = archive_interrupted_attempt(run_dir, "20260820_000000")
    archive_dir = Path(metadata["archive_dir"])
    assert sorted(path.name for path in (archive_dir / "train").glob("*.jsonl")) == [
        "151.jsonl",
        "168.jsonl",
    ]
    assert metadata["checkpoint_step"] == 150
    assert metadata["last_persisted_train_step"] == 168
    assert metadata["archived_incomplete_checkpoint_steps"] == [175]
    assert metadata["preserved_validation_steps"] == [0, 25, 150]
    assert (archive_dir / "status.json").is_file()
    assert not incomplete_checkpoint.exists()
    assert (
        archive_dir / "incomplete_checkpoints/global_step_175/partial.pt"
    ).read_text() == "partial\n"
