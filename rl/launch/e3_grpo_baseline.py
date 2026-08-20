"""Resolve, validate, and launch the approved M4 / E3 veRL config."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# datasets creates lock/index files even for local parquet.  Keep those writes in
# the project instead of mutating the user's shared HF cache.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "rl/cache/huggingface"))

import verl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = PROJECT_ROOT / "rl/configs/e3_grpo_baseline.yaml"
RUN_ROOT = PROJECT_ROOT / "rl/runs"
RUN_NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def _absolute(project_path: str) -> str:
    path = Path(project_path)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def compose_config(config_path: Path, run_name: str, smoke: bool = False) -> DictConfig:
    if not RUN_NAME_RE.fullmatch(run_name):
        raise ValueError("run name may contain only letters, digits, '.', '_' and '-'")
    verl_config_dir = Path(verl.__file__).resolve().parent / "trainer/config"
    with initialize_config_dir(config_dir=str(verl_config_dir), version_base=None):
        base = compose(config_name="ppo_trainer")
    # The official example adds Qwen's chat-template kwarg with Hydra's `+`
    # override syntax.  Relax struct only for merging our equivalent YAML overlay;
    # veRL's dataclass validation below still rejects unknown typed fields.
    OmegaConf.set_struct(base, False)
    experiment = OmegaConf.load(config_path)
    config = OmegaConf.merge(base, experiment)

    for key in ("train_files", "val_files"):
        config.data[key] = _absolute(str(config.data[key]))
    config.actor_rollout_ref.model.path = _absolute(str(config.actor_rollout_ref.model.path))
    config.actor_rollout_ref.rollout.multi_turn.tool_config_path = _absolute(
        str(config.actor_rollout_ref.rollout.multi_turn.tool_config_path)
    )
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = _absolute(
        str(config.actor_rollout_ref.rollout.agent.agent_loop_config_path)
    )
    config.reward.custom_reward_function.path = _absolute(str(config.reward.custom_reward_function.path))

    run_dir = RUN_ROOT / run_name
    config.trainer.experiment_name = run_name
    config.trainer.default_local_dir = str(run_dir / "checkpoints")
    config.trainer.rollout_data_dir = str(run_dir / "predictions/train")
    config.trainer.validation_data_dir = str(run_dir / "predictions/validation")

    if smoke:
        config.data.train_files = str(PROJECT_ROOT / "rl/data/e3_smoke_train_20.parquet")
        config.data.val_files = str(PROJECT_ROOT / "rl/data/e3_smoke_val_8.parquet")
        config.data.train_batch_size = 16
        config.data.max_response_length = 1024
        config.actor_rollout_ref.rollout.n = 4
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 8
        config.trainer.total_training_steps = 5
        config.trainer.save_freq = 5
        config.trainer.test_freq = 5
    return config


def validate_run_target(config: DictConfig, run_dir: Path, resume: bool) -> None:
    """Reject accidental reuse and allow only exact, checkpoint-backed resumes."""
    if not resume:
        if run_dir.exists():
            raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
        return
    if not run_dir.is_dir():
        raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if status.get("state") == "completed":
        raise RuntimeError(f"cannot resume run in state {status.get('state')!r}: {run_dir}")
    if status.get("state") == "running":
        try:
            os.kill(int(status["pid"]), 0)
        except (KeyError, TypeError, ValueError, ProcessLookupError):
            pass
        except PermissionError as exc:
            raise RuntimeError(f"cannot establish whether run is still active: {run_dir}") from exc
        else:
            raise RuntimeError(f"cannot resume active run: {run_dir}")
    tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError(f"no complete veRL checkpoint tracker for resume: {tracker}")
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"missing original resolved config: {resolved_path}")
    saved = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
    current = OmegaConf.to_container(config, resolve=True)
    if saved != current:
        raise ValueError("refusing resume because the resolved config has changed")


def validate_m4_config(config: DictConfig, run_dir: Path, resume: bool = False) -> None:
    required_files = [
        Path(config.data.train_files),
        Path(config.data.val_files),
        Path(config.actor_rollout_ref.model.path),
        Path(config.actor_rollout_ref.rollout.multi_turn.tool_config_path),
        Path(config.actor_rollout_ref.rollout.agent.agent_loop_config_path),
        Path(config.reward.custom_reward_function.path),
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing M4 inputs: {missing}")
    validate_run_target(config, run_dir, resume)
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("E3 must use GRPO")
    if config.actor_rollout_ref.actor.use_kl_loss is not True:
        raise ValueError("E3 must keep KL in the actor loss")
    if config.algorithm.use_kl_in_reward is not False:
        raise ValueError("E3 must not also place KL in the reward")
    if config.actor_rollout_ref.rollout.multi_turn.max_user_turns != 4:
        raise ValueError("E3 tool-call budget must be four")

    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    validate_config(
        config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )


def _write_status(run_dir: Path, state: str, **extra: Any) -> None:
    payload = {
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (run_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resource_preflight(config: DictConfig, smoke: bool) -> dict[str, Any]:
    import pyarrow.parquet as pq
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M4 requires exactly one visible CUDA GPU; none is currently available")
    properties = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    disk = shutil.disk_usage(PROJECT_ROOT)
    minimum_disk = 20 << 30 if smoke else 100 << 30
    if disk.free < minimum_disk:
        raise RuntimeError(f"insufficient disk for M4: {disk.free / 2**30:.1f} GiB free")
    expected_train = 20 if smoke else 5203
    expected_val = 8 if smoke else 100
    train_rows = pq.read_metadata(config.data.train_files).num_rows
    val_rows = pq.read_metadata(config.data.val_files).num_rows
    if (train_rows, val_rows) != (expected_train, expected_val):
        raise ValueError(
            f"unexpected M4 parquet counts: train={train_rows}, val={val_rows}; "
            f"expected {expected_train}/{expected_val}"
        )
    return {
        "gpu": properties.name,
        "gpu_total_gib": round(total_bytes / 2**30, 2),
        "gpu_free_gib": round(free_bytes / 2**30, 2),
        "disk_free_gib": round(disk.free / 2**30, 2),
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "checkpoint": config.actor_rollout_ref.model.path,
    }


def archive_interrupted_attempt(run_dir: Path, stamp: str) -> dict[str, Any]:
    """Preserve outputs newer than the checkpoint before veRL overwrites them."""
    tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    checkpoint_step = int(tracker.read_text(encoding="utf-8").strip())
    train_dir = run_dir / "predictions/train"
    persisted_steps = sorted(
        int(path.stem) for path in train_dir.glob("*.jsonl") if path.stem.isdigit()
    )
    recomputed_steps = [step for step in persisted_steps if step > checkpoint_step]
    validation_steps = sorted(
        int(path.stem)
        for path in (run_dir / "predictions/validation").glob("*.jsonl")
        if path.stem.isdigit()
    )
    archive_dir = run_dir / "recovery" / f"resume_from_{checkpoint_step}_{stamp}"
    archive_train_dir = archive_dir / "train"
    archive_train_dir.mkdir(parents=True)
    for step in recomputed_steps:
        shutil.copy2(train_dir / f"{step}.jsonl", archive_train_dir / f"{step}.jsonl")
    for name in ("status.json", "metrics.json", "summary.md", "monitor.log"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, archive_dir / name)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": checkpoint_step,
        "last_persisted_train_step": max(persisted_steps, default=checkpoint_step),
        "recomputed_train_steps": recomputed_steps,
        "preserved_validation_steps": validation_steps,
        "skip_resume_start_validation": True,
    }
    (archive_dir / "recovery.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {**metadata, "archive_dir": str(archive_dir)}


def launch(config: DictConfig, run_dir: Path, preflight: dict[str, Any], resume: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=resume)
    resolved = OmegaConf.to_yaml(config, resolve=True)
    resume_metadata: dict[str, Any] = {}
    if not resume:
        (run_dir / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
        preflight_path = run_dir / "preflight.json"
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        resume_metadata = archive_interrupted_attempt(run_dir, stamp)
        archive_dir = Path(resume_metadata["archive_dir"])
        (archive_dir / "runtime_config.yaml").write_text(resolved, encoding="utf-8")
        preflight_path = run_dir / f"resume_preflight_{stamp}.json"
    preflight_payload = {**preflight, **resume_metadata}
    preflight_path.write_text(json.dumps(preflight_payload, indent=2) + "\n", encoding="utf-8")
    _write_status(
        run_dir,
        "running",
        pid=os.getpid(),
        resumed=resume,
        resume_from_step=resume_metadata.get("checkpoint_step"),
        recovery_archive=resume_metadata.get("archive_dir"),
    )
    os.environ["TENSORBOARD_DIR"] = str(run_dir / "tensorboard")

    try:
        from verl.trainer.main_ppo import main as ppo_main

        ppo_main(config)
    except BaseException as exc:
        _write_status(run_dir, "failed", error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        _write_status(run_dir, "completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="resolve and validate without creating a run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    run_dir = RUN_ROOT / args.run_name
    config = compose_config(config_path, args.run_name, smoke=args.smoke)
    validate_m4_config(config, run_dir, resume=args.resume)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    if args.resume:
        # fit() validates before continuing even after loading a checkpoint.
        # The checkpoint step already has a persisted validation JSONL, so skip
        # that duplicate evaluation while leaving all optimization controls exact.
        config.trainer.val_before_train = False
    preflight = resource_preflight(config, smoke=args.smoke)
    launch(config, run_dir, preflight, resume=args.resume)


if __name__ == "__main__":
    main()
