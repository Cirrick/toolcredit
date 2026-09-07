"""Resolve, validate, and launch the approved M9 / E3-NoTool control arm.

E3-NoTool is E3 with exactly one thing removed: the tool.  Everything that is not the
tool is reused from the frozen E3 launcher (``rl/launch/e3_grpo_baseline.py``, locked by
the M6/M7 freeze closures and left untouched): config composition, the resume guard, the
five-artifact launch discipline and the recovery archive.

Two E3 helpers cannot be reused as-is, which is why this module carries its own validator:

* ``validate_m4_config`` hard-asserts ``multi_turn.max_user_turns == 4`` and requires
  ``agent.agent_loop_config_path`` to exist on disk.  E3-NoTool has no custom agent loop.
* ``resource_preflight`` uses a flat disk floor.  Plan §7 step 1 requires the
  ``remaining_peak_write + 30 GiB`` figure to be recomputed from measured E3 artifacts.

The structural diff helpers are imported verbatim from the M8 launcher so that the
"only the preregistered fields may differ" gate has identical semantics across milestones.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from rl.launch.e3_grpo_baseline import (
    DEFAULT_CONFIG as E3_CONFIG,
    PROJECT_ROOT,
    RUN_ROOT,
    _sha256,
    compose_config,
    launch,
    validate_run_target,
)
from rl.launch.e5_turn_credit import PINNED_VERL_HASHES
from rl.launch.e5v2_conserved_credit import diff_paths, resolved, resume_disk_estimate

DEFAULT_CONFIG = PROJECT_ROOT / "rl/configs/e3notool_grpo.yaml"
AGENT_LOOP_NAME = "single_turn_agent"
E3_RUN = PROJECT_ROOT / "rl/runs/e3_grpo_baseline_20260819_224555"
DISK_SAFETY_MARGIN_GIB = 30
CHECKPOINTS_KEPT = 3

# Plan §2: the complete set of resolved-config paths E3 -> E3-NoTool may differ on.
# ``rollout.trace.project_name`` interpolates ``trainer.project_name`` in veRL's base config.
E3_DIFF_PATHS = {
    "actor_rollout_ref.rollout.multi_turn.enable",
    "actor_rollout_ref.rollout.agent.default_agent_loop",
    "actor_rollout_ref.rollout.agent.agent_loop_config_path",
    "actor_rollout_ref.rollout.trace.project_name",
    "trainer.project_name",
}
# Plan §2 additionally tolerates any ``multi_turn.*`` subfield that ``enable: false``
# renders inert.  The config above changes none of them; the prefix keeps the gate honest
# about what is *allowed* rather than only what is *observed*.
ALLOWED_DIFF_PREFIXES = ("actor_rollout_ref.rollout.multi_turn.",)
SMOKE_DIFF_PATHS = {"trainer.total_training_steps", "trainer.save_freq", "trainer.test_freq"}

# The native single-turn loop is now part of the treatment surface, so it is pinned too.
M9_PINNED_VERL_HASHES = {
    **PINNED_VERL_HASHES,
    "verl/experimental/agent_loop/single_turn_agent_loop.py": (
        "066fade7d9c73076937af7d263fb9a1cefb38903c667ced79238ce567323758c"
    ),
}


def verl_source_hashes() -> dict[str, str]:
    """Re-verify every pinned veRL source file, including the native single-turn loop."""
    import verl

    root = Path(verl.__file__).resolve().parent.parent
    actual: dict[str, str] = {}
    for relative, expected in M9_PINNED_VERL_HASHES.items():
        digest = _sha256(root / relative)
        if digest != expected:
            raise RuntimeError(f"pinned veRL source drift for {relative}: {digest} != {expected}")
        actual[relative] = digest
    return actual


def compose_e3notool_config(run_name: str, smoke: bool = False) -> DictConfig:
    """Compose E3-NoTool with the full E3 data/rollout controls, even for smoke.

    ``compose_config`` absolutises ``agent.agent_loop_config_path`` unconditionally, which
    turns our ``null`` into a bogus project-relative path.  E3-NoTool registers no custom
    agent loop, so the field is restored to null immediately and never reaches disk.
    """
    config = compose_config(DEFAULT_CONFIG, run_name, smoke=False)
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = None
    if smoke:
        config.trainer.total_training_steps = 5
        # Plan §7 step 2: the smoke must not retain a ~21 GiB checkpoint.
        config.trainer.save_freq = -1
        config.trainer.test_freq = 5
    return config


def config_diff_gate(run_name: str) -> dict[str, list[str]]:
    """Fail closed unless E3 -> E3-NoTool differs only where plan §2 allows."""
    notool = resolved(compose_e3notool_config(run_name))
    e3 = resolved(compose_config(E3_CONFIG, run_name))
    observed = diff_paths(e3, notool)
    illegal = {
        path
        for path in observed
        if path not in E3_DIFF_PATHS and not path.startswith(ALLOWED_DIFF_PREFIXES)
    }
    if illegal:
        raise ValueError(f"E3 -> E3-NoTool config diff gate failed: {sorted(illegal)}")
    if observed != E3_DIFF_PATHS:
        raise ValueError(
            f"E3 -> E3-NoTool config diff is not the preregistered set: "
            f"{sorted(observed)} != {sorted(E3_DIFF_PATHS)}"
        )
    return {"e3_to_notool": sorted(observed)}


def validate_e3notool_config(config: DictConfig, run_dir: Path, resume: bool = False) -> None:
    """Fail closed if the tool is still reachable or any E3 control drifted."""
    required_files = [
        Path(config.data.train_files),
        Path(config.data.val_files),
        Path(config.actor_rollout_ref.model.path),
        Path(config.reward.custom_reward_function.path),
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing M9 inputs: {missing}")
    validate_run_target(config, run_dir, resume)

    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("E3-NoTool must use GRPO")
    if config.actor_rollout_ref.actor.use_kl_loss is not True:
        raise ValueError("E3-NoTool must keep KL in the actor loss")
    if config.algorithm.use_kl_in_reward is not False:
        raise ValueError("E3-NoTool must not also place KL in the reward")

    rollout = config.actor_rollout_ref.rollout
    if rollout.multi_turn.enable is not False:
        raise ValueError("E3-NoTool must disable the multi-turn tool loop")
    if rollout.agent.default_agent_loop != AGENT_LOOP_NAME:
        raise ValueError(f"E3-NoTool must use veRL's native {AGENT_LOOP_NAME} agent loop")
    if rollout.agent.agent_loop_config_path is not None:
        raise ValueError("E3-NoTool must not register a custom agent loop")
    if rollout.n != 8:
        raise ValueError("E3-NoTool rollout n must remain 8, including smoke")
    if rollout.temperature != 1.0:
        raise ValueError("E3-NoTool sampling temperature must match E3")
    if config.data.train_batch_size != 64 or config.data.max_response_length != 3072:
        raise ValueError("E3-NoTool batch/response controls must match E3, including smoke")

    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    validate_config(
        config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    config_diff_gate(str(config.trainer.experiment_name))


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def disk_estimate(smoke: bool, check: bool = True) -> dict[str, Any]:
    """Plan §7 step 1: recompute ``remaining_peak_write + 30 GiB`` from measured E3 artifacts."""
    if smoke:
        peak_gib = 25.0
        basis = "smoke keeps E3 batch/n but saves no checkpoint (save_freq=-1)"
    else:
        checkpoint = _dir_bytes(E3_RUN / "checkpoints/global_step_200")
        predictions = _dir_bytes(E3_RUN / "predictions")
        tensorboard = _dir_bytes(E3_RUN / "tensorboard")
        if checkpoint <= 0:
            raise RuntimeError(f"cannot measure the E3 checkpoint footprint at {E3_RUN}")
        peak_gib = (CHECKPOINTS_KEPT * checkpoint + predictions + tensorboard) / 2**30
        basis = (
            f"{CHECKPOINTS_KEPT} x E3 checkpoint {checkpoint / 2**30:.2f} GiB "
            f"+ predictions {predictions / 2**30:.2f} + tensorboard {tensorboard / 2**30:.3f}; "
            "no tool turns makes E3-NoTool predictions no larger than E3's"
        )
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    minimum_gib = peak_gib + DISK_SAFETY_MARGIN_GIB
    if check and free_gib < minimum_gib:
        raise RuntimeError(f"E3-NoTool disk gate failed: {free_gib:.1f} GiB free < {minimum_gib:.1f} GiB")
    return {
        "remaining_peak_write_gib": round(peak_gib, 2),
        "disk_safety_margin_gib": DISK_SAFETY_MARGIN_GIB,
        "disk_minimum_gib": round(minimum_gib, 2),
        "disk_free_gib": round(free_gib, 2),
        "basis": basis,
    }


def resource_preflight_e3notool(
    config: DictConfig,
    smoke: bool,
    resume: bool = False,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """GPU, data-identity, pinned-source, config-diff and disk gates before any output."""
    import pyarrow.parquet as pq
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E3-NoTool requires exactly one visible CUDA GPU; none is currently available")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)

    if resume:
        if run_dir is None:
            raise ValueError("resume preflight requires the run directory")
        disk = resume_disk_estimate(run_dir, int(config.trainer.total_training_steps))
    else:
        disk = disk_estimate(smoke)

    train_rows = pq.read_metadata(config.data.train_files).num_rows
    val_rows = pq.read_metadata(config.data.val_files).num_rows
    if (train_rows, val_rows) != (5203, 100):
        raise ValueError(f"E3-NoTool must use the full E3 data even for smoke, got {train_rows}/{val_rows}")
    manifest = json.loads((PROJECT_ROOT / "rl/data/manifest.json").read_text(encoding="utf-8"))
    data_hashes: dict[str, str] = {}
    for raw_path in (config.data.train_files, config.data.val_files):
        path = Path(raw_path)
        relative = str(path.relative_to(PROJECT_ROOT))
        digest = _sha256(path)
        if digest != manifest["files"][relative]["sha256"]:
            raise ValueError(f"input hash mismatch for {relative}")
        data_hashes[relative] = digest
    checkpoint_model = Path(config.actor_rollout_ref.model.path) / "model.safetensors"
    if not checkpoint_model.is_file():
        raise FileNotFoundError(f"missing SFT model weights: {checkpoint_model}")

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "arm": "e3_notool",
        "gpu": properties.name,
        "gpu_total_gib": round(total_bytes / 2**30, 2),
        "gpu_free_gib": round(free_bytes / 2**30, 2),
        "disk_free_gib": disk["disk_free_gib"],
        "remaining_peak_write_gib": disk["remaining_peak_write_gib"],
        "disk_safety_margin_gib": DISK_SAFETY_MARGIN_GIB,
        "disk_minimum_gib": disk["disk_minimum_gib"],
        "e3notool_disk_estimate": disk,
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "input_hashes": data_hashes,
        "checkpoint": config.actor_rollout_ref.model.path,
        "checkpoint_model_sha256": _sha256(checkpoint_model),
        "verl_source_hashes": verl_source_hashes(),
        "config_diff_gate": config_diff_gate(str(config.trainer.experiment_name)),
        "tool_schema_in_prompt": False,
        "agent_loop": AGENT_LOOP_NAME,
        "git_head": git_head,
        "git_status_short": git_status,
        "torch_version": torch.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = RUN_ROOT / args.run_name
    config = compose_e3notool_config(args.run_name, smoke=args.smoke)
    validate_e3notool_config(config, run_dir, resume=args.resume)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        print(json.dumps({"config_diff_gate": config_diff_gate(args.run_name)}, indent=2))
        return
    if args.resume:
        config.trainer.val_before_train = False
    preflight = resource_preflight_e3notool(config, smoke=args.smoke, resume=args.resume, run_dir=run_dir)
    launch(config, run_dir, preflight, resume=args.resume)


if __name__ == "__main__":
    main()
