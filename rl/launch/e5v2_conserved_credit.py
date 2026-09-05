"""Resolve, validate, and launch the approved M8 / E5-v2 conserved-credit run.

Everything that is not the v2 treatment is imported from the frozen E5 v1 launcher
(`rl/launch/e5_turn_credit.py`, hashed by the M6/M7 freeze closures and left untouched):
config composition, M4/E3 control validation, the M6 freeze-bundle and resource preflight,
pinned veRL source hashes.  The Ray driver boundary below mirrors v1's ``TurnCreditTaskRunner``
hook-for-hook with the v2 wrapper variant, proof prefix and payload.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf

from rl.custom.turn_advantage import AUDIT_KEY, WRAPPER_PROOF_KEY
from rl.custom.turn_advantage_v2 import (
    ADOPTION_VERSION_V2,
    FROZEN_BETA,
    LEDGER_SCHEMA_VERSION_V2,
    VARIANT_CONSERVED_V2,
    make_conserved_compute_advantage_wrapper,
)
from rl.launch.e3_grpo_baseline import (
    DEFAULT_CONFIG as E3_CONFIG,
    PROJECT_ROOT,
    RUN_ROOT,
    _write_status,
    archive_interrupted_attempt,
    compose_config,
    validate_m4_config,
)
from rl.launch.e5_turn_credit import (
    _sha256,
    _verl_source_hashes,
    compose_e5_config,
    resource_preflight_e5,
    validate_freeze_bundle,
)

DEFAULT_CONFIG = PROJECT_ROOT / "rl/configs/e5v2_conserved_credit.yaml"
AGENT_LOOP_NAME = "toolcredit_turn_credit_v2_agent"
BOUNDARY_VERSION = "e5v2_runtime_wrapper_v1"
PROOF_PREFIX = "e5v2-wrapper-v1:"
LOG_MARKER = "TOOLCREDIT_E5V2_WRAPPER_ACTIVE "
E5_V1_RUN = PROJECT_ROOT / "rl/runs/e5_turn_credit_20260823_082012"
# Plan §5: the only structural differences E5-v1 -> E5-v2 may show.
METADATA_DIFF_PATHS = {
    "actor_rollout_ref.rollout.agent.default_agent_loop",
    "actor_rollout_ref.rollout.agent.agent_loop_config_path",
    "actor_rollout_ref.rollout.trace.project_name",
    "trainer.project_name",
}
E3_CONTROL_DIFF_PATHS = {"turn_credit"} | METADATA_DIFF_PATHS
SMOKE_DIFF_PATHS = {"trainer.total_training_steps", "trainer.save_freq", "trainer.test_freq"}
DISK_SAFETY_MARGIN_GIB = 30
CHECKPOINTS_KEPT = 3


def compose_e5v2_config(run_name: str, smoke: bool = False) -> DictConfig:
    """Compose E5-v2 with full E3 data/rollout controls even for smoke (plan §5)."""
    config = compose_config(DEFAULT_CONFIG, run_name, smoke=False)
    config.turn_credit.diagnostic_freeze_manifest = str(
        PROJECT_ROOT / str(config.turn_credit.diagnostic_freeze_manifest)
    )
    if smoke:
        config.trainer.total_training_steps = 5
        # Plan §8 step 3: the smoke must not retain a large checkpoint; veRL only
        # saves when save_freq > 0.
        config.trainer.save_freq = -1
        config.trainer.test_freq = 5
    return config


def diff_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in left.keys() | right.keys():
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(diff_paths(left[key], right[key], child))
        return paths
    return set() if left == right else {prefix}


def resolved(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config must be a mapping")
    return value


def config_diff_gate(run_name: str) -> dict[str, list[str]]:
    """Fail closed unless E5-v2 differs from E5-v1 and E3 only where plan §5 allows."""
    v2 = resolved(compose_e5v2_config(run_name))
    v1 = resolved(compose_e5_config(run_name))
    e3 = resolved(compose_config(E3_CONFIG, run_name))
    v1_to_v2 = diff_paths(v1, v2)
    illegal = {
        path for path in v1_to_v2 if not path.startswith("turn_credit.") and path not in METADATA_DIFF_PATHS
    }
    if illegal:
        raise ValueError(f"E5-v1 -> E5-v2 config diff gate failed: {sorted(illegal)}")
    e3_to_v2 = diff_paths(e3, v2)
    if e3_to_v2 != E3_CONTROL_DIFF_PATHS:
        raise ValueError(f"E3 -> E5-v2 config diff gate failed: {sorted(e3_to_v2)}")
    return {"v1_to_v2": sorted(v1_to_v2), "e3_to_v2": sorted(e3_to_v2)}


def validate_e5v2_config(config: DictConfig, run_dir: Path, resume: bool = False) -> None:
    """Fail closed if the frozen v2 treatment or any E3 control drifts."""
    validate_m4_config(config, run_dir, resume=resume)
    turn_credit = config.turn_credit
    if turn_credit.enabled is not True:
        raise ValueError("E5-v2 turn credit must be enabled")
    if str(turn_credit.variant) != VARIANT_CONSERVED_V2:
        raise ValueError(f"E5-v2 variant is frozen at {VARIANT_CONSERVED_V2}")
    if float(turn_credit.beta) != FROZEN_BETA:
        raise ValueError(f"E5-v2 beta is frozen at {FROZEN_BETA}")
    if str(turn_credit.adoption_version) != ADOPTION_VERSION_V2:
        raise ValueError("E5-v2 adoption version drifted")
    if str(turn_credit.ledger_schema_version) != LEDGER_SCHEMA_VERSION_V2:
        raise ValueError("E5-v2 ledger schema version drifted")
    if turn_credit.gate_zero_variance is not True or turn_credit.conserve_within_trajectory is not True:
        raise ValueError("E5-v2 gating/conservation flags are frozen at true")
    if "heuristic_version" in turn_credit:
        raise ValueError("E5-v2 must not carry the v1 heuristic_version key")
    if config.actor_rollout_ref.rollout.n != 8:
        raise ValueError("E5-v2 rollout n must remain 8, including smoke")
    if config.data.train_batch_size != 64 or config.data.max_response_length != 3072:
        raise ValueError("E5-v2 batch/response controls must match E3, including smoke")
    if config.actor_rollout_ref.rollout.agent.default_agent_loop != AGENT_LOOP_NAME:
        raise ValueError("E5-v2 must use the independent v2 turn-credit agent loop")
    if config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length != 512:
        raise ValueError("E5-v2 must preserve veRL's 512-character tool response limit")
    if config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side != "middle":
        raise ValueError("E5-v2 must preserve middle tool-response truncation")
    config_diff_gate(str(config.trainer.experiment_name))


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def disk_estimate(smoke: bool, check: bool = True) -> dict[str, Any]:
    """Recompute M6's ``remaining_peak_write + 30 GiB`` gate from measured E5 v1 artifacts."""
    if smoke:
        peak_gib = 25.0
        basis = "v1 smoke gate retained; smoke saves no checkpoint (save_freq=-1)"
    else:
        checkpoint = _dir_bytes(E5_V1_RUN / "checkpoints/global_step_200")
        predictions = _dir_bytes(E5_V1_RUN / "predictions")
        metrics = (E5_V1_RUN / "metrics.json").stat().st_size
        tensorboard = _dir_bytes(E5_V1_RUN / "tensorboard")
        recovery = _dir_bytes(E5_V1_RUN / "recovery")
        measured = (CHECKPOINTS_KEPT * checkpoint + predictions + metrics + tensorboard + recovery) / 2**30
        peak_gib = max(103.0, measured)
        basis = (
            f"max(M6 gate 103, {CHECKPOINTS_KEPT} x checkpoint {checkpoint / 2**30:.1f} GiB "
            f"+ predictions {predictions / 2**30:.1f} + metrics {metrics / 2**30:.1f} "
            f"+ tensorboard {tensorboard / 2**30:.2f} + recovery {recovery / 2**30:.2f} = {measured:.1f} GiB)"
        )
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    minimum_gib = peak_gib + DISK_SAFETY_MARGIN_GIB
    if check and free_gib < minimum_gib:
        raise RuntimeError(f"E5-v2 disk gate failed: {free_gib:.1f} GiB free < {minimum_gib:.1f} GiB")
    return {
        "remaining_peak_write_gib": round(peak_gib, 2),
        "disk_safety_margin_gib": DISK_SAFETY_MARGIN_GIB,
        "disk_minimum_gib": round(minimum_gib, 2),
        "disk_free_gib": round(free_gib, 2),
        "basis": basis,
    }


def resume_disk_estimate(run_dir: Path, total_steps: int = 200) -> dict[str, Any]:
    """Plan §8 step 2 formula for a resume: remaining peak write measured from the run itself.

    With ``max_actor_ckpt_to_keep`` already saturated, later checkpoints rotate (net zero) and
    the transient peak is one extra checkpoint; predictions grow per remaining step; the
    launcher's recovery archive copies every train file newer than the tracker checkpoint.
    """
    tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    checkpoint_step = int(tracker.read_text(encoding="utf-8").strip())
    checkpoint_dir = run_dir / "checkpoints" / f"global_step_{checkpoint_step}"
    checkpoint_bytes = _dir_bytes(checkpoint_dir)
    if checkpoint_bytes <= 0:
        raise RuntimeError(f"tracker checkpoint is empty: {checkpoint_dir}")
    train_dir = run_dir / "predictions/train"
    train_files = sorted(train_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    if not train_files:
        raise RuntimeError("resume without any persisted train step")
    per_step = sum(path.stat().st_size for path in train_files) / len(train_files)
    remaining_steps = total_steps - checkpoint_step
    to_archive = sum(path.stat().st_size for path in train_files if int(path.stem) > checkpoint_step)
    validation_dir = run_dir / "predictions/validation"
    validation_files = list(validation_dir.glob("*.jsonl"))
    per_validation = (
        sum(path.stat().st_size for path in validation_files) / len(validation_files) if validation_files else 0
    )
    remaining_validations = len([step for step in range(checkpoint_step + 25, total_steps + 1, 25)])
    peak = (
        checkpoint_bytes
        + remaining_steps * per_step
        + remaining_validations * per_validation
        + to_archive
        + _dir_bytes(run_dir / "recovery")
    )
    peak_gib = peak / 2**30
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    minimum_gib = peak_gib + DISK_SAFETY_MARGIN_GIB
    if free_gib < minimum_gib:
        raise RuntimeError(f"E5-v2 resume disk gate failed: {free_gib:.1f} GiB free < {minimum_gib:.1f} GiB")
    return {
        "mode": "resume",
        "remaining_peak_write_bytes": int(peak),
        "checkpoint_step": checkpoint_step,
        "remaining_steps": remaining_steps,
        "transient_checkpoint_gib": round(checkpoint_bytes / 2**30, 2),
        "remaining_predictions_gib": round(remaining_steps * per_step / 2**30, 3),
        "remaining_validation_gib": round(remaining_validations * per_validation / 2**30, 3),
        "recovery_archive_gib": round((to_archive + _dir_bytes(run_dir / "recovery")) / 2**30, 3),
        "remaining_peak_write_gib": round(peak_gib, 2),
        "disk_safety_margin_gib": DISK_SAFETY_MARGIN_GIB,
        "disk_minimum_gib": round(minimum_gib, 2),
        "disk_free_gib": round(free_gib, 2),
        "basis": "1 rotating checkpoint (transient) + remaining train/validation JSONL + recovery archive; keep=3 already saturated",
    }


def _resource_preflight_resume(config: DictConfig, run_dir: Path) -> dict[str, Any]:
    """Every v1 gate except the fresh-run disk figure, which the resume estimate replaces."""
    import pyarrow.parquet as pq
    import subprocess

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E5-v2 requires exactly one visible CUDA GPU; none is currently available")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)
    disk = resume_disk_estimate(run_dir, int(config.trainer.total_training_steps))
    train_rows = pq.read_metadata(config.data.train_files).num_rows
    val_rows = pq.read_metadata(config.data.val_files).num_rows
    if (train_rows, val_rows) != (5203, 100):
        raise ValueError(f"E5-v2 must use full E3 data, got {train_rows}/{val_rows}")
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
    freeze = validate_freeze_bundle(Path(config.turn_credit.diagnostic_freeze_manifest))
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "gpu": properties.name,
        "gpu_total_gib": round(total_bytes / 2**30, 2),
        "gpu_free_gib": round(free_bytes / 2**30, 2),
        "disk_free_gib": disk["disk_free_gib"],
        "remaining_peak_write_gib": disk["remaining_peak_write_gib"],
        "disk_safety_margin_gib": DISK_SAFETY_MARGIN_GIB,
        "disk_minimum_gib": disk["disk_minimum_gib"],
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "input_hashes": data_hashes,
        "checkpoint": config.actor_rollout_ref.model.path,
        "checkpoint_model_sha256": _sha256(checkpoint_model),
        "freeze": freeze,
        "verl_source_hashes": _verl_source_hashes(),
        "git_head": git_head,
        "git_status_short": git_status,
        "torch_version": torch.__version__,
        "e5v2_disk_estimate": disk,
    }


def resource_preflight_e5v2(config: DictConfig, smoke: bool, resume: bool = False, run_dir: Path | None = None) -> dict[str, Any]:
    if resume:
        if run_dir is None:
            raise ValueError("resume preflight requires the run directory")
        preflight = _resource_preflight_resume(config, run_dir)
    else:
        preflight = resource_preflight_e5(config, smoke=smoke)
        preflight["e5v2_disk_estimate"] = disk_estimate(smoke)
    preflight["config_diff_gate"] = config_diff_gate(str(config.trainer.experiment_name))
    preflight["variant"] = VARIANT_CONSERVED_V2
    return preflight


def _proof_path(config: DictConfig) -> Path:
    return Path(config.trainer.default_local_dir).parent / "wrapper_installation.json"


class ConservedTurnCreditTaskRunner:
    """E5-v2 Ray driver boundary; mirrors the frozen v1 ``TurnCreditTaskRunner.run`` hook-for-hook."""

    def run(self, config: DictConfig) -> None:
        from verl.trainer.main_ppo import TaskRunner
        from verl.trainer.ppo import ray_trainer

        if not isinstance(self, TaskRunner):
            raise TypeError("ConservedTurnCreditTaskRunner must be combined with veRL TaskRunner")
        source_hashes = _verl_source_hashes()
        base_compute = ray_trainer.compute_advantage
        base_log = ray_trainer.RayPPOTrainer._log_rollout_data
        if hasattr(base_compute, "__toolcredit_wrapper_proof__"):
            raise RuntimeError("refusing to stack turn-credit advantage wrappers")
        proof_payload = {
            "boundary_version": BOUNDARY_VERSION,
            "variant": str(config.turn_credit.variant),
            "beta": float(config.turn_credit.beta),
            "heuristic_version": str(config.turn_credit.adoption_version),
            "ledger_schema_version": str(config.turn_credit.ledger_schema_version),
            "gate_zero_variance": bool(config.turn_credit.gate_zero_variance),
            "conserve_within_trajectory": bool(config.turn_credit.conserve_within_trajectory),
            "base_compute_module": base_compute.__module__,
            "base_compute_name": base_compute.__name__,
            "base_compute_source_file": inspect.getsourcefile(base_compute),
            "verl_source_hashes": source_hashes,
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        proof_id = PROOF_PREFIX + hashlib.sha256(
            json.dumps(proof_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        proof_payload["proof_id"] = proof_id
        wrapped_compute = make_conserved_compute_advantage_wrapper(
            base_compute,
            beta=float(config.turn_credit.beta),
            enabled=bool(config.turn_credit.enabled),
            proof_id=proof_id,
        )

        @wraps(base_log)
        def wrapped_log(
            trainer: Any,
            batch: Any,
            reward_extra_infos_dict: dict[str, Any],
            timing_raw: dict[str, Any],
            rollout_data_dir: str,
        ) -> Any:
            if AUDIT_KEY not in batch.non_tensor_batch or WRAPPER_PROOF_KEY not in batch.non_tensor_batch:
                raise RuntimeError("turn-credit compute wrapper was not active before rollout logging")
            proofs = [str(value) for value in batch.non_tensor_batch[WRAPPER_PROOF_KEY]]
            if not proofs or any(value != proof_id for value in proofs):
                raise RuntimeError("turn-credit wrapper proof mismatch")
            augmented = dict(reward_extra_infos_dict)
            augmented[AUDIT_KEY] = list(batch.non_tensor_batch[AUDIT_KEY])
            augmented[WRAPPER_PROOF_KEY] = proofs
            return base_log(trainer, batch, augmented, timing_raw, rollout_data_dir)

        ray_trainer.compute_advantage = wrapped_compute
        ray_trainer.RayPPOTrainer._log_rollout_data = wrapped_log
        proof_file = _proof_path(config)
        proof_file.parent.mkdir(parents=True, exist_ok=True)
        proof_payload["installed"] = True
        proof_payload["restored"] = False
        proof_file.write_text(json.dumps(proof_payload, indent=2) + "\n", encoding="utf-8")
        print(LOG_MARKER + json.dumps(proof_payload, sort_keys=True))
        try:
            TaskRunner.run(self, config)
        finally:
            ray_trainer.compute_advantage = base_compute
            ray_trainer.RayPPOTrainer._log_rollout_data = base_log
            proof_payload["restored"] = True
            proof_payload["restored_at"] = datetime.now(timezone.utc).isoformat()
            proof_file.write_text(json.dumps(proof_payload, indent=2) + "\n", encoding="utf-8")


def _remote_runner_class() -> Any:
    from verl.trainer.main_ppo import TaskRunner

    concrete = type("E5V2ConservedTaskRunner", (ConservedTurnCreditTaskRunner, TaskRunner), {})
    return ray.remote(num_cpus=1)(concrete)


def launch(config: DictConfig, run_dir: Path, preflight: dict[str, Any], resume: bool = False) -> None:
    """Same five-artifact discipline and resume archival as the frozen v1 launcher."""
    run_dir.mkdir(parents=True, exist_ok=resume)
    resolved_yaml = OmegaConf.to_yaml(config, resolve=True)
    resume_metadata: dict[str, Any] = {}
    if not resume:
        (run_dir / "resolved_config.yaml").write_text(resolved_yaml, encoding="utf-8")
        preflight_path = run_dir / "preflight.json"
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        resume_metadata = archive_interrupted_attempt(run_dir, stamp)
        archive_dir = Path(resume_metadata["archive_dir"])
        (archive_dir / "runtime_config.yaml").write_text(resolved_yaml, encoding="utf-8")
        preflight_path = run_dir / f"resume_preflight_{stamp}.json"
    preflight_path.write_text(json.dumps({**preflight, **resume_metadata}, indent=2) + "\n", encoding="utf-8")
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
        from verl.trainer.main_ppo import run_ppo

        run_ppo(config, task_runner_class=_remote_runner_class())
    except BaseException as exc:
        _write_status(run_dir, "failed", error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        _write_status(run_dir, "completed")


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
    config = compose_e5v2_config(args.run_name, smoke=args.smoke)
    validate_e5v2_config(config, run_dir, resume=args.resume)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        print(json.dumps({"config_diff_gate": config_diff_gate(args.run_name)}, indent=2))
        return
    if args.resume:
        config.trainer.val_before_train = False
    preflight = resource_preflight_e5v2(config, smoke=args.smoke, resume=args.resume, run_dir=run_dir)
    launch(config, run_dir, preflight, resume=args.resume)


if __name__ == "__main__":
    main()
