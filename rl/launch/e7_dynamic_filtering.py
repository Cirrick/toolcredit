"""Resolve, validate, and launch the approved M10 / E7 dynamic-filtering run.

E7 is E3 with exactly one algorithmic change — zero-variance UID groups are dropped and the
actor batch is refilled from the *same* actor snapshot (``rl/custom/dynamic_filter_trainer.py``)
— plus three pure-infrastructure FSDP offload flags (plan §3.2).  Everything else is reused from
the frozen E3 launcher: config composition, the resume guard, the five-artifact launch
discipline and the recovery archive.

Driver boundary (STEP0 §4 option (b), plan §12): inside the Ray driver actor the module-global
name ``verl.trainer.main_ppo.RayPPOTrainer`` is temporarily bound to
``DynamicFilterRayPPOTrainer`` for the duration of ``TaskRunner.run`` and restored in
``finally``.  ``main_ppo.py:299`` resolves that name at call time, so zero upstream lines are
copied.  A proof file and the ``TOOLCREDIT_E7_BOUNDARY_ACTIVE`` / ``TOOLCREDIT_E7_TRAINER_ACTIVE``
log markers make the binding auditable.

Segments (plan §3.5): ``--stop-at-step N`` ends the run at effective step N with a normal
checkpoint + validation; the next segment resumes with ``--resume --stop-at-step M``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf

from rl.custom.dynamic_filter_trainer import (
    CANDIDATES_DIRNAME,
    E3_TOTAL_TRAJECTORIES,
    FILTER_METRIC,
    GROUP_SIZE,
    LEDGER_FILENAME,
    LEDGER_JSONL,
    MAX_GENERATION_BATCHES,
    STOP_REQUEST_FILENAME,
    TARGET_GROUPS,
    TRAINER_ACTIVATIONS_JSONL,
    TRAINER_CLASS_FQN,
    TRAINER_PROOF_FILENAME,
    DynamicFilterRayPPOTrainer,
)
from rl.launch.e3_grpo_baseline import (
    DEFAULT_CONFIG as E3_CONFIG,
    PROJECT_ROOT,
    RUN_ROOT,
    _sha256,
    _write_status,
    archive_interrupted_attempt,
    compose_config,
    validate_m4_config,
)
from rl.launch.e5_turn_credit import PINNED_VERL_HASHES
from rl.launch.e5v2_conserved_credit import diff_paths, resolved, resume_disk_estimate

DEFAULT_CONFIG = PROJECT_ROOT / "rl/configs/e7_dynamic_filtering.yaml"
E3_RUN = PROJECT_ROOT / "rl/runs/e3_grpo_baseline_20260819_224555"
DISK_SAFETY_MARGIN_GIB = 30
CHECKPOINTS_KEPT = 3
EQUAL_COMPUTE_CHECKPOINTS = 1
# Plan §2: expected generation batches per effective step with the late-E3 informative rate.
EXPECTED_BATCHES_PER_STEP = 2.7
BOUNDARY_VERSION = "e7_driver_binding_v1"
BOUNDARY_PROOF_FILENAME = "e7_boundary_installation.json"
BOUNDARY_ACTIVATIONS_JSONL = "e7_boundary_activations.jsonl"
LOG_MARKER = "TOOLCREDIT_E7_BOUNDARY_ACTIVE "

FILTER_PATH = "algorithm.filter_groups"
TRAINER_MARKER_PATH = "trainer.toolcredit_trainer_class"
OFFLOAD_PATHS = {
    "actor_rollout_ref.actor.fsdp_config.param_offload",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.ref.fsdp_config.param_offload",
}
# Plan §3.2: the complete set of resolved-config paths E3 -> E7 may differ on.  ``diff_paths``
# reports the whole ``algorithm.filter_groups`` node because E3 has no such key; the three
# sub-keys are asserted separately in ``validate_e7_config``.
E3_DIFF_PATHS = {
    FILTER_PATH,
    TRAINER_MARKER_PATH,
    "trainer.project_name",
    "actor_rollout_ref.rollout.trace.project_name",
} | OFFLOAD_PATHS
SMOKE_DIFF_PATHS = {"trainer.total_training_steps", "trainer.save_freq", "trainer.test_freq"}
MODES = ("formal", "smoke", "resume_check")
SMOKE_STEPS = 5
RESUME_CHECK_STEPS = 2
RESUME_CHECK_STOP_STEP = 1
SEGMENT_STOP_STEPS = (50, 100, 150, 200)

# The rotation-by-rename trick for the equal-compute checkpoint and the awake-between-chunks
# invariant depend on these files; pin them alongside the E5 set.
M10_PINNED_VERL_HASHES = {
    **PINNED_VERL_HASHES,
    "verl/checkpoint_engine/base.py": "3640690f0a01464620ddb6c3926d15d4476cbbfc8191a2da0f0904aaed2ae382",
    "verl/utils/checkpoint/checkpoint_manager.py": (
        "97ad2cf3d38ca6ba16449ad27772707a82cdfbee8fea5eeaae2c36b1cebcc38a"
    ),
    "verl/utils/checkpoint/fsdp_checkpoint_manager.py": (
        "87036a65beac76806817362da8c58ae14131fa8dcb082f95568eaaa47eb37d10"
    ),
}


def verl_source_hashes() -> dict[str, str]:
    import verl

    root = Path(verl.__file__).resolve().parent.parent
    actual: dict[str, str] = {}
    for relative, expected in M10_PINNED_VERL_HASHES.items():
        digest = _sha256(root / relative)
        if digest != expected:
            raise RuntimeError(f"pinned veRL source drift for {relative}: {digest} != {expected}")
        actual[relative] = digest
    return actual


def compose_e7_config(run_name: str, mode: str = "formal") -> DictConfig:
    """Compose E7 with the full E3 data/rollout controls in every mode.

    ``smoke``: 5 effective steps, no checkpoint (``save_freq=-1``), validation at 0/5.
    ``resume_check``: 2 effective steps with ``save_freq=2`` so ``--stop-at-step 1`` leaves a
    resumable checkpoint and the second segment finishes at step 2 (plan §8 step 3).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    config = compose_config(DEFAULT_CONFIG, run_name, smoke=False)
    if mode == "smoke":
        config.trainer.total_training_steps = SMOKE_STEPS
        config.trainer.save_freq = -1
        config.trainer.test_freq = SMOKE_STEPS
    elif mode == "resume_check":
        config.trainer.total_training_steps = RESUME_CHECK_STEPS
        config.trainer.save_freq = RESUME_CHECK_STEPS
        config.trainer.test_freq = RESUME_CHECK_STEPS
    return config


def config_diff_gate(run_name: str) -> dict[str, list[str]]:
    """Fail closed unless E3 -> E7 differs exactly on the preregistered paths (plan §3.2)."""
    e7 = resolved(compose_e7_config(run_name))
    e3 = resolved(compose_config(E3_CONFIG, run_name))
    observed = diff_paths(e3, e7)
    if observed != E3_DIFF_PATHS:
        raise ValueError(
            "E3 -> E7 config diff gate failed: "
            f"unexpected={sorted(observed - E3_DIFF_PATHS)} missing={sorted(E3_DIFF_PATHS - observed)}"
        )
    filter_groups = e7["algorithm"]["filter_groups"]
    if set(filter_groups) != {"enable", "metric", "max_num_gen_batches"}:
        raise ValueError(f"filter_groups must carry exactly three keys, got {sorted(filter_groups)}")
    return {"e3_to_e7": sorted(observed), "filter_groups": dict(filter_groups)}


def validate_e7_config(config: DictConfig, run_dir: Path, resume: bool = False) -> None:
    """Fail closed if the treatment, the offload flags or any E3 control drifted."""
    validate_m4_config(config, run_dir, resume=resume)
    filter_groups = config.algorithm.get("filter_groups", None)
    if filter_groups is None:
        raise ValueError("E7 requires algorithm.filter_groups")
    if filter_groups.enable is not True:
        raise ValueError("E7 must enable filter_groups")
    if str(filter_groups.metric) != FILTER_METRIC:
        raise ValueError(f"E7 must filter on {FILTER_METRIC!r}, not {filter_groups.metric!r}")
    if int(filter_groups.max_num_gen_batches) != MAX_GENERATION_BATCHES:
        raise ValueError("E7 must allow exactly four generation batches per effective step")
    if config.trainer.get("toolcredit_trainer_class", None) != TRAINER_CLASS_FQN:
        raise ValueError("E7 trainer marker must name the dynamic-filter trainer")
    actor_fsdp = config.actor_rollout_ref.actor.fsdp_config
    ref_fsdp = config.actor_rollout_ref.ref.fsdp_config
    if actor_fsdp.param_offload is not False or actor_fsdp.optimizer_offload is not False:
        raise ValueError("E7 actor FSDP CPU offload must be off (plan v2 §0 item 1)")
    if ref_fsdp.param_offload is not False:
        raise ValueError("E7 ref FSDP CPU offload must be off (plan v2 §0 item 1)")
    if int(config.actor_rollout_ref.rollout.n) != GROUP_SIZE:
        raise ValueError("E7 rollout n must remain 8, including smoke")
    if int(config.data.train_batch_size) != TARGET_GROUPS or int(config.data.max_response_length) != 3072:
        raise ValueError("E7 batch/response controls must match E3, including smoke")
    if config.algorithm.norm_adv_by_std_in_grpo is not True:
        raise ValueError("E7 keeps GRPO std normalisation (Dr. GRPO is out of scope)")
    if int(config.trainer.get("max_actor_ckpt_to_keep", 0)) != CHECKPOINTS_KEPT:
        raise ValueError("E7 must keep max_actor_ckpt_to_keep=3")
    config_diff_gate(str(config.trainer.experiment_name))


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def disk_estimate(mode: str, check: bool = True) -> dict[str, Any]:
    """Plan §8 step 2: recompute ``remaining_peak_write + 30 GiB`` from measured E3 artifacts."""
    checkpoint = _dir_bytes(E3_RUN / "checkpoints/global_step_200")
    if checkpoint <= 0:
        raise RuntimeError(f"cannot measure the E3 checkpoint footprint at {E3_RUN}")
    predictions = _dir_bytes(E3_RUN / "predictions")
    tensorboard = _dir_bytes(E3_RUN / "tensorboard")
    if mode == "smoke":
        peak = predictions * SMOKE_STEPS / 200 * EXPECTED_BATCHES_PER_STEP + tensorboard
        basis = "smoke saves no checkpoint (save_freq=-1); 5 steps of predictions + candidates"
    elif mode == "resume_check":
        peak = 2 * checkpoint + predictions * RESUME_CHECK_STEPS / 200 * EXPECTED_BATCHES_PER_STEP + tensorboard
        basis = "resume check writes two checkpoints (stop step 1, final step 2)"
    else:
        peak = (
            (CHECKPOINTS_KEPT + EQUAL_COMPUTE_CHECKPOINTS) * checkpoint
            + predictions * EXPECTED_BATCHES_PER_STEP
            + tensorboard
        )
        basis = (
            f"{CHECKPOINTS_KEPT} rotating + {EQUAL_COMPUTE_CHECKPOINTS} equal-compute x E3 checkpoint "
            f"{checkpoint / 2**30:.2f} GiB + E3 predictions {predictions / 2**30:.2f} GiB x {EXPECTED_BATCHES_PER_STEP} "
            f"(train JSONL holds the 64 selected groups; candidates JSONL is text-free) + tensorboard"
        )
    peak_gib = peak / 2**30
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    minimum_gib = peak_gib + DISK_SAFETY_MARGIN_GIB
    if check and free_gib < minimum_gib:
        raise RuntimeError(f"E7 disk gate failed: {free_gib:.1f} GiB free < {minimum_gib:.1f} GiB")
    return {
        "mode": mode,
        "remaining_peak_write_gib": round(peak_gib, 2),
        "disk_safety_margin_gib": DISK_SAFETY_MARGIN_GIB,
        "disk_minimum_gib": round(minimum_gib, 2),
        "disk_free_gib": round(free_gib, 2),
        "e3_checkpoint_gib": round(checkpoint / 2**30, 2),
        "basis": basis,
    }


def resource_preflight_e7(
    config: DictConfig, mode: str, resume: bool = False, run_dir: Path | None = None
) -> dict[str, Any]:
    """GPU, data-identity, pinned-source, config-diff, offload and disk gates before any output."""
    import pyarrow.parquet as pq
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E7 requires exactly one visible CUDA GPU; none is currently available")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)
    if resume:
        if run_dir is None:
            raise ValueError("resume preflight requires the run directory")
        disk = resume_disk_estimate(run_dir, int(config.trainer.total_training_steps))
    else:
        disk = disk_estimate(mode)

    train_rows = pq.read_metadata(config.data.train_files).num_rows
    val_rows = pq.read_metadata(config.data.val_files).num_rows
    if (train_rows, val_rows) != (5203, 100):
        raise ValueError(f"E7 must use the full E3 data even for smoke, got {train_rows}/{val_rows}")
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
        "arm": "e7_dynamic_filtering",
        "mode": mode,
        "gpu": properties.name,
        "gpu_total_gib": round(total_bytes / 2**30, 2),
        "gpu_free_gib": round(free_bytes / 2**30, 2),
        "disk_free_gib": disk["disk_free_gib"],
        "remaining_peak_write_gib": disk["remaining_peak_write_gib"],
        "disk_safety_margin_gib": DISK_SAFETY_MARGIN_GIB,
        "disk_minimum_gib": disk["disk_minimum_gib"],
        "e7_disk_estimate": disk,
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "input_hashes": data_hashes,
        "checkpoint": config.actor_rollout_ref.model.path,
        "checkpoint_model_sha256": _sha256(checkpoint_model),
        "verl_source_hashes": verl_source_hashes(),
        "config_diff_gate": config_diff_gate(str(config.trainer.experiment_name)),
        "filter_groups": {
            "enable": bool(config.algorithm.filter_groups.enable),
            "metric": str(config.algorithm.filter_groups.metric),
            "max_num_gen_batches": int(config.algorithm.filter_groups.max_num_gen_batches),
        },
        "prompts_per_chunk": int(config.data.train_batch_size),
        "rollout_n": int(config.actor_rollout_ref.rollout.n),
        "equal_compute_threshold": E3_TOTAL_TRAJECTORIES,
        "offload": {
            "actor_param_offload": bool(config.actor_rollout_ref.actor.fsdp_config.param_offload),
            "actor_optimizer_offload": bool(config.actor_rollout_ref.actor.fsdp_config.optimizer_offload),
            "ref_param_offload": bool(config.actor_rollout_ref.ref.fsdp_config.param_offload),
        },
        "git_head": git_head,
        "git_status_short": git_status,
        "torch_version": torch.__version__,
    }


# --------------------------------------------------------------------------- #
# Driver boundary (STEP0 §4 option (b))
# --------------------------------------------------------------------------- #


def _run_dir_of(config: DictConfig) -> Path:
    return Path(config.trainer.default_local_dir).parent


class DynamicFilteringTaskRunner:
    """Mixin attached to veRL's ``TaskRunner``; binds the trainer name for the driver lifetime."""

    E7_STOP_AT_STEP: int | None = None

    def run(self, config: DictConfig) -> None:
        from verl.trainer import main_ppo
        from verl.trainer.main_ppo import TaskRunner
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        if not isinstance(self, TaskRunner):
            raise TypeError("DynamicFilteringTaskRunner must be combined with veRL TaskRunner")
        base = main_ppo.RayPPOTrainer
        if base is not RayPPOTrainer:
            raise RuntimeError("refusing to stack trainer bindings: main_ppo.RayPPOTrainer is already replaced")
        source_hashes = verl_source_hashes()
        DynamicFilterRayPPOTrainer.E7_STOP_AT_STEP = self.E7_STOP_AT_STEP
        payload: dict[str, Any] = {
            "boundary_version": BOUNDARY_VERSION,
            "bound_name": "verl.trainer.main_ppo.RayPPOTrainer",
            "bound_to": TRAINER_CLASS_FQN,
            "base_class": f"{base.__module__}.{base.__qualname__}",
            "stop_at_step": self.E7_STOP_AT_STEP,
            "verl_source_hashes": source_hashes,
            "pid": os.getpid(),
            "installed": True,
            "restored": False,
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        run_dir = _run_dir_of(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        proof_file = run_dir / BOUNDARY_PROOF_FILENAME
        proof_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        main_ppo.RayPPOTrainer = DynamicFilterRayPPOTrainer
        print(LOG_MARKER + json.dumps(payload, sort_keys=True))
        try:
            TaskRunner.run(self, config)
        finally:
            main_ppo.RayPPOTrainer = base
            payload["restored"] = True
            payload["restored_at"] = datetime.now(timezone.utc).isoformat()
            proof_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with (run_dir / BOUNDARY_ACTIVATIONS_JSONL).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _remote_runner_class(stop_at_step: int | None) -> Any:
    from verl.trainer.main_ppo import TaskRunner

    concrete = type(
        "E7DynamicFilteringTaskRunner",
        (DynamicFilteringTaskRunner, TaskRunner),
        {"E7_STOP_AT_STEP": stop_at_step},
    )
    return ray.remote(num_cpus=1)(concrete)


# --------------------------------------------------------------------------- #
# Launch / segments
# --------------------------------------------------------------------------- #


def tracker_step(run_dir: Path) -> int | None:
    tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    return int(tracker.read_text(encoding="utf-8").strip()) if tracker.is_file() else None


def archive_e7_evidence(run_dir: Path, archive_dir: Path, checkpoint_step: int) -> dict[str, Any]:
    """Preserve E7 evidence newer than the checkpoint before the resumed run overwrites it."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in (
        LEDGER_JSONL,
        TRAINER_PROOF_FILENAME,
        TRAINER_ACTIVATIONS_JSONL,
        BOUNDARY_PROOF_FILENAME,
        BOUNDARY_ACTIVATIONS_JSONL,
        STOP_REQUEST_FILENAME,
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, archive_dir / name)
            copied.append(name)
    candidates_dir = run_dir / "predictions" / CANDIDATES_DIRNAME
    archived_candidates: list[int] = []
    if candidates_dir.is_dir():
        target = archive_dir / CANDIDATES_DIRNAME
        for path in sorted(candidates_dir.glob("*.jsonl")):
            stem = path.stem.split(".")[0]
            if stem.isdigit() and int(stem) > checkpoint_step:
                target.mkdir(exist_ok=True)
                shutil.copy2(path, target / path.name)
                archived_candidates.append(int(stem))
    stop_request = run_dir / STOP_REQUEST_FILENAME
    consumed_stop_request = False
    if stop_request.is_file():
        stop_request.rename(archive_dir / f"{STOP_REQUEST_FILENAME}.consumed")
        consumed_stop_request = True
    return {
        "e7_archived_files": copied,
        "e7_archived_candidate_steps": archived_candidates,
        "e7_consumed_stop_request": consumed_stop_request,
    }


def _final_status(run_dir: Path, config: DictConfig, stop_at_step: int | None) -> tuple[str, dict[str, Any]]:
    total = int(config.trainer.total_training_steps)
    reached = tracker_step(run_dir)
    if int(config.trainer.save_freq) <= 0:
        # Smoke: no checkpoint is ever written; completion is proved by the train JSONL count.
        steps = sorted(int(p.stem) for p in (run_dir / "predictions/train").glob("*.jsonl") if p.stem.isdigit())
        if steps != list(range(1, total + 1)):
            raise RuntimeError(f"smoke did not persist steps 1..{total}: {steps}")
        return "completed", {"final_step": total, "checkpoint": None}
    if reached is None:
        raise RuntimeError("run ended without a checkpoint tracker")
    if reached >= total:
        return "completed", {"final_step": reached}
    stop_request = run_dir / STOP_REQUEST_FILENAME
    if stop_at_step is not None and reached == stop_at_step:
        return "segment_completed", {"segment_stop_step": reached, "stopped_by_request": stop_request.is_file()}
    if stop_request.is_file():
        return "segment_completed", {"segment_stop_step": reached, "stopped_by_request": True}
    raise RuntimeError(f"run stopped at checkpoint {reached} without reaching {stop_at_step or total}")


def launch(
    config: DictConfig,
    run_dir: Path,
    preflight: dict[str, Any],
    resume: bool = False,
    stop_at_step: int | None = None,
) -> None:
    """E3's five-artifact discipline plus E7 evidence archival and segment-aware status."""
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
        resume_metadata.update(archive_e7_evidence(run_dir, archive_dir, int(resume_metadata["checkpoint_step"])))
        preflight_path = run_dir / f"resume_preflight_{stamp}.json"
    preflight_path.write_text(
        json.dumps({**preflight, **resume_metadata, "stop_at_step": stop_at_step}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_status(
        run_dir,
        "running",
        pid=os.getpid(),
        resumed=resume,
        resume_from_step=resume_metadata.get("checkpoint_step"),
        recovery_archive=resume_metadata.get("archive_dir"),
        stop_at_step=stop_at_step,
    )
    os.environ["TENSORBOARD_DIR"] = str(run_dir / "tensorboard")
    try:
        from verl.trainer.main_ppo import run_ppo

        run_ppo(config, task_runner_class=_remote_runner_class(stop_at_step))
        state, extra = _final_status(run_dir, config, stop_at_step)
    except BaseException as exc:
        _write_status(run_dir, "failed", error_type=type(exc).__name__, error=str(exc), stop_at_step=stop_at_step)
        raise
    else:
        _write_status(run_dir, state, stop_at_step=stop_at_step, **extra)


def validate_segment_request(run_dir: Path, config: DictConfig, resume: bool, stop_at_step: int | None) -> None:
    total = int(config.trainer.total_training_steps)
    if stop_at_step is not None:
        if stop_at_step < 1 or stop_at_step > total:
            raise ValueError(f"--stop-at-step must be within 1..{total}")
        if int(config.trainer.save_freq) <= 0:
            raise ValueError("--stop-at-step needs save_freq > 0 (not available in --smoke)")
    if resume:
        reached = tracker_step(run_dir)
        if reached is None:
            raise FileNotFoundError("resume requires a checkpoint tracker")
        checkpoint_ledger = run_dir / "checkpoints" / f"global_step_{reached}" / LEDGER_FILENAME
        if not checkpoint_ledger.is_file():
            raise FileNotFoundError(f"resume checkpoint lacks the E7 ledger: {checkpoint_ledger}")
        if stop_at_step is not None and stop_at_step <= reached:
            raise ValueError(f"--stop-at-step {stop_at_step} is not beyond the checkpoint step {reached}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="5 effective steps, no checkpoint")
    mode.add_argument("--resume-check", action="store_true", help="2 effective steps with save_freq=2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at-step", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "smoke" if args.smoke else "resume_check" if args.resume_check else "formal"
    run_dir = RUN_ROOT / args.run_name
    config = compose_e7_config(args.run_name, mode=mode)
    validate_e7_config(config, run_dir, resume=args.resume)
    validate_segment_request(run_dir, config, args.resume, args.stop_at_step)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        print(json.dumps({"config_diff_gate": config_diff_gate(args.run_name), "mode": mode}, indent=2))
        return
    if args.resume:
        # The checkpoint step already has a persisted validation JSONL (plan §3.5).
        config.trainer.val_before_train = False
    preflight = resource_preflight_e7(config, mode, resume=args.resume, run_dir=run_dir)
    launch(config, run_dir, preflight, resume=args.resume, stop_at_step=args.stop_at_step)


if __name__ == "__main__":
    main()
