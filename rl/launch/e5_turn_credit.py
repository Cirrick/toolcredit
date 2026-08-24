"""Resolve, validate, and launch the approved M6 / E5 Tier-A run."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf

from rl.custom.turn_advantage import (
    ADOPTION_VERSION,
    AUDIT_KEY,
    LEDGER_SCHEMA_VERSION,
    WRAPPER_PROOF_KEY,
    make_compute_advantage_wrapper,
)
from rl.launch.e3_grpo_baseline import (
    PROJECT_ROOT,
    RUN_ROOT,
    _sha256,
    _write_status,
    archive_interrupted_attempt,
    compose_config,
    validate_m4_config,
)

DEFAULT_CONFIG = PROJECT_ROOT / "rl/configs/e5_turn_credit.yaml"
FREEZE_FILES = (
    PROJECT_ROOT / "plans/M7_DIAGNOSTIC_FREEZE_V2.md",
    PROJECT_ROOT / "eval/diagnostic_protocol_v2.json",
    PROJECT_ROOT / "eval/diagnostic_panel_v1.jsonl",
    PROJECT_ROOT / "eval/diagnostic_taxonomy_v1.json",
)
PINNED_VERL_HASHES = {
    "verl/trainer/ppo/ray_trainer.py": "de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66",
    "verl/trainer/main_ppo.py": "e3fe6e73b18d63367402a2570c5ba054a2b05443ac40a168524dbf545b1da392",
    "verl/experimental/agent_loop/agent_loop.py": "9bcbc04a190ac809baf5225aaf2c98107cc91b6e3996cf46d99387d7050dd195",
    "verl/experimental/agent_loop/tool_agent_loop.py": "0cc1ccb7327ecb1f71a85739f1fea71953a0ce414bbc6bfa1830f1ad5a6bbd33",
}


def compose_e5_config(run_name: str, smoke: bool = False) -> DictConfig:
    """Compose E5 without applying E3's reduced-data smoke overrides."""
    config = compose_config(DEFAULT_CONFIG, run_name, smoke=False)
    config.turn_credit.diagnostic_freeze_manifest = str(
        PROJECT_ROOT / str(config.turn_credit.diagnostic_freeze_manifest)
    )
    if smoke:
        config.trainer.total_training_steps = 5
        config.trainer.save_freq = 5
        config.trainer.test_freq = 5
    return config


def validate_e5_config(config: DictConfig, run_dir: Path, resume: bool = False) -> None:
    """Fail closed if any frozen Tier-A treatment or E3 control drifts."""
    validate_m4_config(config, run_dir, resume=resume)
    if config.turn_credit.enabled is not True:
        raise ValueError("E5 turn credit must be enabled")
    if float(config.turn_credit.beta) != 0.5:
        raise ValueError("E5 beta is frozen at 0.5")
    if config.turn_credit.heuristic_version != ADOPTION_VERSION:
        raise ValueError("E5 adoption heuristic version drifted")
    if config.turn_credit.ledger_schema_version != LEDGER_SCHEMA_VERSION:
        raise ValueError("E5 turn ledger schema version drifted")
    if config.actor_rollout_ref.rollout.n != 8:
        raise ValueError("E5 rollout n must remain 8, including smoke")
    if config.data.train_batch_size != 64 or config.data.max_response_length != 3072:
        raise ValueError("E5 batch/response controls must match E3, including smoke")
    if config.actor_rollout_ref.rollout.agent.default_agent_loop != "toolcredit_turn_credit_agent":
        raise ValueError("E5 must use the independent turn-credit agent loop")
    if config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length != 512:
        raise ValueError("E5 must preserve veRL's 512-character tool response limit")
    if config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side != "middle":
        raise ValueError("E5 must preserve middle tool-response truncation")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_freeze_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid freeze manifest line: {line!r}")
        entries[relative] = digest
    return entries


def _require_checkpoint_files(path: Path, relative_files: tuple[str, ...], role: str) -> None:
    missing = [relative for relative in relative_files if not (path / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"{role} checkpoint is incomplete at {path}: missing {missing}")


def _validate_checkpoint_locators(protocol: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on every non-null locator and its frozen checkpoint role."""
    checkpoints = protocol.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"raw", "sft", "e3", "e5"}:
        raise ValueError("freeze protocol must define exactly raw/sft/e3/e5 checkpoint roles")

    summaries: dict[str, Any] = {}
    for name, spec in checkpoints.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("role"), str):
            raise TypeError(f"checkpoint role {name} must be a structured mapping")
        locator = spec.get("locator")
        if locator is None:
            if name != "e5" or spec.get("state") != "future":
                raise ValueError(f"only the future E5 role may have a null locator, got {name}")
            summaries[name] = {"role": spec["role"], "state": "future", "locator": None}
            continue
        if not isinstance(locator, str) or not locator.strip():
            raise TypeError(f"checkpoint locator for {name} must be a non-empty string or null")
        path = Path(locator)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{name} checkpoint locator does not exist as a directory: {path}")

        if name == "raw":
            if spec["role"] != "raw_base_model" or path.name != "Qwen3-1.7B":
                raise ValueError(f"raw locator does not match raw_base_model role: {path}")
            _require_checkpoint_files(
                path,
                ("config.json", "model.safetensors.index.json", "tokenizer_config.json"),
                "raw",
            )
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "qwen3":
                raise ValueError(f"raw checkpoint is not the frozen Qwen3 role: {path}")
            if spec.get("config_sha256") != _sha256(path / "config.json"):
                raise ValueError("raw checkpoint config hash does not match its frozen role")
            if spec.get("model_index_sha256") != _sha256(path / "model.safetensors.index.json"):
                raise ValueError("raw checkpoint model index hash does not match its frozen role")
        elif name == "sft":
            expected = (PROJECT_ROOT / "sft/checkpoints/qwen3-1.7b-sft").resolve()
            if spec["role"] != "sft_start" or path != expected:
                raise ValueError(f"SFT locator does not match the frozen start role: {path}")
            _require_checkpoint_files(
                path,
                ("config.json", "model.safetensors", "tokenizer_config.json", "chat_template.jinja"),
                "sft",
            )
            expected_hash = spec.get("model_sha256")
            if expected_hash != _sha256(path / "model.safetensors"):
                raise ValueError("SFT checkpoint model hash does not match its frozen role")
        elif name == "e3":
            expected_relative = Path(
                "rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200"
            )
            expected = (PROJECT_ROOT / expected_relative).resolve()
            if spec["role"] != "e3_grpo_baseline_step_200" or path != expected:
                raise ValueError(f"E3 locator does not match the frozen baseline role: {path}")
            if path.name != "global_step_200" or path.parent.name != "checkpoints":
                raise ValueError(f"E3 locator has invalid veRL checkpoint structure: {path}")
            _require_checkpoint_files(
                path,
                (
                    "actor/model_world_size_1_rank_0.pt",
                    "actor/optim_world_size_1_rank_0.pt",
                    "actor/extra_state_world_size_1_rank_0.pt",
                    "actor/fsdp_config.json",
                    "data.pt",
                ),
                "e3",
            )
            run_dir = path.parents[1]
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            tracker = (path.parent / "latest_checkpointed_iteration.txt").read_text(
                encoding="utf-8"
            ).strip()
            resolved = OmegaConf.load(run_dir / "resolved_config.yaml")
            if status.get("state") != "completed" or tracker != "200":
                raise ValueError("E3 role is not a completed tracker-confirmed step-200 run")
            if Path(str(resolved.actor_rollout_ref.model.path)).resolve() != (
                PROJECT_ROOT / "sft/checkpoints/qwen3-1.7b-sft"
            ).resolve():
                raise ValueError("E3 role did not start from the frozen SFT checkpoint")
            if int(resolved.trainer.total_training_steps) != 200:
                raise ValueError("E3 role does not identify the canonical 200-step run")
            expected_sizes = spec.get("required_file_bytes")
            if not isinstance(expected_sizes, dict):
                raise TypeError("E3 role must freeze required checkpoint file sizes")
            for relative, expected_bytes in expected_sizes.items():
                if (path / relative).stat().st_size != int(expected_bytes):
                    raise ValueError(f"E3 checkpoint file identity drift: {relative}")
            if spec.get("resolved_config_sha256") != _sha256(run_dir / "resolved_config.yaml"):
                raise ValueError("E3 resolved config hash does not match its frozen role")
            if spec.get("status_sha256") != _sha256(run_dir / "status.json"):
                raise ValueError("E3 status hash does not match its frozen role")
            if spec.get("tracker_sha256") != _sha256(
                path.parent / "latest_checkpointed_iteration.txt"
            ):
                raise ValueError("E3 tracker hash does not match its frozen role")
        else:
            raise ValueError("E5 must remain a null future role in the pre-training v2 freeze")

        summaries[name] = {"role": spec["role"], "state": "validated", "locator": str(path)}
    return summaries


def validate_freeze_bundle(manifest_path: Path) -> dict[str, Any]:
    """Verify the complete preregistered bundle before any E5 output."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing M7 freeze manifest: {manifest_path}")
    entries = _read_freeze_manifest(manifest_path)
    required = {str(path.relative_to(PROJECT_ROOT)) for path in FREEZE_FILES}
    if not required.issubset(entries):
        raise ValueError(f"freeze manifest missing required artifacts: {sorted(required - entries.keys())}")
    for relative, expected in entries.items():
        target = PROJECT_ROOT / relative
        if not target.is_file():
            raise FileNotFoundError(f"freeze closure file is missing: {target}")
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"freeze hash mismatch for {relative}: {actual} != {expected}")
    protocol = json.loads((PROJECT_ROOT / "eval/diagnostic_protocol_v2.json").read_text(encoding="utf-8"))
    taxonomy = json.loads((PROJECT_ROOT / "eval/diagnostic_taxonomy_v1.json").read_text(encoding="utf-8"))
    panel_rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / "eval/diagnostic_panel_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if protocol.get("protocol_version") != "toolcredit_diagnostic_v2":
        raise ValueError("unexpected diagnostic protocol version")
    if taxonomy.get("taxonomy_version") != "toolcredit_failure_taxonomy_v1":
        raise ValueError("unexpected diagnostic taxonomy version")
    if len(panel_rows) != len({row["question_id"] for row in panel_rows}):
        raise ValueError("diagnostic panel contains duplicate question IDs")
    checkpoint_roles = _validate_checkpoint_locators(protocol)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "closure_file_count": len(entries),
        "panel_question_count": len(panel_rows),
        "protocol_version": protocol["protocol_version"],
        "taxonomy_version": taxonomy["taxonomy_version"],
        "checkpoint_roles": checkpoint_roles,
    }


def _verl_source_hashes() -> dict[str, str]:
    import verl

    root = Path(verl.__file__).resolve().parent
    actual: dict[str, str] = {}
    for relative, expected in PINNED_VERL_HASHES.items():
        path = root.parent / relative
        digest = _sha256(path)
        if digest != expected:
            raise RuntimeError(f"pinned veRL source drift for {relative}: {digest} != {expected}")
        actual[relative] = digest
    return actual


def resource_preflight_e5(config: DictConfig, smoke: bool) -> dict[str, Any]:
    """Apply M6's GPU, data, freeze, and recomputable disk gates."""
    import pyarrow.parquet as pq

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E5 requires exactly one visible CUDA GPU; none is currently available")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)
    disk = shutil.disk_usage(PROJECT_ROOT)
    remaining_peak_gib = 25 if smoke else 103
    minimum_gib = remaining_peak_gib + 30
    if disk.free < minimum_gib << 30:
        raise RuntimeError(
            f"E5 disk gate failed: {disk.free / 2**30:.1f} GiB free, need {minimum_gib} GiB"
        )
    train_rows = pq.read_metadata(config.data.train_files).num_rows
    val_rows = pq.read_metadata(config.data.val_files).num_rows
    if (train_rows, val_rows) != (5203, 100):
        raise ValueError(f"E5 must use full E3 data even for smoke, got {train_rows}/{val_rows}")
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
        "disk_free_gib": round(disk.free / 2**30, 2),
        "remaining_peak_write_gib": remaining_peak_gib,
        "disk_safety_margin_gib": 30,
        "disk_minimum_gib": minimum_gib,
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
    }


def _proof_path(config: DictConfig) -> Path:
    return Path(config.trainer.default_local_dir).parent / "wrapper_installation.json"


class TurnCreditTaskRunner:
    """Mixin implementation is attached to veRL's TaskRunner below."""

    def run(self, config: DictConfig) -> None:
        from verl.trainer.main_ppo import TaskRunner
        from verl.trainer.ppo import ray_trainer

        if not isinstance(self, TaskRunner):
            raise TypeError("TurnCreditTaskRunner must be combined with veRL TaskRunner")
        source_hashes = _verl_source_hashes()
        base_compute = ray_trainer.compute_advantage
        base_log = ray_trainer.RayPPOTrainer._log_rollout_data
        if hasattr(base_compute, "__toolcredit_wrapper_proof__"):
            raise RuntimeError("refusing to stack turn-credit advantage wrappers")
        proof_payload = {
            "boundary_version": "e5_runtime_wrapper_v1",
            "beta": float(config.turn_credit.beta),
            "heuristic_version": str(config.turn_credit.heuristic_version),
            "ledger_schema_version": str(config.turn_credit.ledger_schema_version),
            "base_compute_module": base_compute.__module__,
            "base_compute_name": base_compute.__name__,
            "base_compute_source_file": inspect.getsourcefile(base_compute),
            "verl_source_hashes": source_hashes,
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        proof_id = "e5-wrapper-v1:" + _sha256_bytes(
            json.dumps(proof_payload, sort_keys=True).encode("utf-8")
        )
        proof_payload["proof_id"] = proof_id
        wrapped_compute = make_compute_advantage_wrapper(
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
        print("TOOLCREDIT_E5_WRAPPER_ACTIVE " + json.dumps(proof_payload, sort_keys=True))
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

    concrete = type("E5TurnCreditTaskRunner", (TurnCreditTaskRunner, TaskRunner), {})
    return ray.remote(num_cpus=1)(concrete)


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
    config = compose_e5_config(args.run_name, smoke=args.smoke)
    validate_e5_config(config, run_dir, resume=args.resume)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    if args.resume:
        config.trainer.val_before_train = False
    preflight = resource_preflight_e5(config, smoke=args.smoke)
    launch(config, run_dir, preflight, resume=args.resume)


if __name__ == "__main__":
    main()
