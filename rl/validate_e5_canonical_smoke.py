"""Fail-closed canonical E5 smoke gate used before an overnight formal launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from rl.analyze_e5 import _iter_rows, _validate_audit
from rl.e5_recovery import validate_wrapper_proof_payload
from rl.launch.e3_grpo_baseline import DEFAULT_CONFIG as E3_CONFIG
from rl.launch.e3_grpo_baseline import PROJECT_ROOT, compose_config
from rl.launch.e5_turn_credit import (
    compose_e5_config,
    validate_e5_config,
    validate_freeze_bundle,
)
from rl.monitor_run import read_native_scalars

FREEZE_V2_SHA256 = "6fa185e18c17761cf9c238cb2a9754e245cda8179136e59cb69c729c9b07f3f0"
SFT_MODEL_SHA256 = "cb5f43dafad692b2a3dc37ad6bc22f86486ada7682739036a22b6d6117bcb235"
HUMAN_LABEL_SHA256 = "d3dbe70aee2e6b7b172175bec336af0eeaa6077ba56fbd5efd1af8cc5590b1f4"
HUMAN_GATE_SHA256 = "96c27d610aa29331be91589be060961ea18e51dd8baa8190585f4c9abdf78a57"
HUMAN_RUN = PROJECT_ROOT / "rl/runs/e5_turn_credit_smoke_20260822_222603"
SFT_PATH = (PROJECT_ROOT / "sft/checkpoints/qwen3-1.7b-sft").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _resolved(config: Any) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config must be a mapping")
    return value


def _validate_postrun_e5_config(config: Any, completed_run_dir: Path) -> None:
    """Run frozen config checks without reapplying the pre-launch directory guard."""
    if not completed_run_dir.is_dir():
        raise FileNotFoundError(f"completed smoke directory does not exist: {completed_run_dir}")
    validation_target = completed_run_dir.parent / (
        f".{completed_run_dir.name}.postrun-config-validation-target"
    )
    if validation_target.exists():
        raise FileExistsError(f"post-run validation sentinel already exists: {validation_target}")
    # validate_e5_config uses run_dir only for its pre-launch reuse guard. The
    # exact config/run identity comparison below still uses completed_run_dir.
    validate_e5_config(config, validation_target, resume=False)


def validate_resolved_config(run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    actual_config = OmegaConf.load(run_dir / "resolved_config.yaml")
    _validate_postrun_e5_config(actual_config, run_dir)
    actual = _resolved(actual_config)
    expected = _resolved(compose_e5_config(run_dir.name, smoke=smoke))
    exact_drift = sorted(diff_paths(actual, expected))
    if exact_drift:
        raise ValueError(f"resolved E5 config differs from the approved composition: {exact_drift}")

    e3 = _resolved(compose_config(E3_CONFIG, run_dir.name, smoke=False))
    observed_diff = diff_paths(e3, actual)
    allowed = {
        "turn_credit",
        "actor_rollout_ref.rollout.agent.default_agent_loop",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
        "actor_rollout_ref.rollout.trace.project_name",
        "trainer.project_name",
    }
    if smoke:
        allowed |= {
            "trainer.total_training_steps",
            "trainer.save_freq",
            "trainer.test_freq",
        }
    if observed_diff != allowed:
        raise ValueError(
            f"E3-control diff gate failed: {sorted(observed_diff)} != {sorted(allowed)}"
        )
    if Path(str(actual_config.actor_rollout_ref.model.path)).resolve() != SFT_PATH:
        raise ValueError("E5 did not start from the frozen SFT checkpoint")
    if actual_config.trainer.resume_from_path is not None:
        raise ValueError("fresh E5 run has a non-null resume path")
    return {"exact_config_match": True, "e3_control_diff_paths": sorted(observed_diff)}


def _expected_validation_ids() -> set[str]:
    table = pq.read_table(PROJECT_ROOT / "rl/data/e3_val_math500_100.parquet", columns=["extra_info"])
    return {str(row["extra_info"]["id"]) for row in table.to_pylist()}


def validate_validation_files(run_dir: Path, expected_steps: list[int]) -> dict[str, Any]:
    expected_ids = _expected_validation_ids()
    validation_dir = run_dir / "predictions/validation"
    actual_steps = sorted(int(path.stem) for path in validation_dir.glob("*.jsonl"))
    if actual_steps != expected_steps:
        raise ValueError(f"validation cadence mismatch: {actual_steps} != {expected_steps}")
    pass_at_1: dict[str, float] = {}
    for step in expected_steps:
        rows = [
            json.loads(line)
            for line in (validation_dir / f"{step}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        ids = [str(row["sample_id"]) for row in rows]
        if len(rows) != 100 or len(set(ids)) != 100 or set(ids) != expected_ids:
            raise ValueError(f"validation step {step} is not the frozen complete MATH500-100 panel")
        values = [float(row["acc"]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite validation metric at step {step}")
        pass_at_1[str(step)] = sum(values) / len(values)
    return {"steps": actual_steps, "pass_at_1": pass_at_1}


def _load_recovery_proofs(run_dir: Path) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    for path in sorted((run_dir / "recovery").glob("resume_from_*/e5_recovery_proof.json")):
        overlay = json.loads(path.read_text(encoding="utf-8"))
        if overlay.get("schema_version") != "toolcredit_e5_recovery_proof_v1":
            raise ValueError(f"unsupported E5 recovery proof schema: {path}")
        digest_path = path.with_suffix(".sha256")
        expected = digest_path.read_text(encoding="utf-8").split()[0]
        if sha256(path) != expected:
            raise ValueError(f"E5 recovery proof hash mismatch: {path}")
        validate_wrapper_proof_payload(overlay["wrapper_proof"])
        recovery = json.loads((path.parent / "recovery.json").read_text(encoding="utf-8"))
        if int(overlay["checkpoint_step"]) != int(recovery["checkpoint_step"]):
            raise ValueError(f"E5 recovery checkpoint mismatch: {path}")
        archived_steps = sorted(int(item.stem) for item in (path.parent / "train").glob("*.jsonl"))
        classified = sorted(
            list(map(int, overlay["current_attempt_steps"]))
            + list(map(int, overlay["stale_pending_recompute_steps"]))
        )
        if archived_steps != classified or len(classified) != len(set(classified)):
            raise ValueError(f"E5 recovery step classification mismatch: {path}")
        current_proof = str(overlay["wrapper_proof"]["proof_id"])
        for step in overlay["current_attempt_steps"]:
            rows = list(_iter_rows([path.parent / "train" / f"{step}.jsonl"]))
            if {str(row["turn_credit_wrapper_proof"]) for _, row in rows} != {current_proof}:
                raise ValueError(f"E5 recovery current-attempt proof mismatch: {path}, step {step}")
        overlays.append(overlay)
    return overlays


def _validate_segmented_proofs(
    run_dir: Path, step_proofs: dict[int, str], *, recovery_aware: bool
) -> dict[str, Any]:
    wrapper = json.loads((run_dir / "wrapper_installation.json").read_text(encoding="utf-8"))
    invariant = validate_wrapper_proof_payload(wrapper)
    if not wrapper.get("installed") or not wrapper.get("restored"):
        raise ValueError("wrapper installation/restoration proof is incomplete")
    segments: list[dict[str, Any]] = []
    for step, proof_id in sorted(step_proofs.items()):
        if not segments or segments[-1]["proof_id"] != proof_id:
            segments.append({"start_step": step, "end_step": step, "proof_id": proof_id})
        else:
            segments[-1]["end_step"] = step
    if not recovery_aware:
        if len(segments) != 1 or segments[0]["proof_id"] != str(wrapper["proof_id"]):
            raise ValueError("rollout wrapper proof IDs differ from normal-shutdown proof")
        return {"segments": segments, "recovery_event_count": 0}

    overlays = _load_recovery_proofs(run_dir)
    payloads = {str(wrapper["proof_id"]): wrapper}
    for overlay in overlays:
        payload = overlay["wrapper_proof"]
        payloads[str(payload["proof_id"])] = payload
        if validate_wrapper_proof_payload(payload) != invariant:
            raise ValueError("E5 recovery wrapper invariant drift")
    if segments[-1]["proof_id"] != str(wrapper["proof_id"]):
        raise ValueError("final canonical segment does not use the normal-shutdown wrapper proof")
    for segment in segments:
        if segment["proof_id"] not in payloads:
            raise ValueError("canonical segment has no authenticated wrapper proof payload")
    for previous, current in zip(segments, segments[1:]):
        boundary = int(previous["end_step"])
        if int(current["start_step"]) != boundary + 1:
            raise ValueError("non-contiguous E5 proof segments")
        matching = [
            item
            for item in overlays
            if int(item["checkpoint_step"]) == boundary
            and str(item["wrapper_proof"]["proof_id"]) == previous["proof_id"]
        ]
        if not matching:
            raise ValueError("wrapper proof transition is not backed by a recovery checkpoint boundary")
        if not any(
            item.get("interrupted_without_restore") or item["wrapper_proof"].get("restored")
            for item in matching
        ):
            raise ValueError("prior E5 segment lacks interruption/restoration evidence")
    return {"segments": segments, "recovery_event_count": len(overlays)}


def validate_rollouts(
    run_dir: Path,
    expected_steps: int,
    *,
    recovery_aware: bool = False,
    audit_validator: Any = _validate_audit,
) -> dict[str, Any]:
    train_dir = run_dir / "predictions/train"
    paths = sorted(train_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    actual_steps = [int(path.stem) for path in paths]
    if actual_steps != list(range(1, expected_steps + 1)):
        raise ValueError(f"train step files are incomplete: {actual_steps}")

    trajectory_uids: set[str] = set()
    proof_ids: set[str] = set()
    step_proofs: dict[int, str] = {}
    status_counts: Counter[str] = Counter()
    corrections: Counter[str] = Counter()
    trajectories = 0
    turns = 0
    parser_errors = 0
    invalid = 0
    truncated = 0
    numeric_row_fields = ("score", "acc", "tool_error_rate", "invalid", "truncated")
    for path in paths:
        prompt_counts: Counter[str] = Counter()
        rows_in_step = 0
        for _, row in _iter_rows([path]):
            audit = row.get("turn_credit_audit")
            if not isinstance(audit, dict):
                raise ValueError(f"silent audit loss in {path}")
            if row.get("turn_credit_wrapper_proof") != audit.get("wrapper_proof"):
                raise ValueError("wrapper proof mismatch between row and audit")
            audit_validator(audit)
            uid = str(audit["trajectory_uid"])
            if uid in trajectory_uids:
                raise ValueError(f"duplicate trajectory UID: {uid}")
            trajectory_uids.add(uid)
            prompt_counts[str(audit["prompt_uid"])] += 1
            proof_ids.add(str(audit["wrapper_proof"]))
            rows_in_step += 1
            trajectories += 1
            turns += len(audit["turns"])
            for turn in audit["turns"]:
                status_counts[str(turn["adoption_status"])] += 1
                correction = float(turn["correction"])
                if not math.isfinite(correction):
                    raise ValueError("NaN/Inf turn correction")
                corrections["positive" if correction > 0 else "negative" if correction < 0 else "zero"] += 1
            for field in numeric_row_fields:
                if field in row and not math.isfinite(float(row[field])):
                    raise ValueError(f"NaN/Inf rollout field {field}")
            parser_errors += int(float(row.get("tool_parse_errors", 0)) > 0)
            invalid += int(float(row.get("invalid", 0)) > 0)
            truncated += int(float(row.get("truncated", 0)) > 0)
        if rows_in_step != 512:
            raise ValueError(f"step {path.stem} contains {rows_in_step} instead of 512 rollouts")
        if len(prompt_counts) != 64 or set(prompt_counts.values()) != {8}:
            raise ValueError(f"step {path.stem} is not 64 complete groups x 8 rollouts")
        proofs_for_step = {
            str(row["turn_credit_wrapper_proof"])
            for _, row in _iter_rows([path])
        }
        if len(proofs_for_step) != 1:
            raise ValueError(f"step {path.stem} mixes wrapper proofs")
        step_proofs[int(path.stem)] = next(iter(proofs_for_step))

    proof_validation = _validate_segmented_proofs(
        run_dir, step_proofs, recovery_aware=recovery_aware
    )

    native = read_native_scalars(run_dir / "tensorboard")
    scalar_count = 0
    for points in native.values():
        for point in points:
            scalar_count += 1
            if not math.isfinite(float(point["value"])):
                raise ValueError("NaN/Inf TensorBoard scalar")
    return {
        "steps": actual_steps,
        "trajectory_count": trajectories,
        "turn_count": turns,
        "unique_trajectory_uid_count": len(trajectory_uids),
        "formula_turn_span_mask_uid_mismatches": 0,
        "wrapper_proof_ids": sorted(proof_ids),
        "wrapper_proof_segments": proof_validation["segments"],
        "recovery_event_count": proof_validation["recovery_event_count"],
        "wrapper_installed": True,
        "wrapper_restored": True,
        "tensorboard_finite_scalar_count": scalar_count,
        "adoption_status_counts": dict(sorted(status_counts.items())),
        "correction_sign_counts": dict(sorted(corrections.items())),
        "parser_error_trajectories": parser_errors,
        "invalid_trajectories": invalid,
        "truncated_trajectories": truncated,
    }


def validate_checkpoint(run_dir: Path, step: int) -> dict[str, Any]:
    checkpoint_root = run_dir / "checkpoints"
    if (checkpoint_root / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip() != str(step):
        raise ValueError("checkpoint tracker does not identify the required final step")
    checkpoint = checkpoint_root / f"global_step_{step}"
    required = {
        "model": checkpoint / "actor/model_world_size_1_rank_0.pt",
        "optimizer": checkpoint / "actor/optim_world_size_1_rank_0.pt",
        "extra": checkpoint / "actor/extra_state_world_size_1_rank_0.pt",
        "fsdp": checkpoint / "actor/fsdp_config.json",
        "dataloader": checkpoint / "data.pt",
    }
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"incomplete step-{step} checkpoint: {missing}")

    model = torch.load(required["model"], map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(model, dict) or "model.embed_tokens.weight" not in model or "lm_head.weight" not in model:
        raise ValueError("model checkpoint is not readable as the expected actor state")
    del model
    optimizer = torch.load(required["optimizer"], map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(optimizer, dict) or not {"state", "param_groups"}.issubset(optimizer):
        raise ValueError("optimizer checkpoint is not readable")
    del optimizer
    extra = torch.load(required["extra"], map_location="cpu", weights_only=False)
    if not isinstance(extra, dict) or not {"lr_scheduler", "rng"}.issubset(extra):
        raise ValueError("scheduler/RNG checkpoint state is incomplete")
    dataloader = torch.load(required["dataloader"], map_location="cpu", weights_only=False)
    if not isinstance(dataloader, dict) or not {"dataset_state", "_sampler_iter_state"}.issubset(dataloader):
        raise ValueError("DataLoader checkpoint state is incomplete")
    total_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    return {
        "step": step,
        "path": str(checkpoint),
        "total_bytes": total_bytes,
        "required_file_bytes": {name: path.stat().st_size for name, path in required.items()},
        "model_optimizer_scheduler_rng_dataloader_readable": True,
    }


def validate_human_gate() -> dict[str, Any]:
    labels = HUMAN_RUN / "analysis/e5_adoption_human_review_labels.jsonl"
    gate_path = HUMAN_RUN / "analysis/e5_human_adoption_gate.json"
    if sha256(labels) != HUMAN_LABEL_SHA256 or sha256(gate_path) != HUMAN_GATE_SHA256:
        raise ValueError("user-confirmed human gate artifact hash drift")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected_counts = {"decisive_denominator": 15, "true_positive": 14, "false_positive": 1, "uncertain": 0}
    if gate.get("counts") != expected_counts or gate.get("precision") != 14 / 15:
        raise ValueError("user-confirmed human adoption counts drifted")
    if not gate.get("gate_passed") or gate.get("agent_preliminary_labels_used") is not False:
        raise ValueError("human adoption gate is no longer valid or agent-only labels leaked in")
    return {
        "joined_manifest_sha256": HUMAN_LABEL_SHA256,
        "gate_report_sha256": HUMAN_GATE_SHA256,
        "counts": expected_counts,
        "precision": gate["precision"],
        "wilson_95_ci": gate["wilson_95_ci"],
        "gate_passed": True,
    }


def validate_canonical_smoke(run_dir: Path, launch_ledger: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"smoke did not exit normally: {status}")
    ledger = json.loads(launch_ledger.read_text(encoding="utf-8"))
    if ledger.get("run_name") != run_dir.name or ledger.get("resumed") is not False:
        raise ValueError("fresh-launch ledger does not prove resumed=false")
    if Path(str(ledger.get("checkpoint"))).resolve() != SFT_PATH:
        raise ValueError("fresh-launch ledger does not identify the frozen SFT checkpoint")

    freeze = validate_freeze_bundle(PROJECT_ROOT / "eval/diagnostic_freeze_v2.sha256")
    if freeze["manifest_sha256"] != FREEZE_V2_SHA256:
        raise ValueError("freeze v2 manifest hash drift")
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("checkpoint_model_sha256") != SFT_MODEL_SHA256:
        raise ValueError("smoke preflight SFT model hash drift")
    if preflight.get("freeze", {}).get("manifest_sha256") != FREEZE_V2_SHA256:
        raise ValueError("smoke preflight did not use freeze v2")

    config = validate_resolved_config(run_dir, smoke=True)
    validation = validate_validation_files(run_dir, [0, 5])
    rollouts = validate_rollouts(run_dir, 5)
    checkpoint = validate_checkpoint(run_dir, 5)
    human = validate_human_gate()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal launch requires one healthy visible GPU")
    free_gpu, total_gpu = torch.cuda.mem_get_info(0)
    disk = shutil.disk_usage(PROJECT_ROOT)
    smoke_checkpoint_gib = math.ceil(checkpoint["total_bytes"] / 2**30)
    formal_minimum_gib = 103 + 30 + smoke_checkpoint_gib
    if disk.free < formal_minimum_gib << 30:
        raise RuntimeError(
            f"formal disk gate failed: {disk.free / 2**30:.1f} GiB free < {formal_minimum_gib} GiB"
        )

    report = {
        "schema_version": "toolcredit_canonical_smoke_gate_v1",
        "run_name": run_dir.name,
        "gate_passed": True,
        "fresh_launch": {"resumed": False, "checkpoint": str(SFT_PATH), "model_sha256": SFT_MODEL_SHA256},
        "freeze": freeze,
        "resolved_config": config,
        "validation": validation,
        "rollouts": rollouts,
        "checkpoint": checkpoint,
        "human_adoption_gate": human,
        "resources_before_formal": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_free_gib": round(free_gpu / 2**30, 2),
            "gpu_total_gib": round(total_gpu / 2**30, 2),
            "disk_free_gib": round(disk.free / 2**30, 2),
            "formal_minimum_including_retained_smoke_checkpoint_gib": formal_minimum_gib,
        },
        "frozen_treatment": {"beta": 0.5, "adoption_version": "toolcredit_adoption_v1"},
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    report_path = analysis_dir / "canonical_smoke_gate.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "evidence_classification.json").write_text(
        json.dumps(
            {
                "classification": "canonical_v2_five_step_smoke",
                "canonical_five_step_gate_passed": True,
                "gate_report": "analysis/canonical_smoke_gate.json",
                "resume_allowed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--launch-ledger", type=Path, required=True)
    args = parser.parse_args()
    report = validate_canonical_smoke(args.run_dir.resolve(), args.launch_ledger.resolve())
    print(json.dumps({"run_name": report["run_name"], "gate_passed": True}, indent=2))


if __name__ == "__main__":
    main()
