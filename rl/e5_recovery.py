"""Recovery proof archival for E5 without changing the frozen training launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from rl.custom.turn_advantage import WRAPPER_PROOF_KEY
from rl.launch.e3_grpo_baseline import _sha256, archive_interrupted_attempt

PROOF_DYNAMIC_FIELDS = {"proof_id", "installed", "restored", "restored_at"}


def validate_wrapper_proof_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a runtime proof and return its experiment-invariant identity."""
    proof_body = {key: value for key, value in payload.items() if key not in PROOF_DYNAMIC_FIELDS}
    expected = "e5-wrapper-v1:" + hashlib.sha256(
        json.dumps(proof_body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if payload.get("proof_id") != expected:
        raise ValueError("E5 wrapper proof ID does not match its payload")
    required = {
        "boundary_version",
        "beta",
        "heuristic_version",
        "ledger_schema_version",
        "base_compute_module",
        "base_compute_name",
        "base_compute_source_file",
        "verl_source_hashes",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"E5 wrapper proof is missing invariant fields: {sorted(missing)}")
    return {
        key: value
        for key, value in payload.items()
        if key not in PROOF_DYNAMIC_FIELDS | {"installed_at"}
    }


def proofs_in_train_file(path: Path) -> set[str]:
    proofs: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        proof = json.loads(line).get(WRAPPER_PROOF_KEY)
        if not isinstance(proof, str) or not proof:
            raise ValueError(f"missing E5 wrapper proof in {path}")
        proofs.add(proof)
    if len(proofs) != 1:
        raise ValueError(f"step file mixes E5 wrapper proofs: {path}: {sorted(proofs)}")
    return proofs


def _proof_from_log(run_dir: Path, proof_id: str) -> dict[str, Any]:
    log_path = run_dir.parent / f"{run_dir.name}.log"
    marker = "TOOLCREDIT_E5_WRAPPER_ACTIVE "
    if not log_path.is_file():
        raise FileNotFoundError(f"cannot recover archived wrapper proof without {log_path}")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        offset = line.find(marker)
        if offset < 0:
            continue
        try:
            payload = json.loads(line[offset + len(marker) :])
        except json.JSONDecodeError:
            continue
        if payload.get("proof_id") == proof_id:
            validate_wrapper_proof_payload(payload)
            return payload
    raise ValueError(f"wrapper proof {proof_id} is absent from {log_path}")


def _write_overlay(
    archive_dir: Path,
    wrapper: dict[str, Any],
    *,
    current_steps: list[int],
    stale_steps: list[int],
    provenance: str,
) -> None:
    validate_wrapper_proof_payload(wrapper)
    recovery = json.loads((archive_dir / "recovery.json").read_text(encoding="utf-8"))
    overlay = {
        "schema_version": "toolcredit_e5_recovery_proof_v1",
        "checkpoint_step": int(recovery["checkpoint_step"]),
        "wrapper_proof": wrapper,
        "proof_provenance": provenance,
        "interrupted_without_restore": not bool(wrapper.get("restored")),
        "current_attempt_steps": current_steps,
        "stale_pending_recompute_steps": stale_steps,
    }
    path = archive_dir / "e5_recovery_proof.json"
    path.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
    (archive_dir / "e5_recovery_proof.sha256").write_text(
        f"{_sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def migrate_existing_recovery_proofs(run_dir: Path) -> None:
    """Add proof overlays without mutating existing recovery ledgers or hash manifests."""
    root = run_dir / "recovery"
    if not root.is_dir():
        return
    for archive in sorted(path for path in root.glob("resume_from_*") if path.is_dir()):
        if (archive / "e5_recovery_proof.json").is_file():
            continue
        paths = sorted((archive / "train").glob("*.jsonl"))
        if not paths:
            continue
        proof_ids = set().union(*(proofs_in_train_file(path) for path in paths))
        if len(proof_ids) != 1:
            raise ValueError(f"legacy recovery archive mixes wrapper proofs: {archive}")
        wrapper = _proof_from_log(run_dir, next(iter(proof_ids)))
        _write_overlay(
            archive,
            wrapper,
            current_steps=sorted(int(path.stem) for path in paths),
            stale_steps=[],
            provenance="recovered_from_immutable_run_log_marker",
        )


def prepare_resume(run_dir: Path, stamp: str) -> dict[str, Any]:
    """Preserve and classify current post-checkpoint rows before frozen launcher resume."""
    migrate_existing_recovery_proofs(run_dir)
    wrapper_path = run_dir / "wrapper_installation.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    validate_wrapper_proof_payload(wrapper)
    checkpoint = int(
        (run_dir / "checkpoints/latest_checkpointed_iteration.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    by_proof: dict[str, list[int]] = {}
    for path in sorted((run_dir / "predictions/train").glob("*.jsonl")):
        if path.stem.isdigit() and int(path.stem) > checkpoint:
            proof_id = next(iter(proofs_in_train_file(path)))
            by_proof.setdefault(proof_id, []).append(int(path.stem))
    current = sorted(by_proof.pop(str(wrapper["proof_id"]), []))
    stale = sorted(step for steps in by_proof.values() for step in steps)
    metadata = archive_interrupted_attempt(run_dir, f"{stamp}_proof")
    archive = Path(metadata["archive_dir"])
    shutil.copy2(wrapper_path, archive / "wrapper_installation.json")
    _write_overlay(
        archive,
        wrapper,
        current_steps=current,
        stale_steps=stale,
        provenance="copied_from_run_before_frozen_launcher_resume",
    )
    return {**metadata, "current_attempt_steps": current, "stale_steps": stale}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_resume(args.run_dir.resolve(), args.stamp), indent=2))


if __name__ == "__main__":
    main()
