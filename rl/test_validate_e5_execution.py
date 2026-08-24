from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import rl.validate_e5_canonical_smoke as canonical_smoke
from rl.e5_recovery import proofs_in_train_file, validate_wrapper_proof_payload
from rl.validate_e5_canonical_smoke import diff_paths


def _proof(installed_at: str, *, beta: float = 0.5, restored: bool = True) -> dict:
    payload = {
        "boundary_version": "e5_runtime_wrapper_v1",
        "beta": beta,
        "heuristic_version": "toolcredit_adoption_v1",
        "ledger_schema_version": "toolcredit_turn_ledger_v1",
        "base_compute_module": "verl.trainer.ppo.ray_trainer",
        "base_compute_name": "compute_advantage",
        "base_compute_source_file": "/pinned/verl/trainer/ppo/ray_trainer.py",
        "verl_source_hashes": {"ray_trainer.py": "abc"},
        "installed_at": installed_at,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload.update(
        proof_id=f"e5-wrapper-v1:{digest}", installed=True, restored=restored
    )
    return payload


def _write_overlay(
    root: Path, name: str, checkpoint: int, proof: dict, *, interrupted: bool = True
) -> None:
    archive = root / "recovery" / name
    (archive / "train").mkdir(parents=True)
    (archive / "recovery.json").write_text(
        json.dumps({"checkpoint_step": checkpoint}) + "\n", encoding="utf-8"
    )
    overlay = {
        "schema_version": "toolcredit_e5_recovery_proof_v1",
        "checkpoint_step": checkpoint,
        "wrapper_proof": proof,
        "proof_provenance": "fixture",
        "interrupted_without_restore": interrupted,
        "current_attempt_steps": [],
        "stale_pending_recompute_steps": [],
    }
    path = archive / "e5_recovery_proof.json"
    path.write_text(json.dumps(overlay) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (archive / "e5_recovery_proof.sha256").write_text(
        f"{digest}  e5_recovery_proof.json\n", encoding="utf-8"
    )


def test_diff_paths_is_exact_and_nested() -> None:
    left = {"a": {"b": 1, "c": 2}, "same": [1, 2]}
    right = {"a": {"b": 1, "c": 3, "d": 4}, "same": [1, 2]}
    assert diff_paths(left, right) == {"a.c", "a.d"}


def test_postrun_config_validation_does_not_reapply_existing_run_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed_run = tmp_path / "completed_smoke"
    completed_run.mkdir()
    calls: list[tuple[object, Path, bool]] = []

    def fake_validate(config: object, run_dir: Path, resume: bool = False) -> None:
        if run_dir.exists():
            raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
        calls.append((config, run_dir, resume))

    monkeypatch.setattr(canonical_smoke, "validate_e5_config", fake_validate)
    config = object()
    canonical_smoke._validate_postrun_e5_config(config, completed_run)

    assert calls == [
        (config, tmp_path / ".completed_smoke.postrun-config-validation-target", False)
    ]


def test_postrun_config_validation_requires_existing_completed_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="completed smoke directory does not exist"):
        canonical_smoke._validate_postrun_e5_config(object(), tmp_path / "missing")


def test_single_wrapper_segment_remains_valid_without_recovery(tmp_path: Path) -> None:
    proof = _proof("2026-08-23T00:00:00+00:00")
    (tmp_path / "wrapper_installation.json").write_text(json.dumps(proof), encoding="utf-8")
    result = canonical_smoke._validate_segmented_proofs(
        tmp_path, {1: proof["proof_id"], 2: proof["proof_id"]}, recovery_aware=False
    )
    assert result["segments"] == [
        {"start_step": 1, "end_step": 2, "proof_id": proof["proof_id"]}
    ]


def test_one_recovery_requires_checkpoint_bound_proof_transition(tmp_path: Path) -> None:
    old = _proof("2026-08-23T00:00:00+00:00", restored=False)
    new = _proof("2026-08-23T01:00:00+00:00")
    (tmp_path / "wrapper_installation.json").write_text(json.dumps(new), encoding="utf-8")
    _write_overlay(tmp_path, "resume_from_2_a", 2, old)
    result = canonical_smoke._validate_segmented_proofs(
        tmp_path,
        {1: old["proof_id"], 2: old["proof_id"], 3: new["proof_id"]},
        recovery_aware=True,
    )
    assert [item["end_step"] for item in result["segments"]] == [2, 3]


def test_multiple_recoveries_are_validated_as_distinct_segments(tmp_path: Path) -> None:
    first = _proof("2026-08-23T00:00:00+00:00", restored=False)
    second = _proof("2026-08-23T01:00:00+00:00", restored=False)
    final = _proof("2026-08-23T02:00:00+00:00")
    (tmp_path / "wrapper_installation.json").write_text(json.dumps(final), encoding="utf-8")
    _write_overlay(tmp_path, "resume_from_2_a", 2, first)
    _write_overlay(tmp_path, "resume_from_4_b", 4, second)
    step_proofs = {
        1: first["proof_id"], 2: first["proof_id"],
        3: second["proof_id"], 4: second["proof_id"], 5: final["proof_id"],
    }
    assert len(
        canonical_smoke._validate_segmented_proofs(
            tmp_path, step_proofs, recovery_aware=True
        )["segments"]
    ) == 3


def test_transition_away_from_checkpoint_boundary_fails_closed(tmp_path: Path) -> None:
    old = _proof("2026-08-23T00:00:00+00:00", restored=False)
    new = _proof("2026-08-23T01:00:00+00:00")
    (tmp_path / "wrapper_installation.json").write_text(json.dumps(new), encoding="utf-8")
    _write_overlay(tmp_path, "resume_from_1_a", 1, old)
    with pytest.raises(ValueError, match="not backed by a recovery checkpoint boundary"):
        canonical_smoke._validate_segmented_proofs(
            tmp_path,
            {1: old["proof_id"], 2: old["proof_id"], 3: new["proof_id"]},
            recovery_aware=True,
        )


def test_forged_proof_id_and_invariant_drift_fail_closed(tmp_path: Path) -> None:
    forged = _proof("2026-08-23T00:00:00+00:00")
    forged["beta"] = 0.25
    with pytest.raises(ValueError, match="does not match its payload"):
        validate_wrapper_proof_payload(forged)

    old = _proof("2026-08-23T00:00:00+00:00", beta=0.25, restored=False)
    new = _proof("2026-08-23T01:00:00+00:00")
    (tmp_path / "wrapper_installation.json").write_text(json.dumps(new), encoding="utf-8")
    _write_overlay(tmp_path, "resume_from_1_a", 1, old)
    with pytest.raises(ValueError, match="invariant drift"):
        canonical_smoke._validate_segmented_proofs(
            tmp_path, {1: old["proof_id"], 2: new["proof_id"]}, recovery_aware=True
        )


def test_mixed_proof_within_one_step_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "1.jsonl"
    path.write_text(
        json.dumps({"turn_credit_wrapper_proof": "proof-a"})
        + "\n"
        + json.dumps({"turn_credit_wrapper_proof": "proof-b"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mixes E5 wrapper proofs"):
        proofs_in_train_file(path)
