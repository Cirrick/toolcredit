"""Audit a deliberately stopped E5 smoke without claiming normal completion."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from rl.analyze_e5 import _iter_rows, _validate_audit


def _active_proof(log_path: Path) -> dict[str, Any]:
    marker = "TOOLCREDIT_E5_WRAPPER_ACTIVE "
    matches = [line.split(marker, 1)[1] for line in log_path.read_text(encoding="utf-8").splitlines() if marker in line]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one wrapper activation line, got {len(matches)}")
    return json.loads(matches[0])


def analyze_partial(run_dir: Path) -> dict[str, Any]:
    train_dir = run_dir / "predictions/train"
    prediction_paths = sorted(train_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    completed_steps = [int(path.stem) for path in prediction_paths]
    if completed_steps != list(range(1, max(completed_steps, default=0) + 1)):
        raise ValueError(f"noncontiguous completed steps: {completed_steps}")

    active = _active_proof(run_dir.with_suffix(".log"))
    if not active.get("installed") or active.get("restored"):
        raise ValueError("wrapper activation line does not show the expected active state")
    wrapper_proofs: set[str] = set()
    trajectory_uids: set[str] = set()
    prompt_counts_by_step: dict[int, Counter[str]] = {}
    turns_per_trajectory: Counter[int] = Counter()
    adoption_statuses: Counter[str] = Counter()
    corrections: Counter[str] = Counter()
    trajectories = 0
    turns = 0
    policy_span_tokens = 0
    tool_span_tokens = 0
    parser_error_trajectories = 0
    invalid_trajectories = 0
    truncated_trajectories = 0

    for path in prediction_paths:
        prompt_counts: Counter[str] = Counter()
        rows_in_step = 0
        for _, row in _iter_rows([path]):
            audit = row.get("turn_credit_audit")
            if not isinstance(audit, dict):
                raise ValueError(f"missing turn-credit audit in {path}")
            if row.get("turn_credit_wrapper_proof") != audit.get("wrapper_proof"):
                raise ValueError("row/audit wrapper proof mismatch")
            validated = _validate_audit(audit)
            proof = str(audit["wrapper_proof"])
            wrapper_proofs.add(proof)
            trajectory_uid = str(audit["trajectory_uid"])
            if trajectory_uid in trajectory_uids:
                raise ValueError(f"duplicate trajectory UID: {trajectory_uid}")
            trajectory_uids.add(trajectory_uid)
            prompt_counts[str(audit["prompt_uid"])] += 1
            rows_in_step += 1
            trajectories += 1
            policy_span_tokens += validated["policy_span_tokens"]
            tool_span_tokens += validated["tool_span_tokens"]
            turn_rows = audit["turns"]
            turns += len(turn_rows)
            turns_per_trajectory[len(turn_rows)] += 1
            for turn in turn_rows:
                adoption_statuses[str(turn["adoption_status"])] += 1
                correction = float(turn["correction"])
                if not math.isfinite(correction):
                    raise ValueError("non-finite turn correction")
                corrections["positive" if correction > 0 else "negative" if correction < 0 else "zero"] += 1
            parser_error_trajectories += int(float(row.get("tool_parse_errors", 0)) > 0)
            invalid_trajectories += int(float(row.get("invalid", 0)) > 0)
            truncated_trajectories += int(float(row.get("truncated", 0)) > 0)
        if rows_in_step != 512:
            raise ValueError(f"step {path.stem} has {rows_in_step} rows instead of 512")
        if len(prompt_counts) != 64 or set(prompt_counts.values()) != {8}:
            raise ValueError(f"step {path.stem} does not contain 64 complete groups of 8")
        prompt_counts_by_step[int(path.stem)] = prompt_counts

    if wrapper_proofs != {str(active["proof_id"])}:
        raise ValueError("rollout wrapper proof does not match the Ray-driver activation line")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    wrapper_file = run_dir / "wrapper_installation.json"
    wrapper_restore_artifact_valid = False
    if wrapper_file.stat().st_size:
        wrapper_state = json.loads(wrapper_file.read_text(encoding="utf-8"))
        wrapper_restore_artifact_valid = bool(wrapper_state.get("installed") and wrapper_state.get("restored"))

    report = {
        "schema_version": "toolcredit_partial_smoke_audit_v1",
        "run_name": run_dir.name,
        "classification": "canonical_v2_user_stopped_after_step_3",
        "normal_five_step_smoke_completed": False,
        "completed_steps": completed_steps,
        "launcher_terminal_status": status,
        "trajectory_count": trajectories,
        "turn_count": turns,
        "unique_trajectory_uid_count": len(trajectory_uids),
        "complete_groups_per_step": {str(step): len(groups) for step, groups in prompt_counts_by_step.items()},
        "rollouts_per_group": 8,
        "turns_per_trajectory": dict(sorted(turns_per_trajectory.items())),
        "adoption_status_counts": dict(sorted(adoption_statuses.items())),
        "correction_sign_counts": dict(sorted(corrections.items())),
        "policy_span_tokens": policy_span_tokens,
        "tool_span_tokens": tool_span_tokens,
        "formula_span_mask_uid_mismatches": 0,
        "parser_error_trajectories": parser_error_trajectories,
        "invalid_trajectories": invalid_trajectories,
        "truncated_trajectories": truncated_trajectories,
        "wrapper_active_proof_id": active["proof_id"],
        "wrapper_active_in_all_rollouts": True,
        "wrapper_restore_artifact_valid": wrapper_restore_artifact_valid,
        "wrapper_restore_artifact_bytes": wrapper_file.stat().st_size,
        "step_4_rollout_present": (train_dir / "4.jsonl").exists(),
        "checkpoint_present": (run_dir / "checkpoints").exists(),
        "step_3_validation_present": (run_dir / "predictions/validation/3.jsonl").exists(),
        "deviations": [
            "User shortened the authorized 5-step smoke to 3 completed actor updates while it was running.",
            "The step-5 checkpoint and final validation cadence were therefore not reached.",
            "Ctrl-C truncated the non-atomic wrapper restoration proof file; process termination removes the runtime patch, but file-based restoration proof is unavailable.",
        ],
    }
    if report["step_4_rollout_present"]:
        raise ValueError("step-4 rollout exists despite the requested stop-after-3 boundary")
    output = run_dir / "analysis/e5_partial_smoke_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_partial(args.run_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
