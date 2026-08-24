"""Validate the completed 200-step E5 run and emit a factual status artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rl.validate_e5_canonical_smoke import (
    PROJECT_ROOT,
    validate_checkpoint,
    validate_resolved_config,
    validate_rollouts,
    validate_validation_files,
)

FORMAL_VALIDATION_STEPS = [0, 25, 50, 75, 100, 125, 150, 175, 200]


def validate_formal(run_dir: Path) -> dict:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"formal E5 run is not completed: {status}")
    config = validate_resolved_config(run_dir, smoke=False)
    validation = validate_validation_files(run_dir, FORMAL_VALIDATION_STEPS)
    rollouts = validate_rollouts(run_dir, 200, recovery_aware=True)
    checkpoint = validate_checkpoint(run_dir, 200)
    recovery_dir = run_dir / "recovery"
    recovery_events = sorted(str(path.relative_to(run_dir)) for path in recovery_dir.glob("resume_from_*") if path.is_dir()) if recovery_dir.exists() else []
    disk = shutil.disk_usage(PROJECT_ROOT)
    report = {
        "schema_version": "toolcredit_formal_completion_gate_v1",
        "run_name": run_dir.name,
        "status": "completed",
        "effective_actor_updates": 200,
        "resolved_config": config,
        "fixed_panel_pass_at_1": validation["pass_at_1"],
        "rollouts": rollouts,
        "checkpoint": checkpoint,
        "recovery_events": recovery_events,
        "disk_free_gib": round(disk.free / 2**30, 2),
        "interpretation_performed": False,
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    (analysis_dir / "formal_completion_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = validate_formal(args.run_dir.resolve())
    print(json.dumps({"run_name": report["run_name"], "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
