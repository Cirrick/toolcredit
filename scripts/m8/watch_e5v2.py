"""Low-frequency watcher for the E5-v2 formal run (plans/M8.md §8 steps 4–5).

Every ``--interval-seconds`` it snapshots status/steps/checkpoints/disk and appends changes
to ``recovery_monitor.jsonl``.  It enforces three fail-closed stops by sending SIGINT to the
driver (veRL then resumes only from the tracker checkpoint):
  * disk below the floor,
  * exposure over steps 1–25 below 8% (plan §4 preregistered floor),
  * behavior pause thresholds at steps 50 and 100 (mean calls > 2.0 or 4-call > 15%).
On normal completion it runs the formal completion gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GATE_STEPS = (50, 100)
EXPOSURE_STEP = 25


def _snapshot(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"state": "starting"}
    train = run_dir / "predictions/train"
    validation = run_dir / "predictions/validation"
    checkpoints = run_dir / "checkpoints"
    train_steps = sorted(int(p.stem) for p in train.glob("*.jsonl")) if train.exists() else []
    validation_steps = sorted(int(p.stem) for p in validation.glob("*.jsonl")) if validation.exists() else []
    checkpoint_steps = (
        sorted(
            int(p.name.removeprefix("global_step_"))
            for p in checkpoints.glob("global_step_*")
            if p.is_dir() and p.name.removeprefix("global_step_").isdigit()
        )
        if checkpoints.exists()
        else []
    )
    tracker = checkpoints / "latest_checkpointed_iteration.txt"
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "state": status.get("state"),
        "driver_pid": status.get("pid"),
        "latest_train_step": max(train_steps, default=0),
        "validation_steps": validation_steps,
        "checkpoint_steps": checkpoint_steps,
        "tracker": tracker.read_text(encoding="utf-8").strip() if tracker.exists() else None,
        "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
    }


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _stop_driver(snapshot: dict[str, Any]) -> None:
    pid = snapshot.get("driver_pid")
    if isinstance(pid, int) and pid > 1:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass


def _validator(run_dir: Path, *args: str) -> dict[str, Any]:
    result = subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "toolcredit", "python", "-m", "rl.validate_e5v2_run",
         args[0], str(run_dir), *args[1:]],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"error": result.stderr[-4000:], "return_code": result.returncode}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw_stdout": result.stdout[-4000:], "return_code": 0}


def watch(run_dir: Path, *, interval_seconds: int, disk_floor_gib: int) -> int:
    output = run_dir / "recovery_monitor.jsonl"
    previous: tuple[Any, ...] | None = None
    exposure_checked = False
    gates_checked: set[int] = set()
    while True:
        time.sleep(interval_seconds)
        snapshot = _snapshot(run_dir)
        signature = (snapshot["state"], snapshot["latest_train_step"], tuple(snapshot["validation_steps"]), tuple(snapshot["checkpoint_steps"]))
        if signature != previous:
            _append(output, snapshot)
            previous = signature
        if float(snapshot["disk_free_gib"]) < disk_floor_gib:
            _append(output, {**snapshot, "event": "disk_gate_failed", "floor_gib": disk_floor_gib})
            _stop_driver(snapshot)
            return 2
        latest = int(snapshot["latest_train_step"])
        if not exposure_checked and latest >= EXPOSURE_STEP:
            report = _validator(run_dir, "exposure", "--start", "1", "--end", str(EXPOSURE_STEP))
            exposure_checked = True
            _append(output, {**snapshot, "event": "exposure_check", "report": report})
            if report.get("error") or not report.get("exposure_above_floor", False):
                _append(output, {**snapshot, "event": "exposure_gate_failed_pausing"})
                _stop_driver(snapshot)
                return 4
        for gate_step in GATE_STEPS:
            if gate_step in gates_checked or latest < gate_step:
                continue
            report = _validator(run_dir, "behavior", "--upto", str(gate_step))
            gates_checked.add(gate_step)
            _append(output, {**snapshot, "event": f"behavior_gate_step_{gate_step}", "report": report})
            if report.get("error") or report.get("pause_required", True):
                _append(output, {**snapshot, "event": f"behavior_gate_step_{gate_step}_failed_pausing"})
                _stop_driver(snapshot)
                return 3
        if snapshot["state"] == "failed":
            _append(output, {**snapshot, "event": "driver_failed"})
            return 1
        if snapshot["state"] == "completed":
            report = _validator(run_dir, "formal")
            _append(output, {**_snapshot(run_dir), "event": "formal_completion_gate", "report": report})
            return 0 if not report.get("error") else 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--disk-floor-gib", type=int, default=60)
    args = parser.parse_args()
    if args.interval_seconds < 60:
        raise ValueError("watch interval must remain lightweight (at least 60 seconds)")
    return watch(args.run_dir.resolve(), interval_seconds=args.interval_seconds, disk_floor_gib=args.disk_floor_gib)


if __name__ == "__main__":
    raise SystemExit(main())
