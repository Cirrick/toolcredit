"""Lightweight 30-minute watcher for an already running E5 formal resume."""

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


def _snapshot(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    train_steps = sorted(int(path.stem) for path in (run_dir / "predictions/train").glob("*.jsonl"))
    validation_steps = sorted(
        int(path.stem) for path in (run_dir / "predictions/validation").glob("*.jsonl")
    )
    checkpoint_steps = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in (run_dir / "checkpoints").glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    )
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "state": status.get("state"),
        "driver_pid": status.get("pid"),
        "latest_train_step": max(train_steps, default=0),
        "validation_steps": validation_steps,
        "checkpoint_steps": checkpoint_steps,
        "tracker": (run_dir / "checkpoints/latest_checkpointed_iteration.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
    }


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _stop_driver(snapshot: dict[str, Any]) -> None:
    pid = snapshot.get("driver_pid")
    if isinstance(pid, int) and pid > 1:
        os.kill(pid, signal.SIGINT)


def watch(run_dir: Path, *, interval_seconds: int, disk_floor_gib: int) -> int:
    output = run_dir / "recovery_monitor.jsonl"
    previous: tuple[Any, ...] | None = None
    while True:
        time.sleep(interval_seconds)
        snapshot = _snapshot(run_dir)
        signature = (
            snapshot["state"],
            tuple(snapshot["validation_steps"]),
            tuple(snapshot["checkpoint_steps"]),
        )
        if signature != previous:
            _append(output, snapshot)
            previous = signature
        if float(snapshot["disk_free_gib"]) < disk_floor_gib:
            _append(output, {**snapshot, "event": "disk_gate_failed", "floor_gib": disk_floor_gib})
            _stop_driver(snapshot)
            return 2
        if snapshot["state"] == "failed":
            return 1
        if snapshot["state"] == "completed":
            result = subprocess.run(
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "toolcredit",
                    "python",
                    "-m",
                    "rl.validate_e5_formal_completion",
                    str(run_dir),
                ],
                cwd=ROOT,
            )
            _append(
                output,
                {**_snapshot(run_dir), "event": "formal_completion_gate", "return_code": result.returncode},
            )
            return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--disk-floor-gib", type=int, default=133)
    args = parser.parse_args()
    if args.interval_seconds < 60:
        raise ValueError("watch interval must remain lightweight (at least 60 seconds)")
    return watch(
        args.run_dir.resolve(),
        interval_seconds=args.interval_seconds,
        disk_floor_gib=args.disk_floor_gib,
    )


if __name__ == "__main__":
    raise SystemExit(main())
