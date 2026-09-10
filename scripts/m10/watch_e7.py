"""Low-frequency watcher for E7 formal segments (plans/M10.md §4).

Every ``--interval-seconds`` it snapshots status / ledger / checkpoints / disk / memwatch and
appends changes to ``recovery_monitor.jsonl``.  Three plan §4 gates are enforced by writing the
trainer's stop request (``e7_stop_request.json``): the run then ends at the next checkpoint
with a normal save + validation, and the launcher records ``segment_completed``.

* cgroup ``anon`` above 110 GiB (memwatch v2 sample)      -> early segment end;
* ten consecutive effective steps at the 4-batch cap     -> pause and report;
* disk below the floor                                   -> SIGINT (as the M9 watcher did).

An underfilled step or any trainer exception surfaces as ``status.state == "failed"``.
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
ANON_LIMIT_GIB = 110.0
STOP_REQUEST = "e7_stop_request.json"
STREAK_LIMIT = 10


def _tail_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                last = line
    try:
        return json.loads(last.decode("utf-8")) if last else None
    except json.JSONDecodeError:
        return None


def _snapshot(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"state": "starting"}
    checkpoints = run_dir / "checkpoints"
    tracker = checkpoints / "latest_checkpointed_iteration.txt"
    ledger_row = _tail_json(run_dir / "e7_ledger.jsonl") or {}
    mem = _tail_json(run_dir / "analysis/container_memory.jsonl") or {}
    anon = mem.get("anon")
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "state": status.get("state"),
        "driver_pid": status.get("pid"),
        "stop_at_step": status.get("stop_at_step"),
        "ledger_step": ledger_row.get("effective_step", 0),
        "ledger_batches": ledger_row.get("generation_batches_used"),
        "cumulative_trajectories": ledger_row.get("cumulative_trajectories"),
        "equal_compute_checkpoint": ledger_row.get("equal_compute_checkpoint"),
        "checkpoint_steps": sorted(
            int(p.name.removeprefix("global_step_")) for p in checkpoints.glob("global_step_*") if p.is_dir()
        ) if checkpoints.exists() else [],
        "equal_compute_dirs": sorted(p.name for p in checkpoints.glob("equal_compute_step_*")) if checkpoints.exists() else [],
        "tracker": tracker.read_text(encoding="utf-8").strip() if tracker.exists() else None,
        "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
        "anon_gib": None if anon in (None, "null") else round(float(anon) / 2**30, 2),
        "memwatch_at": mem.get("at"),
        "stop_request_present": (run_dir / STOP_REQUEST).is_file(),
    }


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _request_stop(run_dir: Path, reason: str, snapshot: dict[str, Any]) -> None:
    path = run_dir / STOP_REQUEST
    if path.exists():
        return
    path.write_text(
        json.dumps({"reason": reason, "requested_at": datetime.now(timezone.utc).isoformat(), "snapshot": snapshot}, indent=2) + "\n",
        encoding="utf-8",
    )


def _stop_driver(snapshot: dict[str, Any]) -> None:
    pid = snapshot.get("driver_pid")
    if isinstance(pid, int) and pid > 1:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass


def _streak(run_dir: Path) -> int:
    result = subprocess.run(
        ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "toolcredit", "python", "-m", "rl.validate_e7_run", "streak", str(run_dir)],
        cwd=ROOT, capture_output=True, text=True,
    )
    try:
        return int(json.loads(result.stdout)["current_streak"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0


def watch(run_dir: Path, *, interval_seconds: int, disk_floor_gib: int) -> int:
    output = run_dir / "recovery_monitor.jsonl"
    previous: tuple[Any, ...] | None = None
    while True:
        time.sleep(interval_seconds)
        snapshot = _snapshot(run_dir)
        signature = (snapshot["state"], snapshot["ledger_step"], tuple(snapshot["checkpoint_steps"]), snapshot["anon_gib"])
        if signature != previous:
            _append(output, snapshot)
            previous = signature
        if float(snapshot["disk_free_gib"]) < disk_floor_gib:
            _append(output, {**snapshot, "event": "disk_gate_failed", "floor_gib": disk_floor_gib})
            _stop_driver(snapshot)
            return 2
        if snapshot["anon_gib"] is not None and snapshot["anon_gib"] > ANON_LIMIT_GIB and not snapshot["stop_request_present"]:
            _request_stop(run_dir, f"cgroup anon {snapshot['anon_gib']} GiB > {ANON_LIMIT_GIB}", snapshot)
            _append(output, {**snapshot, "event": "anon_limit_stop_requested"})
        if snapshot["ledger_batches"] == 4 and not snapshot["stop_request_present"]:
            streak = _streak(run_dir)
            if streak >= STREAK_LIMIT:
                _request_stop(run_dir, f"{streak} consecutive steps at 4 generation batches", snapshot)
                _append(output, {**snapshot, "event": "full_batch_streak_stop_requested", "streak": streak})
        if snapshot["state"] == "failed":
            _append(output, {**snapshot, "event": "driver_failed"})
            return 1
        if snapshot["state"] in ("completed", "segment_completed"):
            _append(output, {**snapshot, "event": f"driver_{snapshot['state']}"})
            return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--disk-floor-gib", type=int, default=120)
    args = parser.parse_args()
    if args.interval_seconds < 60:
        raise ValueError("watch interval must remain lightweight (at least 60 seconds)")
    return watch(args.run_dir.resolve(), interval_seconds=args.interval_seconds, disk_floor_gib=args.disk_floor_gib)


if __name__ == "__main__":
    raise SystemExit(main())
