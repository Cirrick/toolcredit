"""Run canonical E5 smoke -> fail-closed gate -> independent formal run."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "rl/runs"
SFT = ROOT / "sft/checkpoints/qwen3-1.7b-sft"
FORMAL_DISK_FLOOR_GIB = 135


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def snapshot(run_dir: Path) -> dict[str, Any]:
    train = run_dir / "predictions/train"
    validation = run_dir / "predictions/validation"
    checkpoints = run_dir / "checkpoints"
    train_steps = sorted(int(path.stem) for path in train.glob("*.jsonl")) if train.exists() else []
    validation_steps = sorted(int(path.stem) for path in validation.glob("*.jsonl")) if validation.exists() else []
    checkpoint_steps = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in checkpoints.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    ) if checkpoints.exists() else []
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"state": "starting"}
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_dir.name,
        "status": status.get("state"),
        "latest_train_step": max(train_steps, default=0),
        "validation_steps": validation_steps,
        "checkpoint_steps": checkpoint_steps,
        "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
    }


def run_monitored(command: list[str], env: dict[str, str], run_dir: Path, progress_path: Path, *, interval: int, disk_floor_gib: int) -> int:
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    previous: tuple[Any, ...] | None = None
    while True:
        try:
            return_code = process.wait(timeout=interval)
            state = snapshot(run_dir)
            state["process_return_code"] = return_code
            append_jsonl(progress_path, state)
            return return_code
        except subprocess.TimeoutExpired:
            state = snapshot(run_dir)
            signature = (
                state["status"],
                state["latest_train_step"],
                tuple(state["validation_steps"]),
                tuple(state["checkpoint_steps"]),
            )
            if signature != previous:
                append_jsonl(progress_path, state)
                previous = signature
            if state["disk_free_gib"] < disk_floor_gib:
                append_jsonl(
                    progress_path,
                    {**state, "event": "disk_gate_failed", "required_gib": disk_floor_gib},
                )
                process.send_signal(signal.SIGINT)
                return process.wait()


def main() -> int:
    orchestration_name = f"m6_overnight_{utc_stamp()}"
    orchestration_dir = RUN_ROOT / orchestration_name
    orchestration_dir.mkdir(parents=False)
    state_path = orchestration_dir / "state.json"
    progress_path = orchestration_dir / "progress.jsonl"
    smoke_name = f"e5_turn_credit_smoke_v2_{utc_stamp()}"
    smoke_dir = RUN_ROOT / smoke_name
    if smoke_dir.exists():
        raise FileExistsError(smoke_dir)
    launch_ledger = orchestration_dir / "smoke_fresh_launch.json"
    write_json(
        launch_ledger,
        {
            "run_name": smoke_name,
            "resumed": False,
            "checkpoint": str(SFT),
            "run_dir_absent_before_launch": True,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    state: dict[str, Any] = {
        "orchestration": orchestration_name,
        "state": "smoke_running",
        "smoke_run_name": smoke_name,
        "formal_run_name": None,
        "formal_authorized_by_user": True,
    }
    write_json(state_path, state)
    env = dict(os.environ)
    env["RUN_NAME"] = smoke_name
    smoke_rc = run_monitored(
        ["bash", "scripts/m6/run_e5.sh", "smoke"],
        env,
        smoke_dir,
        progress_path,
        interval=120,
        disk_floor_gib=55,
    )
    if smoke_rc != 0:
        state.update(state="smoke_process_failed", smoke_return_code=smoke_rc)
        write_json(state_path, state)
        return smoke_rc or 1

    gate_command = [
        "conda", "run", "--no-capture-output", "-n", "toolcredit", "python", "-m",
        "rl.validate_e5_canonical_smoke", str(smoke_dir), "--launch-ledger", str(launch_ledger),
    ]
    gate = subprocess.run(gate_command, cwd=ROOT)
    if gate.returncode != 0:
        state.update(state="canonical_smoke_gate_failed", smoke_gate_return_code=gate.returncode)
        write_json(state_path, state)
        return gate.returncode

    state["state"] = "canonical_smoke_passed"
    write_json(state_path, state)
    formal_name = f"e5_turn_credit_{utc_stamp()}"
    formal_dir = RUN_ROOT / formal_name
    if formal_dir.exists():
        raise FileExistsError(formal_dir)
    state.update(state="formal_running", formal_run_name=formal_name)
    write_json(state_path, state)
    env["RUN_NAME"] = formal_name
    formal_rc = run_monitored(
        ["bash", "scripts/m6/run_e5.sh", "full"],
        env,
        formal_dir,
        progress_path,
        interval=600,
        disk_floor_gib=FORMAL_DISK_FLOOR_GIB,
    )
    if formal_rc != 0:
        state.update(state="formal_process_failed", formal_return_code=formal_rc)
        write_json(state_path, state)
        return formal_rc or 1

    completion = subprocess.run(
        [
            "conda", "run", "--no-capture-output", "-n", "toolcredit", "python", "-m",
            "rl.validate_e5_formal_completion", str(formal_dir),
        ],
        cwd=ROOT,
    )
    if completion.returncode != 0:
        state.update(state="formal_completion_gate_failed", completion_return_code=completion.returncode)
        write_json(state_path, state)
        return completion.returncode
    state.update(state="formal_completed_and_validated", completed_at=datetime.now(timezone.utc).isoformat())
    write_json(state_path, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
