"""Python code sandbox executor — M1 minimal version (v0).

Subprocess + timeout + output truncation only. M2 hardens this (no network,
memory limit, no file writes) and adds the full test suite (PLAN §6.1);
callers must depend only on `run_python`'s interface, which will not change.
"""

import subprocess
import sys
import time
from typing import TypedDict

MAX_OUTPUT_CHARS = 2000
DEFAULT_TIMEOUT_S = 5.0


class SandboxResult(TypedDict):
    status: str  # "ok" | "error" | "timeout"
    stdout: str
    stderr: str
    wall_time: float


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[truncated at {MAX_OUTPUT_CHARS} chars]"


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT_S) -> SandboxResult:
    """Execute python code in a subprocess and capture its output."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "ok" if proc.returncode == 0 else "error"
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = f"TimeoutError: execution exceeded {timeout}s"
    return SandboxResult(
        status=status,
        stdout=_truncate(stdout),
        stderr=_truncate(stderr),
        wall_time=round(time.monotonic() - start, 3),
    )


if __name__ == "__main__":
    print(run_python("print(1 + 1)"))
    print(run_python("1 / 0"))
    print(run_python("while True: pass", timeout=1.0))
