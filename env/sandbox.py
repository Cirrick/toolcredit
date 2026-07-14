"""Python code sandbox executor — hardened version (M2, PLAN §6.1).

Guarantees (tested in test_sandbox.py):
  - wall-clock timeout with whole-process-group SIGKILL (survives fork bombs);
  - resource limits in the child: address space 1 GB, file size 16 MB, CPU time,
    process count (dynamic headroom over the user's current count);
  - network isolation via `unshare --map-root-user -n` when the platform allows
    unprivileged user namespaces (probed once at import; GH200 pod: allowed);
  - ephemeral tmp working directory (relative writes vanish afterwards), minimal env;
  - stdout/stderr captured, truncated to 2000 chars.

Documented gaps (industrial setups use containers/gVisor for these):
  - absolute-path reads/writes/deletes on files owned by the invoking user are NOT
    blocked (same uid, no mount namespace remount) — only root-owned paths fail;
  - no syscall filtering (seccomp).

Interface `run_python` is unchanged from the M1 minimal version.
`prepare_tool_code` (fence strip + AST auto-print) lives here as the single source
for every tool wrapper (probe, M4 training tool).
"""

import ast
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import TypedDict

MAX_OUTPUT_CHARS = 2000
DEFAULT_TIMEOUT_S = 5.0
MEMORY_LIMIT_BYTES = 1 << 30  # 1 GB
FILE_SIZE_LIMIT_BYTES = 16 << 20  # 16 MB
NPROC_HEADROOM = 128  # RLIMIT_NPROC counts per-uid across the system; leave room over current
CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


class SandboxResult(TypedDict):
    status: str  # "ok" | "error" | "timeout"
    stdout: str
    stderr: str
    wall_time: float


def _probe_netns_prefix() -> list[str]:
    """Return the unshare prefix if unprivileged netns isolation is available."""
    try:
        ok = subprocess.run(
            ["unshare", "--map-root-user", "-n", "true"], capture_output=True, timeout=10
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return ["unshare", "--map-root-user", "-n"] if ok else []


NETNS_PREFIX = _probe_netns_prefix()


def _current_nproc() -> int:
    return len([p for p in os.listdir("/proc") if p.isdigit()])


def _make_preexec(timeout: float):
    nproc_limit = _current_nproc() + NPROC_HEADROOM
    cpu_limit = max(1, int(timeout) + 1)

    def preexec() -> None:
        os.setsid()  # own process group, so the parent can kill the whole tree
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE_LIMIT_BYTES, FILE_SIZE_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc_limit, nproc_limit))

    return preexec


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[truncated at {MAX_OUTPUT_CHARS} chars]"


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT_S) -> SandboxResult:
    """Execute python code in an isolated subprocess and capture its output."""
    start = time.monotonic()
    workdir = tempfile.mkdtemp(prefix="toolcredit_sandbox_")
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": workdir,
        "TMPDIR": workdir,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    try:
        proc = subprocess.Popen(
            NETNS_PREFIX + [sys.executable, "-c", code],
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_make_preexec(timeout),
            text=True,
            errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            status = "ok" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            status = "timeout"
            stderr = (stderr or "") + f"\nTimeoutError: execution exceeded {timeout}s"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return SandboxResult(
        status=status,
        stdout=_truncate(stdout or ""),
        stderr=_truncate((stderr or "").strip()),
        wall_time=round(time.monotonic() - start, 3),
    )


def prepare_tool_code(raw: str) -> str:
    """Fence strip + REPL-style auto-print for code_interpreter tool calls.

    Robust version of the official verl-tutorial SandboxTool heuristic: only wraps
    the last statement in print() when it is a bare expression (the official
    line-based version breaks on indented/assignment last lines — see plans/M1.md).
    """
    matches = CODE_FENCE_RE.findall(raw)
    code = matches[0].strip() if matches else raw
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # let the sandbox report the error verbatim
    last = tree.body[-1] if tree.body else None
    already_print = (
        isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Call)
        and getattr(last.value.func, "id", "") == "print"
    )
    if isinstance(last, ast.Expr) and not already_print:
        tree.body[-1] = ast.Expr(
            ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[last.value], keywords=[])
        )
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    return code


if __name__ == "__main__":
    print("netns isolation:", "on" if NETNS_PREFIX else "OFF (unshare unavailable)")
    print(run_python("print(1 + 1)"))
    print(run_python("1 / 0"))
    print(run_python("while True: pass", timeout=1.0))
