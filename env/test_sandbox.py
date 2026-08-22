"""Sandbox test suite (M2, PLAN §6.1). Run: python -m pytest env/test_sandbox.py -v"""

import os
import tempfile

import pytest

from env.sandbox import MAX_OUTPUT_CHARS, NETNS_PREFIX, prepare_tool_code, run_python

# ---------- basic contract ----------


def test_ok_basic() -> None:
    r = run_python("print(1 + 1)")
    assert r["status"] == "ok"
    assert r["stdout"] == "2\n"
    assert r["wall_time"] > 0


def test_error_captures_traceback() -> None:
    r = run_python("1 / 0")
    assert r["status"] == "error"
    assert "ZeroDivisionError" in r["stderr"]


def test_timeout_infinite_loop() -> None:
    r = run_python("while True: pass", timeout=1.0)
    assert r["status"] == "timeout"
    assert r["wall_time"] < 3.0


def test_output_truncated() -> None:
    r = run_python(f"print('x' * {MAX_OUTPUT_CHARS * 50})")
    assert r["status"] == "ok"
    assert len(r["stdout"]) <= MAX_OUTPUT_CHARS + 100
    assert "truncated" in r["stdout"]


# ---------- hostile payloads (PLAN-mandated) ----------


def test_fork_bomb_contained() -> None:
    code = "import os\nwhile True:\n    try: os.fork()\n    except OSError: pass"
    r = run_python(code, timeout=3.0)
    assert r["status"] in ("timeout", "error")  # nproc limit + group kill; must return
    assert r["wall_time"] < 8.0
    # and this test process is still alive to assert anything at all


def test_memory_bomb_limited() -> None:
    r = run_python("x = bytearray(2 << 30)\nprint('allocated')")  # 2 GB vs 1 GB limit
    assert r["status"] == "error"
    assert "MemoryError" in r["stderr"]
    assert r["wall_time"] < 5.0


def test_delete_root_owned_file_fails() -> None:
    r = run_python("import os\nos.remove('/etc/hostname')")
    assert r["status"] == "error"
    # Depending on the pod mount, the kernel reports either EACCES or EROFS.
    assert "PermissionError" in r["stderr"] or "Read-only file system" in r["stderr"]
    assert os.path.exists("/etc/hostname")


def test_huge_file_write_limited() -> None:
    code = "with open('big.bin', 'wb') as f:\n    f.write(b'0' * (64 << 20))\nprint('wrote')"
    r = run_python(code, timeout=4.0)
    assert r["status"] in ("error", "timeout")  # RLIMIT_FSIZE (16 MB)


# ---------- isolation properties ----------


def test_cwd_is_ephemeral_and_writes_vanish() -> None:
    r = run_python("import os\nopen('leak.txt', 'w').write('x')\nprint(os.getcwd())")
    assert r["status"] == "ok"
    workdir = r["stdout"].strip()
    assert "toolcredit_sandbox_" in workdir
    assert not os.path.exists(workdir)  # ephemeral dir removed after the run
    assert not os.path.exists("leak.txt")  # nothing lands in the caller's cwd


def test_home_env_isolated() -> None:
    r = run_python("import os\nprint(os.path.expanduser('~'))")
    assert r["status"] == "ok"
    assert "toolcredit_sandbox_" in r["stdout"]


@pytest.mark.skipif(not NETNS_PREFIX, reason="unprivileged netns unavailable on this platform")
def test_network_blocked() -> None:
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(2)\n"
        "s.connect(('1.1.1.1', 80))\n"
        "print('connected')"
    )
    r = run_python(code, timeout=6.0)
    assert r["status"] == "error"
    assert "connected" not in r["stdout"]


def test_user_file_delete_is_a_documented_gap() -> None:
    """Same-uid absolute-path deletes are NOT blocked (no mount ns) — documented in
    sandbox.py docstring and environment.md. This test pins the current behavior so
    we notice if the isolation story ever changes."""
    fd, path = tempfile.mkstemp(prefix="sandbox_gap_probe_")
    os.close(fd)
    try:
        r = run_python(f"import os\nos.remove({path!r})\nprint('removed')")
        assert r["status"] == "ok"
        assert not os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------- prepare_tool_code (tool-side preprocessing) ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("x = 2+3\nx", "x = 2 + 3\nprint(x)"),  # bare-name last line wrapped
        ("print(42)", "print(42)"),  # already printed: untouched
        ("y = 5", "y = 5"),  # assignment last line: untouched (no broken print(y = 5))
        ("for i in range(3):\n    print(i)", "for i in range(3):\n    print(i)"),  # block: untouched
        ("```python\na=1\na\n```", "a = 1\nprint(a)"),  # ```python fence
        ("```py\nb=2\nb\n```", "b = 2\nprint(b)"),  # ```py fence
        ("x = ", "x = "),  # syntax error: passthrough for the sandbox to report
    ],
)
def test_prepare_tool_code(raw: str, expected: str) -> None:
    assert prepare_tool_code(raw) == expected


def test_prepare_tool_code_executes() -> None:
    r = run_python(prepare_tool_code("import math\nresult = math.sqrt(16)\nresult"))
    assert r["status"] == "ok"
    assert r["stdout"] == "4.0\n"
