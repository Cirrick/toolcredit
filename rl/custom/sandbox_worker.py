"""Lightweight spawned-process entry point for the M4 sandbox tool."""

from __future__ import annotations

import json
import sys
from typing import Any

from env.sandbox import run_python


def run_sandbox(code: str, timeout: float) -> dict[str, Any]:
    """Call the canonical sandbox from a single-threaded helper process."""
    return dict(run_python(code, timeout))


def main() -> None:
    request = json.load(sys.stdin)
    result = run_sandbox(str(request["code"]), float(request["timeout"]))
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
