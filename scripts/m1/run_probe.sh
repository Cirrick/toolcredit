#!/usr/bin/env bash
# M1: tool-gain probe runner. Full run is a long task: run inside tmux, e.g.
#   tmux new-session -d -s probe 'bash scripts/m1/run_probe.sh'
# Dry-run (fast, foreground OK): bash scripts/m1/run_probe.sh --limit 10
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=~/.conda/envs/toolcredit/bin/python
LOG_DIR="$PROJECT_DIR/scripts/m1/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"
exec "$PY" data/tool_gain_probe.py "$@" 2>&1 | tee "$LOG_DIR/probe_$(date +%Y%m%d_%H%M%S).log"
