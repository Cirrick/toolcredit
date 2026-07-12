#!/usr/bin/env bash
# M0 official-example run. Long task: launch inside tmux, never in the foreground
# of an interactive session, e.g.:
#   tmux new-session -d -s m0 'bash scripts/m0/run.sh'
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=~/.conda/envs/toolcredit/bin/python
LOG_DIR="$PROJECT_DIR/scripts/m0/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR/scripts/m0"
exec "$PY" run_official_example.py "$@" 2>&1 | tee "$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
