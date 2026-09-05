#!/usr/bin/env bash
# Launch with: tmux new-session -d -s m8-e5v2-watch 'bash scripts/m8/watch_e5v2.sh rl/runs/<run_name>'
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${1:?run dir required}"
cd "$PROJECT_DIR"
conda run --no-capture-output -n toolcredit python scripts/m8/watch_e5v2.py "$RUN_DIR" "${@:2}" \
  2>&1 | tee -a "$RUN_DIR/watcher.log"
