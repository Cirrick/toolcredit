#!/usr/bin/env bash
# Launch with: tmux new-session -d -s m9-notool-watch 'bash scripts/m9/watch_e3notool.sh rl/runs/<run_name>'
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${1:?run dir required}"
cd "$PROJECT_DIR"
conda run --no-capture-output -n toolcredit python scripts/m9/watch_e3notool.py "$RUN_DIR" "${@:2}" \
  2>&1 | tee -a "$RUN_DIR/watcher.log"
