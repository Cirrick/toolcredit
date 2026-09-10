#!/usr/bin/env bash
# Launch with: tmux new-session -d -s m10-e7-watch 'bash scripts/m10/watch_e7.sh rl/runs/<run_name>'
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${1:?run dir required}"
cd "$PROJECT_DIR"
/opt/conda/bin/conda run --no-capture-output -n toolcredit python scripts/m10/watch_e7.py "$RUN_DIR" "${@:2}" \
  2>&1 | tee -a "$RUN_DIR/watcher.log"
