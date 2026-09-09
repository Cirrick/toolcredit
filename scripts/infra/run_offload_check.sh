#!/usr/bin/env bash
# FSDP-offload-off infrastructure check (HANDOFF "下一步" item 5).
# Launch with: tmux new-session -d -s offload-check 'bash scripts/infra/run_offload_check.sh'
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-e3_offload_check_${STAMP}}"
RUN_DIR="$PROJECT_DIR/rl/runs/$RUN_NAME"
[[ -e "$RUN_DIR" ]] && { echo "refusing to reuse run directory: $RUN_DIR" >&2; exit 1; }

cd "$PROJECT_DIR"
mkdir -p rl/runs

# Container-memory sampler (read-only) starts once the launcher has created the run dir.
( while [[ ! -d "$RUN_DIR" ]]; do sleep 5; done
  mkdir -p "$RUN_DIR/analysis"
  exec bash scripts/m9/memwatch.sh "$RUN_DIR" 30 ) &
MEMWATCH_PID=$!
trap 'kill "$MEMWATCH_PID" 2>/dev/null || true' EXIT

set +e
conda run --no-capture-output -n toolcredit \
  python -m rl.launch.e3_offload_check --run-name "$RUN_NAME" \
  2>&1 | tee "rl/runs/${RUN_NAME}.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [[ -d "$RUN_DIR" ]]; then
  conda run --no-capture-output -n toolcredit \
    python -m rl.monitor_run --run-dir "$RUN_DIR" || true
fi
echo "offload check exit=$TRAIN_EXIT run_dir=$RUN_DIR"
exit "$TRAIN_EXIT"
