#!/usr/bin/env bash
# Launch with: tmux new-session -d -s m6-e5-smoke 'bash scripts/m6/run_e5.sh smoke'
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-smoke}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

case "$MODE" in
  smoke) MODE_ARG="--smoke"; PREFIX="e5_turn_credit_smoke" ;;
  full) MODE_ARG=""; PREFIX="e5_turn_credit" ;;
  *) echo "mode must be 'smoke' or 'full'" >&2; exit 2 ;;
esac

RUN_NAME="${RUN_NAME:-${PREFIX}_${STAMP}}"
RUN_DIR="$PROJECT_DIR/rl/runs/$RUN_NAME"
RESUME_ARG=""
TEE_MODE=""
if [[ "${RESUME:-0}" == "1" ]]; then
  [[ -d "$RUN_DIR" ]] || { echo "resume run directory does not exist: $RUN_DIR" >&2; exit 1; }
  RESUME_ARG="--resume"
  TEE_MODE="--append"
elif [[ -e "$RUN_DIR" ]]; then
  echo "refusing to reuse run directory: $RUN_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/rl/runs"
set +e
conda run --no-capture-output -n toolcredit \
  python -m rl.launch.e5_turn_credit \
  --run-name "$RUN_NAME" $MODE_ARG $RESUME_ARG \
  2>&1 | tee $TEE_MODE "$PROJECT_DIR/rl/runs/${RUN_NAME}.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [[ -d "$RUN_DIR" ]]; then
  conda run --no-capture-output -n toolcredit \
    python -m rl.monitor_run --run-dir "$RUN_DIR" || true
fi
exit "$TRAIN_EXIT"
