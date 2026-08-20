#!/usr/bin/env bash
# Launch with: tmux new-session -d -s m4-e3-smoke 'bash scripts/m4/run_e3.sh smoke'
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-full}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

case "$MODE" in
  smoke)
    RUN_NAME="${RUN_NAME:-e3_smoke_${STAMP}}"
    MODE_ARG="--smoke"
    ;;
  full)
    RUN_NAME="${RUN_NAME:-e3_grpo_baseline_${STAMP}}"
    MODE_ARG=""
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

RUN_DIR="$PROJECT_DIR/rl/runs/$RUN_NAME"
RESUME_ARG=""
TEE_MODE=""
if [[ "${RESUME:-0}" == "1" ]]; then
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "resume run directory does not exist: $RUN_DIR" >&2
    exit 1
  fi
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
  python -m rl.launch.e3_grpo_baseline \
  --run-name "$RUN_NAME" $MODE_ARG $RESUME_ARG \
  2>&1 | tee $TEE_MODE "$PROJECT_DIR/rl/runs/${RUN_NAME}.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [[ -d "$RUN_DIR" ]]; then
  conda run --no-capture-output -n toolcredit \
    python -m rl.monitor_run --run-dir "$RUN_DIR" || true
fi
exit "$TRAIN_EXIT"
