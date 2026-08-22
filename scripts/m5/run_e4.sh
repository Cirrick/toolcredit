#!/usr/bin/env bash
# Examples: run_e4.sh a smoke; run_e4.sh b full
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VARIANT="${1:?usage: $0 <a|b> [smoke|full]}"
MODE="${2:-full}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

case "$VARIANT" in
  a)
    CONFIG="rl/configs/e4a_exec_only.yaml"
    PREFIX="e4a_exec_only"
    ;;
  b)
    CONFIG="rl/configs/e4b_joint_shaping.yaml"
    PREFIX="e4b_joint_shaping"
    ;;
  *)
    echo "variant must be 'a' or 'b'" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke) MODE_ARG="--smoke"; PREFIX="${PREFIX}_smoke" ;;
  full) MODE_ARG="" ;;
  *) echo "mode must be 'smoke' or 'full'" >&2; exit 2 ;;
esac

RUN_NAME="${RUN_NAME:-${PREFIX}_${STAMP}}"
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
  --config "$CONFIG" --run-name "$RUN_NAME" $MODE_ARG $RESUME_ARG \
  2>&1 | tee $TEE_MODE "$PROJECT_DIR/rl/runs/${RUN_NAME}.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [[ -d "$RUN_DIR" ]]; then
  conda run --no-capture-output -n toolcredit \
    python -m rl.monitor_run --run-dir "$RUN_DIR" || true
fi
exit "$TRAIN_EXIT"
