#!/usr/bin/env bash
# M9 / E3-NoTool launcher (plans/M9.md §7 steps 2-3).
# Launch with: tmux new-session -d -s m9-notool-smoke  'bash scripts/m9/run_e3notool.sh smoke'
#          or: tmux new-session -d -s m9-notool-formal 'bash scripts/m9/run_e3notool.sh full'
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-smoke}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

case "$MODE" in
  smoke) MODE_ARG="--smoke"; PREFIX="e3notool_grpo_smoke" ;;
  full) MODE_ARG=""; PREFIX="e3notool_grpo" ;;
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
LEDGER="$PROJECT_DIR/rl/runs/${RUN_NAME}.launch.json"
if [[ -z "$RESUME_ARG" ]]; then
  printf '{"run_name": "%s", "resumed": false, "checkpoint": "%s", "run_dir_absent_before_launch": true, "requested_at": "%s"}\n' \
    "$RUN_NAME" "$PROJECT_DIR/sft/checkpoints/qwen3-1.7b-sft" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LEDGER"
fi
set +e
conda run --no-capture-output -n toolcredit \
  python -m rl.launch.e3notool_grpo \
  --run-name "$RUN_NAME" $MODE_ARG $RESUME_ARG \
  2>&1 | tee $TEE_MODE "$PROJECT_DIR/rl/runs/${RUN_NAME}.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [[ -d "$RUN_DIR" ]]; then
  conda run --no-capture-output -n toolcredit \
    python -m rl.monitor_run --run-dir "$RUN_DIR" || true
fi
exit "$TRAIN_EXIT"
