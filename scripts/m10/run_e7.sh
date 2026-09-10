#!/usr/bin/env bash
# M10 / E7 launcher (plans/M10.md §8 steps 3–4).  Always run inside tmux:
#   smoke (5 steps, no checkpoint):
#     tmux new-session -d -s m10-e7-smoke 'bash scripts/m10/run_e7.sh smoke'
#   resume check (2 steps: stop at 1, then resume to 2):
#     tmux new-session -d -s m10-e7-rc1 'bash scripts/m10/run_e7.sh resume-check --stop-at-step 1'
#     RUN_NAME=<same> RESUME=1 tmux new-session -d -s m10-e7-rc2 'bash scripts/m10/run_e7.sh resume-check'
#   formal, one segment per invocation (plan §3.5):
#     tmux new-session -d -s m10-e7-seg1 'bash scripts/m10/run_e7.sh formal --stop-at-step 50'
#     RUN_NAME=<same> RESUME=1 tmux new-session -d -s m10-e7-seg2 'bash scripts/m10/run_e7.sh formal --stop-at-step 100'
# memwatch v2 (60 s) runs alongside; the gate for the mode runs after the launcher exits.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:?mode: smoke | resume-check | formal}"; shift || true
STAMP="$(date -u +%Y%m%d_%H%M%S)"
CONDA=/opt/conda/bin/conda
PY=("$CONDA" run --no-capture-output -n toolcredit python)

case "$MODE" in
  smoke) MODE_ARG="--smoke"; PREFIX="e7_dynamic_filtering_smoke"; GATE="smoke" ;;
  resume-check) MODE_ARG="--resume-check"; PREFIX="e7_dynamic_filtering_resumecheck"; GATE="resume-check" ;;
  formal) MODE_ARG=""; PREFIX="e7_dynamic_filtering"; GATE="segment" ;;
  *) echo "mode must be smoke | resume-check | formal" >&2; exit 2 ;;
esac

STOP_AT_STEP=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-at-step) STOP_AT_STEP="$2"; EXTRA_ARGS+=("--stop-at-step" "$2"); shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

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
if [[ -z "${TMUX:-}" ]]; then
  echo "E7 GPU runs must be launched inside tmux (CLAUDE.md 禁止事项 #4)" >&2
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p rl/runs
LEDGER="rl/runs/${RUN_NAME}.launch.jsonl"
printf '{"run_name":"%s","mode":"%s","resumed":%s,"stop_at_step":%s,"requested_at":"%s"}\n' \
  "$RUN_NAME" "$MODE" "$([[ -n "$RESUME_ARG" ]] && echo true || echo false)" "${STOP_AT_STEP:-null}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LEDGER"

# memwatch v2 (read-only) starts once the launcher has created the run dir.
( while [[ ! -d "$RUN_DIR" ]]; do sleep 5; done
  exec bash scripts/m10/memwatch_v2.sh "$RUN_DIR" 60 ) &
MEMWATCH_PID=$!
trap 'kill "$MEMWATCH_PID" 2>/dev/null || true' EXIT

"${PY[@]}" -m rl.launch.e7_dynamic_filtering --run-name "$RUN_NAME" $MODE_ARG $RESUME_ARG "${EXTRA_ARGS[@]}" \
  2>&1 | tee $TEE_MODE "rl/runs/${RUN_NAME}.log"
TRAIN_EXIT="${PIPESTATUS[0]}"

if [[ -d "$RUN_DIR" ]]; then
  "${PY[@]}" -m rl.monitor_run --run-dir "$RUN_DIR" || true
  "${PY[@]}" analysis/m10_memwatch_report.py "$RUN_DIR" || true
  if [[ "$TRAIN_EXIT" == "0" ]]; then
    GATE_ARGS=()
    if [[ "$GATE" == "segment" ]]; then
      TRACKER="$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt"
      [[ -f "$TRACKER" ]] && GATE_ARGS=(--stop-step "$(tr -d '[:space:]' < "$TRACKER")")
    fi
    if [[ "$GATE" == "resume-check" && -z "$RESUME_ARG" ]]; then
      echo "resume-check first segment done; run the second segment with RESUME=1 before the gate"
    else
      "${PY[@]}" -m rl.validate_e7_run "$GATE" "$RUN_DIR" "${GATE_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/analysis/${GATE//-/_}_gate.log"
      GATE_EXIT="${PIPESTATUS[0]}"
      echo "gate exit=$GATE_EXIT"
      # The last formal segment also runs the whole-run completion gate, whose output path
      # (analysis/formal_completion_gate.json) the v6 evaluation freeze binds by name.
      if [[ "$GATE" == "segment" && "${GATE_ARGS[1]:-}" == "200" && "$GATE_EXIT" == "0" ]]; then
        "${PY[@]}" -m rl.validate_e7_run formal "$RUN_DIR" \
          2>&1 | tee "$RUN_DIR/analysis/formal_completion_gate.log"
        echo "formal gate exit=${PIPESTATUS[0]}"
      fi
    fi
  fi
fi
echo "e7 launcher exit=$TRAIN_EXIT run_dir=$RUN_DIR"
exit "$TRAIN_EXIT"
