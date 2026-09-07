#!/usr/bin/env bash
# Exploratory step-125 sensitivity analysis for E3-NoTool (user-authorized 2026-09-07; not protocol v5).
# Launch with: tmux new-session -d -s m9-sens 'bash scripts/m9/run_m9_step125_sensitivity.sh <canonical m9 eval run dir>'
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/jovyan/.conda/envs/toolcredit/bin/python"
CANONICAL_RUN_DIR="${1:?canonical M9 eval run dir required}"
RUN_NAME="${M9_SENS_RUN_NAME:-m9_step125_sensitivity_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/eval/runs/$RUN_NAME"
MODEL="$PROJECT_DIR/eval/checkpoints/m9_sensitivity_e3notool_grpo_step_125_hf"
PORT="${M9_SERVER_PORT:-30000}"
export M9_EVAL_APPROVED=YES

if [[ -z "${TMUX:-}" ]]; then echo "must run inside tmux" >&2; exit 2; fi
if [[ -e "$RUN_DIR" && -z "${M9_SENS_RESUME:-}" ]]; then echo "refusing to reuse run directory: $RUN_DIR" >&2; exit 1; fi
cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR/server_logs" "$RUN_DIR/stage_logs"
server_pid=""
cleanup_server() { if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; server_pid=""; fi; }
on_exit() { code=$?; cleanup_server; if [[ "$code" -ne 0 ]]; then printf '{"status":"failed","exit_code":%d}\n' "$code" > "$RUN_DIR/stage_logs/failed.json"; fi; exit "$code"; }
trap on_exit EXIT INT TERM

if [[ ! -d "$MODEL" ]]; then
  "$PY" -m eval.m9_step125_sensitivity materialize > "$RUN_DIR/stage_logs/materialize.json"
fi
"$PY" -m eval.m9_step125_sensitivity resolve > "$RUN_DIR/stage_logs/resolve.json"

"$PY" -m sglang.launch_server --model-path "$MODEL" \
  --tokenizer-path "$PROJECT_DIR/sft/checkpoints/qwen3-1.7b-sft" \
  --host 127.0.0.1 --port "$PORT" --mem-fraction-static 0.85 \
  > "$RUN_DIR/server_logs/step125.log" 2>&1 &
server_pid=$!
ready=0
for _ in $(seq 1 120); do
  if curl --fail --silent "http://127.0.0.1:$PORT/get_model_info" >/dev/null; then ready=1; break; fi
  sleep 2
done
[[ "$ready" == "1" ]] || { echo "SGLang failed to become ready" >&2; exit 1; }
"$PY" -m eval.m9_step125_sensitivity generate --endpoint "http://127.0.0.1:$PORT" --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/generate.json"
cleanup_server
"$PY" -m eval.m9_step125_sensitivity score --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/score.json"
"$PY" -m eval.m9_step125_sensitivity downstream --run-dir "$RUN_DIR" --canonical-run-dir "$CANONICAL_RUN_DIR" > "$RUN_DIR/stage_logs/downstream.json"
"$PY" -m eval.m9_step125_sensitivity closure --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/closure.json"
printf '{"status":"sensitivity_completed","completed_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/stage_logs/completed.json"
trap - EXIT INT TERM
