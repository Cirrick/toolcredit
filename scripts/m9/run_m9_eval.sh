#!/usr/bin/env bash
# M9 / notool unified evaluation under protocol v5 (plans/M9.md §7 step 4).
# Launch with: tmux new-session -d -s m9-eval 'bash scripts/m9/run_m9_eval.sh'
# Smoke (one question per source, no scoring/downstream): M9_EVAL_SMOKE=1 bash scripts/m9/run_m9_eval.sh
# Four notool roles are generated (15,200 trajectories); the five tir roles are reused from M7/M8.
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/jovyan/.conda/envs/toolcredit/bin/python"
if [[ -n "${M9_EVAL_SMOKE:-}" ]]; then
  RUN_NAME="${M9_EVAL_RUN_NAME:-m9_notool_eval_smoke_$(date -u +%Y%m%d_%H%M%S)}"
  LIMIT_ARGS=(--limit-questions 1)
else
  RUN_NAME="${M9_EVAL_RUN_NAME:-m9_notool_eval_$(date -u +%Y%m%d_%H%M%S)}"
  LIMIT_ARGS=()
fi
RUN_DIR="$PROJECT_DIR/eval/runs/$RUN_NAME"
PORT="${M9_SERVER_PORT:-30000}"
ROLES=(raw_base_model_notool sft_start_notool e3_grpo_baseline_step_200_notool e3notool_grpo_step_200)
declare -A MODELS=(
  [raw_base_model_notool]="/home/jovyan/Qwen/Qwen3-1.7B"
  [sft_start_notool]="$PROJECT_DIR/sft/checkpoints/qwen3-1.7b-sft"
  [e3_grpo_baseline_step_200_notool]="$PROJECT_DIR/eval/checkpoints/m7_e3_grpo_baseline_step_200_hf"
  [e3notool_grpo_step_200]="$PROJECT_DIR/eval/checkpoints/m9_e3notool_grpo_step_200_hf"
)
SETTINGS=(greedy sampled)
SOURCES=(MATH500 AIME2024 AIME2025 GSM8K)
# The approved M9 plan (§7 step 4) authorizes this stage.
export M9_EVAL_APPROVED=YES

if [[ -z "${TMUX:-}" ]]; then
  echo "M9 GPU evaluation must run inside tmux" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" && -z "${M9_EVAL_RESUME:-}" ]]; then
  echo "refusing to reuse eval run directory: $RUN_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR/generation" "$RUN_DIR/server_logs" "$RUN_DIR/stage_logs"
server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
on_exit() {
  code=$?
  cleanup_server
  if [[ "$code" -ne 0 ]]; then
    printf '{"status":"failed","exit_code":%d,"failed_at":"%s"}\n' "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$RUN_DIR/stage_logs/eval.failed.json"
  fi
  exit "$code"
}
trap on_exit EXIT INT TERM

"$PY" -m eval.m9_notool_eval preflight --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/preflight.json"

for role in "${ROLES[@]}"; do
  if [[ -f "$RUN_DIR/stage_logs/generate_${role}.json" ]]; then
    continue
  fi
  "$PY" -m sglang.launch_server \
    --model-path "${MODELS[$role]}" \
    --tokenizer-path "$PROJECT_DIR/sft/checkpoints/qwen3-1.7b-sft" \
    --host 127.0.0.1 --port "$PORT" \
    --mem-fraction-static 0.85 \
    > "$RUN_DIR/server_logs/$role.log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 120); do
    if curl --fail --silent "http://127.0.0.1:$PORT/get_model_info" >/dev/null; then ready=1; break; fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "SGLang failed to become ready for $role" >&2
    exit 1
  fi
  "$PY" -m eval.m9_notool_eval generate-role \
    --role "$role" --endpoint "http://127.0.0.1:$PORT" --run-dir "$RUN_DIR" "${LIMIT_ARGS[@]}" \
    > "$RUN_DIR/stage_logs/generate_${role}.json.partial"
  mv "$RUN_DIR/stage_logs/generate_${role}.json.partial" "$RUN_DIR/stage_logs/generate_${role}.json"
  cleanup_server
done
printf '{"status":"generation_completed","completed_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_DIR/stage_logs/generation.completed.json"

if [[ -n "${M9_EVAL_SMOKE:-}" ]]; then
  printf '{"status":"smoke_generation_completed","completed_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$RUN_DIR/stage_logs/eval.completed.json"
  trap - EXIT INT TERM
  exit 0
fi

for role in "${ROLES[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    for source in "${SOURCES[@]}"; do
      output="$RUN_DIR/scored/$role/$setting/$source.jsonl"
      [[ -e "$output" ]] && continue
      "$PY" -m eval.m9_notool_eval score \
        --input "$RUN_DIR/generation/$role/$setting/$source.jsonl" --output "$output" \
        > "$RUN_DIR/stage_logs/score_${role}_${setting}_${source}.json"
    done
  done
done
"$PY" -m eval.m9_notool_eval downstream --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/downstream.json"
"$PY" -m eval.m9_notool_eval closure --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/closure.json"
printf '{"status":"eval_completed","completed_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_DIR/stage_logs/eval.completed.json"
trap - EXIT INT TERM
