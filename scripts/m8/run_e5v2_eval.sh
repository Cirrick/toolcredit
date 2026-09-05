#!/usr/bin/env bash
# M8 / E5-v2 unified evaluation under protocol v4 (plans/M8.md §8 step 6).
# Launch with: tmux new-session -d -s m8-e5v2-eval 'bash scripts/m8/run_e5v2_eval.sh'
# Only the E5-v2 role is generated (3,800 trajectories); raw/SFT/E3/E5-v1 are reused from M7.
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/jovyan/.conda/envs/toolcredit/bin/python"
RUN_NAME="${M8_EVAL_RUN_NAME:-m8_e5v2_eval_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/eval/runs/$RUN_NAME"
ROLE="e5v2_conserved_credit_step_200"
MODEL="$PROJECT_DIR/eval/checkpoints/m8_e5v2_conserved_credit_step_200_hf"
PORT="${M8_SERVER_PORT:-30000}"
SETTINGS=(greedy sampled)
SOURCES=(MATH500 AIME2024 AIME2025 GSM8K)
# The approved M8 plan (§8 step 6, one-time authorization of steps 0–6) authorizes this stage.
export M8_EVAL_APPROVED=YES

if [[ -z "${TMUX:-}" ]]; then
  echo "M8 GPU evaluation must run inside tmux" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" && -z "${M8_EVAL_RESUME:-}" ]]; then
  echo "refusing to reuse eval run directory: $RUN_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR/generation/$ROLE" "$RUN_DIR/server_logs" "$RUN_DIR/stage_logs"
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

"$PY" -m eval.m8_e5v2_eval preflight --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/preflight.json"

"$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$PROJECT_DIR/sft/checkpoints/qwen3-1.7b-sft" \
  --host 127.0.0.1 --port "$PORT" \
  --tool-call-parser qwen25 --mem-fraction-static 0.85 \
  > "$RUN_DIR/server_logs/$ROLE.log" 2>&1 &
server_pid=$!
ready=0
for _ in $(seq 1 120); do
  if curl --fail --silent "http://127.0.0.1:$PORT/get_model_info" >/dev/null; then ready=1; break; fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "SGLang failed to become ready for $ROLE" >&2
  exit 1
fi
for setting in "${SETTINGS[@]}"; do
  for source in "${SOURCES[@]}"; do
    "$PY" -m eval.m8_e5v2_eval generate-shard \
      --setting "$setting" --source "$source" \
      --endpoint "http://127.0.0.1:$PORT" \
      --output "$RUN_DIR/generation/$ROLE/$setting/$source.jsonl" \
      > "$RUN_DIR/stage_logs/generate_${setting}_${source}.json"
  done
done
cleanup_server
printf '{"status":"generation_completed","completed_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_DIR/stage_logs/generation.completed.json"

for setting in "${SETTINGS[@]}"; do
  for source in "${SOURCES[@]}"; do
    output="$RUN_DIR/scored/$ROLE/$setting/$source.jsonl"
    [[ -e "$output" ]] && continue
    "$PY" -m eval.m8_e5v2_eval score \
      --input "$RUN_DIR/generation/$ROLE/$setting/$source.jsonl" --output "$output" \
      > "$RUN_DIR/stage_logs/score_${setting}_${source}.json"
  done
done
"$PY" -m eval.m8_e5v2_eval downstream --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/downstream.json"
"$PY" -m eval.m8_e5v2_eval closure --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/closure.json"
printf '{"status":"eval_completed","completed_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_DIR/stage_logs/eval.completed.json"
trap - EXIT INT TERM
