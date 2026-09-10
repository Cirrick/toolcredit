#!/usr/bin/env bash
# M10 / E7 unified evaluation under protocol v6 (plans/M10.md §8 step 5).
# Launch with: tmux new-session -d -s m10-eval 'bash scripts/m10/run_m10_eval.sh'
# Smoke (one question per source, no scoring/downstream): M10_EVAL_SMOKE=1 bash scripts/m10/run_m10_eval.sh
# Two tir roles are generated (7,600 trajectories); every other role is reused from M7/M8/M9.
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/jovyan/.conda/envs/toolcredit/bin/python"
if [[ -n "${M10_EVAL_SMOKE:-}" ]]; then
  RUN_NAME="${M10_EVAL_RUN_NAME:-m10_e7_eval_smoke_$(date -u +%Y%m%d_%H%M%S)}"
  LIMIT_ARGS=(--limit-questions 1)
else
  RUN_NAME="${M10_EVAL_RUN_NAME:-m10_e7_eval_$(date -u +%Y%m%d_%H%M%S)}"
  LIMIT_ARGS=()
fi
RUN_DIR="$PROJECT_DIR/eval/runs/$RUN_NAME"
PORT="${M10_SERVER_PORT:-30000}"
ROLES=(e7_dynamic_filter_step_200 e7_equal_compute)
SETTINGS=(greedy sampled)
SOURCES=(MATH500 AIME2024 AIME2025 GSM8K)
# The approved M10 plan (§8 step 5) authorizes this stage.
export M10_EVAL_APPROVED=YES

if [[ -z "${TMUX:-}" ]]; then
  echo "M10 GPU evaluation must run inside tmux" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" && -z "${M10_EVAL_RESUME:-}" ]]; then
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

"$PY" -m eval.m10_e7_eval preflight --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/preflight.json"

# The inference locator for each role comes from the frozen v6 protocol, never from a literal here.
locator_for() {
  "$PY" - "$1" <<'PY'
import json, sys
from eval.build_diagnostic_freeze_v6 import PROTOCOL_PATH
print(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["checkpoints"][sys.argv[1]]["inference_locator"])
PY
}

for role in "${ROLES[@]}"; do
  if [[ -f "$RUN_DIR/stage_logs/generate_${role}.json" ]]; then
    continue
  fi
  MODEL="$(locator_for "$role")"
  "$PY" -m sglang.launch_server \
    --model-path "$MODEL" \
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
  for setting in "${SETTINGS[@]}"; do
    for source in "${SOURCES[@]}"; do
      output="$RUN_DIR/generation/$role/$setting/$source.jsonl"
      stage="$RUN_DIR/stage_logs/generate_${role}_${setting}_${source}.json"
      [[ -f "$stage" ]] && continue
      mkdir -p "$(dirname "$output")"
      "$PY" -m eval.m10_e7_eval generate-shard \
        --role "$role" --setting "$setting" --source "$source" \
        --endpoint "http://127.0.0.1:$PORT" --output "$output" "${LIMIT_ARGS[@]}" > "$stage.partial"
      mv "$stage.partial" "$stage"
    done
  done
  printf '{"status":"role_completed","role":"%s","completed_at":"%s"}\n' "$role" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$RUN_DIR/stage_logs/generate_${role}.json"
  cleanup_server
done
printf '{"status":"generation_completed","completed_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_DIR/stage_logs/generation.completed.json"

if [[ -n "${M10_EVAL_SMOKE:-}" ]]; then
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
      "$PY" -m eval.m10_e7_eval score \
        --input "$RUN_DIR/generation/$role/$setting/$source.jsonl" --output "$output" \
        > "$RUN_DIR/stage_logs/score_${role}_${setting}_${source}.json"
    done
  done
done
"$PY" -m eval.m10_e7_eval downstream --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/downstream.json"
"$PY" -m eval.m10_e7_eval closure --run-dir "$RUN_DIR" > "$RUN_DIR/stage_logs/closure.json"
printf '{"status":"eval_completed","completed_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_DIR/stage_logs/eval.completed.json"
trap - EXIT INT TERM
