#!/usr/bin/env bash
# M7 unified evaluator. GPU modes are approval-gated in eval.generate and must run in tmux.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/jovyan/.conda/envs/toolcredit/bin/python"
MODE="${1:-preflight}"
RUN_NAME="${M7_RUN_NAME:-m7_unified_eval_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/eval/runs/$RUN_NAME"

cd "$PROJECT_DIR"

if [[ "$MODE" == "preflight" ]]; then
  "$PY" -m eval.verify_diagnostic_freeze_v3_semantics
  "$PY" -m eval.generate preflight
  exit 0
fi

if [[ "$MODE" != "smoke" && "$MODE" != "canonical" ]]; then
  echo "usage: $0 {preflight|smoke|canonical}" >&2
  exit 2
fi
if [[ "${TMUX:-}" == "" ]]; then
  echo "M7 GPU work must run inside tmux" >&2
  exit 2
fi
if [[ "${M7_GPU_SMOKE_APPROVED:-}" != "YES" ]]; then
  echo "M7_GPU_SMOKE_APPROVED=YES is required after explicit user approval" >&2
  exit 2
fi
if [[ "$MODE" == "canonical" && "${M7_CANONICAL_EVAL_APPROVED:-}" != "YES" ]]; then
  echo "M7_CANONICAL_EVAL_APPROVED=YES is required after separate canonical approval" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/generation" "$RUN_DIR/server_logs"
"$PY" -m eval.verify_diagnostic_freeze_v3_semantics > "$RUN_DIR/preflight.json"

declare -A MODELS=(
  [raw_base_model]="/home/jovyan/Qwen/Qwen3-1.7B"
  [sft_start]="$PROJECT_DIR/sft/checkpoints/qwen3-1.7b-sft"
  [e3_grpo_baseline_step_200]="$PROJECT_DIR/eval/checkpoints/m7_e3_grpo_baseline_step_200_hf"
  [e5_turn_credit_step_200]="$PROJECT_DIR/eval/checkpoints/m7_e5_turn_credit_step_200_hf"
)
ROLES=(raw_base_model sft_start e3_grpo_baseline_step_200 e5_turn_credit_step_200)
SETTINGS=(greedy sampled)
SOURCES=(MATH500 AIME2024 AIME2025 GSM8K)
PORT="${M7_SERVER_PORT:-30000}"

for role in "${ROLES[@]}"; do
  model="${MODELS[$role]}"
  "$PY" -m sglang.launch_server \
    --model-path "$model" \
    --tokenizer-path "$PROJECT_DIR/sft/checkpoints/qwen3-1.7b-sft" \
    --host 127.0.0.1 --port "$PORT" \
    --tool-call-parser qwen25 --mem-fraction-static 0.85 \
    > "$RUN_DIR/server_logs/$role.log" 2>&1 &
  server_pid=$!
  cleanup_server() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap cleanup_server EXIT INT TERM
  ready=0
  for _ in $(seq 1 120); do
    if curl --fail --silent "http://127.0.0.1:$PORT/get_model_info" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "SGLang failed to become ready for $role" >&2
    exit 1
  fi
  for setting in "${SETTINGS[@]}"; do
    for source in "${SOURCES[@]}"; do
      output="$RUN_DIR/generation/$role/$setting/$source.jsonl"
      "$PY" -m eval.generate generate-shard \
        --scope "$MODE" --role "$role" --setting "$setting" --source "$source" \
        --endpoint "http://127.0.0.1:$PORT" --output "$output"
    done
  done
  cleanup_server
  trap - EXIT INT TERM
done
