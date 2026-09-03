#!/usr/bin/env bash
# E3-only longer-budget diagnostic. Must run in tmux after explicit user approval.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/jovyan/.conda/envs/toolcredit/bin/python"
RUN_NAME="${E3_BUDGET_RUN_NAME:-e3_budget_diagnostic_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/eval/runs/$RUN_NAME"
PORT="${E3_BUDGET_SERVER_PORT:-30000}"
MODEL="$PROJECT_DIR/eval/checkpoints/m7_e3_grpo_baseline_step_200_hf"

cd "$PROJECT_DIR"

if [[ -z "${TMUX:-}" ]]; then
  echo "E3 budget diagnostic GPU work must run inside tmux" >&2
  exit 2
fi
if [[ "${E3_BUDGET_DIAGNOSTIC_APPROVED:-}" != "YES" ]]; then
  echo "E3_BUDGET_DIAGNOSTIC_APPROVED=YES is required after explicit user approval" >&2
  exit 2
fi

"$PY" -m eval.budget_diagnostic prepare --run-dir "$RUN_DIR"
mkdir -p "$RUN_DIR/server_logs"

"$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$PROJECT_DIR/sft/checkpoints/qwen3-1.7b-sft" \
  --host 127.0.0.1 --port "$PORT" \
  --tool-call-parser qwen25 --mem-fraction-static 0.85 \
  > "$RUN_DIR/server_logs/e3.log" 2>&1 &
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
  echo "SGLang failed to become ready for E3 budget diagnostic" >&2
  exit 1
fi

"$PY" -m eval.budget_diagnostic generate \
  --run-dir "$RUN_DIR" --endpoint "http://127.0.0.1:$PORT"

cleanup_server
trap - EXIT INT TERM

"$PY" -m eval.budget_diagnostic finalize --run-dir "$RUN_DIR"
cd "$RUN_DIR"
sha256sum --quiet -c hashes.sha256
