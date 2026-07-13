#!/usr/bin/env bash
# M1: sglang server for the tool-gain probe. Long task: run inside tmux, e.g.
#   tmux new-session -d -s sgl 'bash scripts/m1/serve_sglang.sh'
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=~/.conda/envs/toolcredit/bin/python
MODEL="${MODEL:-$HOME/Qwen/Qwen3-1.7B}"
PORT="${PORT:-30000}"
LOG_DIR="$PROJECT_DIR/scripts/m1/logs"
mkdir -p "$LOG_DIR"

exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --tool-call-parser qwen25 \
  --mem-fraction-static 0.85 \
  2>&1 | tee "$LOG_DIR/sglang_$(date +%Y%m%d_%H%M%S).log"
