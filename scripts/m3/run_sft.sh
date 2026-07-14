#!/usr/bin/env bash
# M3: LoRA SFT training. Long task: run inside tmux, e.g.
#   tmux new-session -d -s sft 'bash scripts/m3/run_sft.sh'
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=~/.conda/envs/toolcredit/bin/python
LOG_DIR="$PROJECT_DIR/scripts/m3/logs"
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"
exec "$PY" sft/train_sft.py "$@" 2>&1 | tee "$LOG_DIR/sft_$(date +%Y%m%d_%H%M%S).log"
