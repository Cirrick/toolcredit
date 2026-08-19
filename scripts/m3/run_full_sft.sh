#!/usr/bin/env bash
# Scoped M3 control: one full-parameter SFT run, no hyperparameter search.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=~/.conda/envs/toolcredit/bin/python
LOG_DIR="$PROJECT_DIR/scripts/m3/logs"
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"
exec "$PY" -m sft.train_full_sft "$@" 2>&1 | tee "$LOG_DIR/full_sft_20260819.log"
