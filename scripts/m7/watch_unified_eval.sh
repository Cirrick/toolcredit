#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${1:?usage: watch_unified_eval.sh <run-name>}"
RUN_DIR="$PROJECT_DIR/eval/runs/$RUN_NAME"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "missing M7 run directory: $RUN_DIR" >&2
  exit 2
fi

find "$RUN_DIR/generation" -type f -name '*.jsonl' -print0 2>/dev/null \
  | sort -z \
  | xargs -0 -r wc -l
du -sh "$RUN_DIR"
df -h "$PROJECT_DIR"
