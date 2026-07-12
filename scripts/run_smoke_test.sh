#!/usr/bin/env bash
# M0 smoke test (PLAN §4): 20 samples -> multi-turn rollout w/ sandbox tool ->
# reward -> 1 GRPO gradient update. Acceptance: end-to-end < 10 minutes.
# Long-run convention: launch via tmux, e.g.
#   tmux new-session -d -s smoke 'bash scripts/run_smoke_test.sh'
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START=$(date +%s)

bash "$PROJECT_DIR/scripts/m0/run.sh" --smoke

ELAPSED=$(( $(date +%s) - START ))
echo "SMOKE TEST WALL TIME: ${ELAPSED}s (budget 600s)"
if [ "$ELAPSED" -lt 600 ]; then
    echo "SMOKE TEST PASSED"
else
    echo "SMOKE TEST FAILED: exceeded 10 min budget"
    exit 1
fi
