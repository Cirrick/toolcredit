#!/usr/bin/env bash
# Lightweight checkpoint monitor: summarize only new validation steps or terminal state.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${1:?usage: $0 <run_name> [interval_seconds]}"
INTERVAL="${2:-600}"
RUN_DIR="$PROJECT_DIR/rl/runs/$RUN_NAME"
MONITOR_LOG="$RUN_DIR/monitor.log"
LAST_VALIDATION_STEP="-1"
LAST_STATE="unknown"

if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "interval_seconds must be a positive integer" >&2
  exit 2
fi

while true; do
  if [[ -f "$RUN_DIR/status.json" ]]; then
    STATE="$(jq -r '.state // "unknown"' "$RUN_DIR/status.json")"
  else
    STATE="unknown"
  fi

  if [[ -d "$RUN_DIR/predictions/validation" ]]; then
    LATEST_VALIDATION_STEP="$(
      find "$RUN_DIR/predictions/validation" -maxdepth 1 -type f -name '*.jsonl' -printf '%f\n' \
        | sed 's/\.jsonl$//' | sort -n | awk 'END { print }'
    )"
    LATEST_VALIDATION_STEP="${LATEST_VALIDATION_STEP:--1}"
  else
    LATEST_VALIDATION_STEP="-1"
  fi

  if (( LATEST_VALIDATION_STEP > LAST_VALIDATION_STEP )) || [[ "$STATE" != "$LAST_STATE" && "$STATE" != "running" ]]; then
    if conda run --no-capture-output -n toolcredit \
      python -m rl.monitor_run --run-dir "$RUN_DIR" >/dev/null; then
      printf '%s ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$MONITOR_LOG"
      jq -c '{status: .status.state, headline: .headline}' "$RUN_DIR/metrics.json" >>"$MONITOR_LOG"
      LAST_VALIDATION_STEP="$LATEST_VALIDATION_STEP"
    else
      printf '%s monitor_failed state=%s validation_step=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STATE" "$LATEST_VALIDATION_STEP" >>"$MONITOR_LOG"
    fi
  fi

  LAST_STATE="$STATE"
  [[ "$STATE" == "completed" || "$STATE" == "failed" ]] && exit 0
  sleep "$INTERVAL"
done
