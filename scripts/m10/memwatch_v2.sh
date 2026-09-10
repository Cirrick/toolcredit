#!/usr/bin/env bash
# memwatch v2 (M10, plans/M10.md §4): container-memory sampler for locating the anonymous-memory
# growth process.  Read-only; never touches the training process.
#   bash scripts/m10/memwatch_v2.sh <run_dir> [interval_seconds=60]
# Appends one JSON line per interval to <run_dir>/analysis/container_memory.jsonl with
#   * cgroup current/peak/max/anon/file/shmem/file_mapped/kernel/oom_kill,
#   * the latest effective step, read from <run_dir>/e7_ledger.jsonl, which the trainer fsyncs once
#     per effective step (the v1 source, <run_dir>.log, lags by minutes because Ray batches worker
#     stdout; the log grep is kept only as a fallback and uses /usr/bin/grep -a so ANSI/binary
#     heuristics and shell function wrappers cannot blank it),
#   * per-process Pss_Anon / Pss_Shmem / Rss (GiB) for the top-12 RSS processes, keyed by "<comm>:<pid>",
#     so ``analysis/m10_memwatch_report.py`` can rank Pss_Anon growth per process.
RUN_DIR="${1:?run dir}"; INTERVAL="${2:-60}"
OUT="$RUN_DIR/analysis/container_memory.jsonl"
mkdir -p "$RUN_DIR/analysis"
CG="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/1/cgroup)"
GREP=/usr/bin/grep
rd() { cat "$1" 2>/dev/null || echo null; }
st() { local v; v=$($GREP -a -E "^$1 " "$CG/memory.stat" 2>/dev/null | awk '{print $2}'); echo "${v:-null}"; }
while true; do
  step=$(sed -n 's/.*"effective_step": *\([0-9]*\).*/\1/p' "$RUN_DIR/e7_ledger.jsonl" 2>/dev/null | tail -1)
  [ -n "$step" ] || step=$($GREP -a -o "training/global_step:[0-9]*" "$RUN_DIR.log" 2>/dev/null | tail -1 | cut -d: -f2)
  procs=""
  while read -r rss pid comm; do
    [ -r "/proc/$pid/smaps_rollup" ] || continue
    fields=$(awk '/^(Rss|Pss_Anon|Pss_Shmem|Pss_File|Private_Dirty):/ {gsub(":","",$1); printf "%s\"%s\":%.3f", (n++?",":""), $1, $2/1048576}' "/proc/$pid/smaps_rollup" 2>/dev/null)
    [ -n "$fields" ] || continue
    procs="$procs${procs:+,}\"$comm:$pid\":{$fields}"
  done < <(ps -eo rss,pid,comm --sort=-rss --no-headers | head -12)
  oom=$($GREP -a -E "^oom_kill " "$CG/memory.events" 2>/dev/null | awk '{print $2}'); oom=${oom:-null}
  printf '{"at":"%s","step":%s,"cgroup_current":%s,"cgroup_peak":%s,"cgroup_max":%s,"anon":%s,"file":%s,"shmem":%s,"file_mapped":%s,"kernel":%s,"oom_kill":%s,"processes":{%s}}\n' \
    "$(date -u +%FT%TZ)" "${step:-0}" "$(rd $CG/memory.current)" "$(rd $CG/memory.peak)" "$(rd $CG/memory.max)" \
    "$(st anon)" "$(st file)" "$(st shmem)" "$(st file_mapped)" "$(st kernel)" "$oom" "$procs" >> "$OUT"
  sleep "$INTERVAL"
done
