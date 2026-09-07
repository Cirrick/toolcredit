#!/usr/bin/env bash
# Lightweight container-memory sampler for diagnosing pod interruptions (M9, 2026-09-07).
# Appends one JSON line per interval; read-only, never touches the training process.
RUN_DIR="${1:?run dir}"; INTERVAL="${2:-300}"
OUT="$RUN_DIR/analysis/container_memory.jsonl"
CG="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/1/cgroup)"
rd() { cat "$1" 2>/dev/null || echo null; }
st() { local v; v=$(grep -E "^$1 " "$CG/memory.stat" 2>/dev/null | awk '{print $2}'); echo "${v:-null}"; }
while true; do
  step=$(grep -o "training/global_step:[0-9]*" "$RUN_DIR.log" 2>/dev/null | tail -1 | cut -d: -f2)
  top=$(ps -eo rss,pid,comm --sort=-rss --no-headers | head -6 | awk '{printf "%s\"%s:%s\":%.2f", (NR>1?",":""), $3, $2, $1/1048576}')
  wpid=$(pgrep -f "ray::WorkerDict" | head -1)
  wd="null"
  if [ -n "$wpid" ] && [ -r /proc/$wpid/smaps_rollup ]; then
    wd=$(awk '/^(Rss|Pss_Anon|Pss_Shmem|Pss_File|Private_Dirty):/ {gsub(":","",$1); printf "%s\"%s\":%.2f", (n++?",":""), $1, $2/1048576}' /proc/$wpid/smaps_rollup); wd="{$wd}"
  fi
  oom=$(grep -E "^oom_kill " "$CG/memory.events" 2>/dev/null | awk '{print $2}'); oom=${oom:-null}
  printf '{"at":"%s","step":%s,"cgroup_current":%s,"cgroup_peak":%s,"cgroup_max":%s,"anon":%s,"file":%s,"shmem":%s,"file_mapped":%s,"kernel":%s,"oom_kill":%s,"workerdict_gib":%s,"top_rss_gib":{%s}}\n' \
    "$(date -u +%FT%TZ)" "${step:-0}" "$(rd $CG/memory.current)" "$(rd $CG/memory.peak)" "$(rd $CG/memory.max)" "$(st anon)" "$(st file)" "$(st shmem)" "$(st file_mapped)" "$(st kernel)" "$oom" "$wd" "$top" >> "$OUT"
  sleep "$INTERVAL"
done
