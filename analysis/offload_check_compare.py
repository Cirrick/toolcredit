"""Compare the offload-off 5-step check against the first 5 steps of the E3 formal run.

Reads the two veRL console logs (``rl/runs/<name>.log``), extracts the per-step metric
lines, and writes ``<check_run>/analysis/offload_check_compare.{json,md}``.  Also
summarises the container-memory samples written by ``scripts/m9/memwatch.sh``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E3_LOG = PROJECT_ROOT / "rl/runs/e3_grpo_baseline_20260819_224555.log"
STEP_RE = re.compile(r"step:(\d+) - (.*)$")
METRIC_RE = re.compile(r"([A-Za-z0-9_/.\-]+):(?:np\.\w+\()?(-?[0-9.eE+\-]+|nan)\)?")

NUMERIC_KEYS = [
    "prompt_length/mean",
    "critic/score/mean",
    "critic/advantages/mean",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/entropy",
    "actor/grad_norm",
    "actor/ppo_kl",
    "response_length/mean",
    "tool_call_counts/mean",
    "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb",
    "actor/perf/cpu_memory_used_gb",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/save_checkpoint",
    "timing_s/testing",
    "timing_s/step",
]


def parse_steps(log_path: Path, max_step: int) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        if step > max_step or step in steps:
            continue
        metrics = {k: float(v) for k, v in METRIC_RE.findall(match.group(2))}
        if "training/global_step" in metrics:
            steps[step] = metrics
    return steps


def parse_validation(run_dir: Path) -> list[dict[str, float]]:
    """Validation series from the run's ``metrics.json`` (written by ``rl.monitor_run``)."""
    path = run_dir / "metrics.json"
    if not path.is_file():
        return []
    native = json.loads(path.read_text(encoding="utf-8")).get("native", {})
    return native.get("val-core/toolcredit_math500/acc/mean@1", [])


def memwatch_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"samples": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gib = 2**30
    series = [
        {
            "at": r["at"],
            "step": r["step"],
            "cgroup_current_gib": round((r.get("cgroup_current") or 0) / gib, 2),
            "anon_gib": round((r.get("anon") or 0) / gib, 2),
            "shmem_gib": round((r.get("shmem") or 0) / gib, 2),
            "workerdict_rss_gib": (r.get("workerdict_gib") or {}).get("Rss"),
            "oom_kill": r.get("oom_kill"),
        }
        for r in rows
    ]
    return {
        "samples": len(rows),
        "cgroup_current_gib_min": min(s["cgroup_current_gib"] for s in series),
        "cgroup_current_gib_max": max(s["cgroup_current_gib"] for s in series),
        "cgroup_peak_gib": round(max((r.get("cgroup_peak") or 0) for r in rows) / gib, 2),
        "cgroup_max_gib": round((rows[-1].get("cgroup_max") or 0) / gib, 2),
        "anon_gib_first_last": [series[0]["anon_gib"], series[-1]["anon_gib"]],
        "shmem_gib_first_last": [series[0]["shmem_gib"], series[-1]["shmem_gib"]],
        "workerdict_rss_gib_max": max((s["workerdict_rss_gib"] or 0) for s in series),
        "oom_kill_last": series[-1]["oom_kill"],
        "series": series,
    }


def table(e3: dict[int, dict[str, float]], check: dict[int, dict[str, float]]) -> str:
    steps = sorted(set(e3) & set(check))
    lines = ["| metric | " + " | ".join(f"s{s} E3 / off" for s in steps) + " |", "|---|" + "---|" * len(steps)]
    for key in NUMERIC_KEYS:
        cells = []
        for s in steps:
            a, b = e3[s].get(key), check[s].get(key)
            fa = "–" if a is None else f"{a:.4g}"
            fb = "–" if b is None else f"{b:.4g}"
            cells.append(f"{fa} / {fb}")
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--e3-log", type=Path, default=E3_LOG)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    check_log = run_dir.parent / f"{run_dir.name}.log"
    e3 = parse_steps(args.e3_log, args.steps)
    check = parse_steps(check_log, args.steps)
    result = {
        "e3_log": str(args.e3_log),
        "check_log": str(check_log),
        "steps_compared": sorted(set(e3) & set(check)),
        "e3": {str(k): {m: v.get(m) for m in NUMERIC_KEYS} for k, v in e3.items()},
        "check": {str(k): {m: v.get(m) for m in NUMERIC_KEYS} for k, v in check.items()},
        "validation_e3_first": parse_validation(args.e3_log.parent / args.e3_log.stem)[:2],
        "validation_check": parse_validation(run_dir),
        "memwatch": memwatch_summary(run_dir / "analysis/container_memory.jsonl"),
    }
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "offload_check_compare.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = ["# Offload-off check vs E3 formal steps 1–5", "", table(e3, check), ""]
    mw = result["memwatch"]
    if mw.get("samples"):
        md += [
            f"memwatch: {mw['samples']} samples; cgroup current {mw['cgroup_current_gib_min']}–"
            f"{mw['cgroup_current_gib_max']} GiB (peak {mw['cgroup_peak_gib']}, limit {mw['cgroup_max_gib']}); "
            f"anon {mw['anon_gib_first_last']}, shmem {mw['shmem_gib_first_last']}, "
            f"WorkerDict RSS max {mw['workerdict_rss_gib_max']} GiB, oom_kill={mw['oom_kill_last']}",
        ]
    (out_dir / "offload_check_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("steps_compared", "validation_check")}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
