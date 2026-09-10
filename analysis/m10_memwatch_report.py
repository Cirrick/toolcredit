"""Rank anonymous-memory growth per process from memwatch v2 samples (plans/M10.md §4, §13 row 8).

Diagnosis only (plan §1: locate, do not fix).  For every process key ``<comm>:<pid>`` seen in
``analysis/container_memory.jsonl`` the report fits a least-squares slope of ``Pss_Anon`` against
wall-clock hours and against the training step, and reports cgroup ``anon`` the same way.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

GIB = 2**30
ANON_STOP_THRESHOLD_GIB = 110.0


def _slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or max(xs) == min(xs):
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def steps_from_ledger(rows: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    """Recover a per-sample step when the sampler's own ``step`` field is stale or absent.

    The ledger records ``recorded_at`` once per effective step, so a sample taken at time ``t``
    belongs to the last step recorded at or before ``t``.  Samples with a usable ``step`` are
    left alone; this only repairs runs sampled before the ledger-reading memwatch (v2.1).
    """
    ledger_path = run_dir / "e7_ledger.jsonl"
    if not ledger_path.is_file():
        return rows
    marks: list[tuple[datetime, int]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            marks.append((datetime.fromisoformat(entry["recorded_at"]), int(entry["effective_step"])))
    marks.sort()
    if not marks or max(float(r.get("step") or 0) for r in rows) >= marks[-1][1]:
        return rows
    repaired = []
    for row in rows:
        at = datetime.fromisoformat(row["at"].replace("Z", "+00:00"))
        step = 0
        for mark_at, mark_step in marks:
            if mark_at <= at:
                step = mark_step
            else:
                break
        repaired.append({**row, "step": step, "step_source": "ledger_timestamp"})
    return repaired


def _hours(rows: list[dict[str, Any]]) -> list[float]:
    t0 = datetime.fromisoformat(rows[0]["at"].replace("Z", "+00:00"))
    return [(datetime.fromisoformat(r["at"].replace("Z", "+00:00")) - t0).total_seconds() / 3600 for r in rows]


def _driver_pids(row: dict[str, Any]) -> frozenset[str]:
    """PIDs of the Ray driver actor in one sample; a new set means the run process restarted."""
    return frozenset(key.split(":", 1)[1] for key in (row.get("processes") or {}) if key.startswith("ray::E7Dynamic"))


def split_segments(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Partition samples at process restarts (plan §3.5 segments all append to one file).

    A cgroup slope fitted across a restart is meaningless: each segment reloads the model, so
    ``anon`` drops to a fresh baseline and climbs again.  Samples are cut whenever the driver
    actor's PID set changes to one that shares no PID with the current segment.
    """
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    live: frozenset[str] = frozenset()
    for row in rows:
        pids = _driver_pids(row)
        if current and pids and live and not (pids & live):
            segments.append(_trim_teardown(current))
            current = []
            live = frozenset()
        current.append(row)
        if pids:
            live = pids if not live else (live & pids) or pids
    if current:
        segments.append(_trim_teardown(current))
    return [segment for segment in segments if segment]


def _trim_teardown(segment: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop trailing samples taken after the driver exited.

    memwatch keeps sampling for up to one interval after the run ends.  Those samples still carry
    the last effective step (the ledger is on disk) but show ``anon`` collapsing as the workers are
    reaped, which drags the per-step fit toward zero.  A sample with no driver actor process is
    past the end of the segment.
    """
    end = len(segment)
    while end > 0 and not _driver_pids(segment[end - 1]):
        end -= 1
    return segment[:end] if end else segment


def _steady_state(pairs: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """(step, anon) pairs after this segment's first completed step, i.e. past model loading.

    A resumed segment starts with the previous segment's last step already in the ledger, so its
    startup samples carry a stale step number.  Fitting from the first *increase* within the
    segment drops the reload ramp in both the fresh and the resumed case.
    """
    steps = [s for _, s, _ in pairs]
    first = steps[0]
    start = next((i for i, s in enumerate(steps) if s > first), None)
    if start is None:
        return []
    return [(s, a) for _, s, a in pairs[start:] if s > 0]


def _cgroup_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """cgroup anon summary for one uninterrupted process lifetime."""
    hours = _hours(rows)
    pairs = [(h, s, float(r["anon"]) / GIB) for h, s, r in
             zip(hours, [float(r.get("step") or 0) for r in rows], rows) if r.get("anon") not in (None, "null")]
    if not pairs:
        return {"samples": len(rows)}
    stepped = _steady_state(pairs)
    return {
        "samples": len(rows),
        "hours": round(pairs[-1][0], 3),
        "step_first": int(min(s for _, s, _ in pairs if s > 0)) if stepped else None,
        "step_last": int(max(s for _, s, _ in pairs)),
        "anon_gib_first": round(pairs[0][2], 2),
        "anon_gib_last": round(pairs[-1][2], 2),
        "anon_gib_peak": round(max(a for _, _, a in pairs), 2),
        "anon_gib_per_hour": _slope([h for h, _, _ in pairs], [a for _, _, a in pairs]),
        "anon_gib_per_step": _slope([s for s, _ in stepped], [a for _, a in stepped]),
        "steady_state_samples": len(stepped),
        "anon_gib_steady_first": round(stepped[0][1], 2) if stepped else None,
        "anon_gib_steady_last": round(stepped[-1][1], 2) if stepped else None,
    }


def report(rows: list[dict[str, Any]], min_samples: int = 5) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    hours = _hours(rows)
    steps = [float(r.get("step") or 0) for r in rows]
    anon = [float(r["anon"]) / GIB for r in rows if r.get("anon") not in (None, "null")]
    stepped = [(s, float(r["anon"]) / GIB) for s, r in zip(steps, rows) if s > 0 and r.get("anon") not in (None, "null")]
    cgroup = {
        "samples": len(rows),
        "hours": round(hours[-1], 3),
        "step_first": int(min(s for s in steps if s > 0)) if any(s > 0 for s in steps) else None,
        "step_last": int(max(steps)),
        "anon_gib_first": round(anon[0], 2) if anon else None,
        "anon_gib_last": round(anon[-1], 2) if anon else None,
        "anon_gib_peak": round(max(anon), 2) if anon else None,
        "anon_gib_per_hour": _slope(hours[: len(anon)], anon),
        "anon_gib_per_step": _slope([s for s, _ in stepped], [a for _, a in stepped]),
        "cgroup_peak_gib": round(max(float(r["cgroup_peak"]) for r in rows if r.get("cgroup_peak") not in (None, "null")) / GIB, 2),
        "oom_kill_last": rows[-1].get("oom_kill"),
        "anon_stop_threshold_gib": ANON_STOP_THRESHOLD_GIB,
        "anon_exceeded_threshold": bool(anon) and max(anon) > ANON_STOP_THRESHOLD_GIB,
    }
    segments = split_segments(rows)
    cgroup["segments"] = [_cgroup_slice(segment) for segment in segments]
    # Pooling across restarts fits a meaningless slope (each segment reloads the model), so the
    # headline per-step slope comes from the longest uninterrupted segment.
    usable = [s for s in cgroup["segments"] if s.get("anon_gib_per_step") is not None]
    longest = max(usable, key=lambda s: s["steady_state_samples"]) if usable else None
    cgroup["anon_gib_per_step_pooled"] = cgroup["anon_gib_per_step"]
    cgroup["anon_gib_per_step"] = longest["anon_gib_per_step"] if longest else None
    cgroup["slope_from_segment_samples"] = longest["steady_state_samples"] if longest else None
    cgroup["restarts"] = len(segments) - 1
    per_process: dict[str, dict[str, Any]] = {}
    for hour, step, row in zip(hours, steps, rows):
        for key, fields in (row.get("processes") or {}).items():
            if "Pss_Anon" not in fields:
                continue
            entry = per_process.setdefault(key, {"hours": [], "steps": [], "pss_anon": [], "rss": []})
            entry["hours"].append(hour)
            entry["steps"].append(step)
            entry["pss_anon"].append(float(fields["Pss_Anon"]))
            entry["rss"].append(float(fields.get("Rss", 0.0)))
    ranking = []
    for key, entry in per_process.items():
        if len(entry["pss_anon"]) < min_samples:
            continue
        stepped_proc = [(s, a) for s, a in zip(entry["steps"], entry["pss_anon"]) if s > 0]
        ranking.append(
            {
                "process": key,
                "samples": len(entry["pss_anon"]),
                "pss_anon_gib_first": round(entry["pss_anon"][0], 3),
                "pss_anon_gib_last": round(entry["pss_anon"][-1], 3),
                "pss_anon_gib_delta": round(entry["pss_anon"][-1] - entry["pss_anon"][0], 3),
                "pss_anon_gib_per_hour": _slope(entry["hours"], entry["pss_anon"]),
                "pss_anon_gib_per_step": _slope([s for s, _ in stepped_proc], [a for _, a in stepped_proc]),
                "rss_gib_last": round(entry["rss"][-1], 3),
            }
        )
    ranking.sort(key=lambda item: -(item["pss_anon_gib_delta"]))
    return {"cgroup": cgroup, "process_ranking": ranking}


def render_markdown(payload: dict[str, Any]) -> str:
    if payload.get("samples") == 0:
        return "# memwatch v2\n\nno samples\n"
    cg = payload["cgroup"]
    lines = [
        "# memwatch v2 — anonymous memory growth",
        "",
        f"- samples: {cg['samples']} over {cg['hours']} h, steps {cg['step_first']}–{cg['step_last']}",
        f"- cgroup anon: {cg['anon_gib_first']} → {cg['anon_gib_last']} GiB (peak {cg['anon_gib_peak']}); "
        f"cgroup peak {cg['cgroup_peak_gib']} GiB; oom_kill {cg['oom_kill_last']}; "
        f"threshold {cg['anon_stop_threshold_gib']} GiB exceeded: {cg['anon_exceeded_threshold']}",
        f"- per-step anon slope: {cg['anon_gib_per_step']} GiB/step from the longest of "
        f"{cg['restarts'] + 1} process segment(s) ({cg['slope_from_segment_samples']} post-startup samples); "
        f"pooling across restarts would give {cg['anon_gib_per_step_pooled']} GiB/step and is not used",
        "",
        "| segment | samples | steps | anon first→last (GiB) | GiB/h | GiB/step |",
        "|---|---|---|---|---|---|",
        *[
            f"| {i + 1} | {s.get('samples')} ({s.get('steady_state_samples')} post-startup) | "
            f"{s.get('step_first')}–{s.get('step_last')} | "
            f"{s.get('anon_gib_first')}→{s.get('anon_gib_last')} "
            f"(steady {s.get('anon_gib_steady_first')}→{s.get('anon_gib_steady_last')}) | "
            f"{'n/a' if s.get('anon_gib_per_hour') is None else format(s['anon_gib_per_hour'], '.3f')} | "
            f"{'n/a' if s.get('anon_gib_per_step') is None else format(s['anon_gib_per_step'], '.4f')} |"
            for i, s in enumerate(cg["segments"])
        ],
        "",
        "| process | samples | Pss_Anon first→last (GiB) | Δ (GiB) | GiB/h | GiB/step | RSS last |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in payload["process_ranking"]:
        per_hour = "n/a" if item["pss_anon_gib_per_hour"] is None else f"{item['pss_anon_gib_per_hour']:.3f}"
        per_step = "n/a" if item["pss_anon_gib_per_step"] is None else f"{item['pss_anon_gib_per_step']:.4f}"
        lines.append(
            f"| `{item['process']}` | {item['samples']} | {item['pss_anon_gib_first']}→{item['pss_anon_gib_last']} | "
            f"{item['pss_anon_gib_delta']} | {per_hour} | {per_step} | {item['rss_gib_last']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()
    samples = args.run_dir / "analysis/container_memory.jsonl"
    payload = report(steps_from_ledger(load_samples(samples), args.run_dir), min_samples=args.min_samples)
    (args.run_dir / "analysis/memwatch_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "analysis/memwatch_report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload.get("cgroup", payload), indent=2))


if __name__ == "__main__":
    main()
