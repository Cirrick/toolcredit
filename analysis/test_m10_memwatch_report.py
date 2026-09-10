"""M10 memwatch v2 report: step recovery from the ledger and restart-aware slope fitting.

The formal run writes all four segments into one ``container_memory.jsonl`` (plan §3.5), and each
segment reloads the model, so a slope fitted across a restart is meaningless.  These tests pin the
two behaviours the leak diagnosis (plan §4, §13 row 8) depends on.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analysis.m10_memwatch_report import (
    GIB,
    report,
    split_segments,
    steps_from_ledger,
)


def _sample(minute: int, anon_gib: float, driver_pid: str, step: int = 0) -> dict:
    at = (datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "at": at,
        "step": step,
        "anon": int(anon_gib * GIB),
        "cgroup_peak": int(80 * GIB),
        "oom_kill": 0,
        "processes": {
            f"ray::E7DynamicF:{driver_pid}": {"Pss_Anon": 1.0 + 0.01 * minute, "Rss": 2.0},
            f"ray::WorkerDict:{int(driver_pid) + 1}": {"Pss_Anon": 5.0 + 0.5 * minute, "Rss": 24.0},
        },
    }


def test_segments_split_at_a_driver_restart() -> None:
    rows = [_sample(m, 40 + 0.2 * m, "100", step=m) for m in range(6)]
    rows += [_sample(m, 39 + 0.2 * (m - 10), "200", step=m - 4) for m in range(10, 16)]
    segments = split_segments(rows)
    assert [len(s) for s in segments] == [6, 6]
    assert all(_sample_pid(r) == "100" for r in segments[0])
    assert all(_sample_pid(r) == "200" for r in segments[1])
    # A single uninterrupted run is one segment.
    assert len(split_segments(rows[:6])) == 1


def _sample_pid(row: dict) -> str:
    return next(key.split(":")[-1] for key in row["processes"] if key.startswith("ray::E7DynamicF"))


def test_headline_slope_comes_from_one_segment_not_the_pool() -> None:
    # Segment 1: steps 1-5 at +0.1 GiB/step.  Segment 2 restarts lower and rises at the same rate.
    rows = [_sample(m, 40 + 0.1 * m, "100", step=m) for m in range(1, 6)]
    rows += [_sample(20 + m, 38 + 0.1 * m, "200", step=5 + m) for m in range(1, 6)]
    payload = report(rows, min_samples=2)
    cg = payload["cgroup"]
    assert cg["restarts"] == 1 and len(cg["segments"]) == 2
    assert cg["anon_gib_per_step"] == pytest.approx(0.1)
    # Pooling across the restart would report a much smaller (here even negative) slope.
    assert cg["anon_gib_per_step_pooled"] != pytest.approx(0.1, abs=0.01)
    assert cg["slope_from_segment_samples"] == 4  # the segment's first sample is its startup baseline
    assert cg["anon_gib_peak"] == pytest.approx(40.5)


def test_steps_recovered_from_ledger_timestamps_when_sampler_field_is_stale(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    marks = [
        {"effective_step": 1, "recorded_at": "2026-09-09T12:03:00+00:00"},
        {"effective_step": 2, "recorded_at": "2026-09-09T12:07:00+00:00"},
    ]
    (run / "e7_ledger.jsonl").write_text("".join(json.dumps(m) + "\n" for m in marks), encoding="utf-8")
    rows = [_sample(m, 40.0, "100", step=1 if m >= 3 else 0) for m in (1, 4, 5, 8)]
    repaired = steps_from_ledger(rows, run)
    assert [r["step"] for r in repaired] == [0, 1, 1, 2]
    assert all(r["step_source"] == "ledger_timestamp" for r in repaired)
    # Samples that already reach the ledger's last step are left untouched.
    fresh = [_sample(m, 40.0, "100", step=2) for m in (1, 8)]
    assert steps_from_ledger(fresh, run) == fresh
    # No ledger on disk: rows pass through unchanged.
    assert steps_from_ledger(rows, tmp_path / "missing") == rows


def test_resumed_segment_startup_is_excluded_from_the_slope() -> None:
    """A resumed segment reads the previous segment's step during model load; those samples must go."""
    rows = [_sample(m, 40 + 0.1 * m, "100", step=m) for m in range(1, 6)]
    # Segment 2 reloads: four samples still report step 5 while anon climbs 17 -> 39 GiB.
    rows += [_sample(20 + m, 17 + 7 * m, "200", step=5) for m in range(4)]
    rows += [_sample(25 + m, 39 + 0.1 * m, "200", step=6 + m) for m in range(4)]
    payload = report(rows, min_samples=2)
    cg = payload["cgroup"]
    assert cg["restarts"] == 1
    second = cg["segments"][1]
    assert second["steady_state_samples"] == 4 and second["anon_gib_steady_first"] == pytest.approx(39.0)
    assert second["anon_gib_per_step"] == pytest.approx(0.1)
    assert cg["anon_gib_per_step"] == pytest.approx(0.1)


def test_teardown_samples_after_the_driver_exits_are_dropped() -> None:
    """memwatch samples once more after the run ends; anon collapses there and must not be fitted."""
    rows = [_sample(m, 40 + 0.5 * m, "100", step=m) for m in range(1, 7)]
    teardown = _sample(8, 4.0, "100", step=6)
    teardown["processes"] = {"sglang::schedul:999": {"Pss_Anon": 0.1, "Rss": 0.2}}  # driver gone
    rows.append(teardown)
    payload = report(rows, min_samples=2)
    cg = payload["cgroup"]
    assert cg["restarts"] == 0
    assert cg["segments"][0]["samples"] == 6
    assert cg["segments"][0]["anon_gib_steady_last"] == pytest.approx(43.0)
    assert cg["anon_gib_per_step"] == pytest.approx(0.5)
    # The cgroup peak still reflects every sample, including the teardown one.
    assert cg["anon_gib_peak"] == pytest.approx(43.0)


def test_process_ranking_orders_by_pss_anon_growth() -> None:
    rows = [_sample(m, 40 + 0.1 * m, "100", step=m) for m in range(1, 8)]
    payload = report(rows, min_samples=2)
    ranking = payload["process_ranking"]
    assert ranking[0]["process"].startswith("ray::WorkerDict")
    assert ranking[0]["pss_anon_gib_delta"] > ranking[1]["pss_anon_gib_delta"]
    assert ranking[0]["pss_anon_gib_per_step"] == pytest.approx(0.5)
