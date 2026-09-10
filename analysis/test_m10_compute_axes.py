"""M10 Fixture T — the frozen AUC rule on the step and trajectory axes (plans/M10.md §7, §9)."""

from __future__ import annotations

import pytest

from analysis.m10_compute_axes import (
    E3_TOTAL_TRAJECTORIES,
    compare_arms,
    e3_trajectories_by_step,
    interpolate,
    normalized_auc,
    trajectory_curve,
)


def test_fixture_t_linear_interpolation_and_normalised_auc_match_hand_computation() -> None:
    # A ramp from 0.5 to 0.7 over [0, 100]: mean value 0.6.
    ramp = [(0, 0.5), (50, 0.6), (100, 0.7)]
    assert normalized_auc(ramp, 100) == pytest.approx(0.6)
    assert interpolate(ramp, 25) == pytest.approx(0.55)
    # Truncating to [0, 50] integrates only the first segment: mean 0.55.
    assert normalized_auc(ramp, 50) == pytest.approx(0.55)
    # A curve with a plateau: area = 0.5*25 + (0.5+0.8)/2*25 + 0.8*50 = 12.5 + 16.25 + 40 = 68.75 -> /100.
    plateau = [(0, 0.5), (25, 0.5), (50, 0.8), (100, 0.8)]
    assert normalized_auc(plateau, 100) == pytest.approx(0.6875)


def test_fixture_t_e3_step_and_trajectory_axis_aucs_are_identical() -> None:
    curve = {0: 0.60, 25: 0.65, 50: 0.68, 75: 0.70, 100: 0.71, 125: 0.72, 150: 0.72, 175: 0.73, 200: 0.73}
    step_auc = normalized_auc(sorted((float(k), v) for k, v in curve.items()), 200)
    traj_points = trajectory_curve(curve, e3_trajectories_by_step())
    traj_auc = normalized_auc(traj_points, E3_TOTAL_TRAJECTORIES)
    assert traj_auc == pytest.approx(step_auc)


def test_fixture_t_e7_trajectory_axis_uses_ledger_counts_and_common_interval() -> None:
    e3 = {0: 0.60, 100: 0.70, 200: 0.72}
    # E7 spends 2.5 batches per step: 200 steps = 256,000 trajectories; equal compute at step 80.
    e7 = {0: 0.60, 100: 0.74, 200: 0.75}
    e7_traj = {step: step * 64 * 8 * 25 // 10 for step in range(1, 201)}
    report = compare_arms(e3, e7, e7_traj)
    assert report["step_axis"]["delta_e7_minus_e3"] > 0
    # On the trajectory axis E7's step-100 point sits at 128,000 > 102,400, so only the
    # interpolated segment up to 102,400 counts: (0.60 + interp(102,400)) / 2.
    e7_at_budget = 0.60 + (0.74 - 0.60) * 102_400 / 128_000
    assert report["trajectory_axis"]["e7_auc"] == pytest.approx((0.60 + e7_at_budget) / 2)
    assert report["trajectory_axis"]["x_max"] == E3_TOTAL_TRAJECTORIES


def test_fixture_t_rejects_curves_that_do_not_cover_the_interval() -> None:
    with pytest.raises(ValueError):
        normalized_auc([(0, 0.5), (50, 0.6)], 100)
    with pytest.raises(ValueError):
        interpolate([(0, 0.5), (50, 0.6)], 60)
