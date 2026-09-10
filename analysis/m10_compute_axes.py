"""M10 / E7 compute-normalised comparison rules (plans/M10.md §9, Fixture T).

Two x-axes for the fixed-100 greedy validation curve:

* effective step (validation points 0/25/…/200 align directly), and
* cumulative rollout trajectories, where the E7 ledger maps every effective step to the number of
  trajectories generated so far and E3 is exactly ``512 x step``.

The frozen AUC rule: linearly interpolate the curve on the chosen axis, integrate over the common
budget interval ``[0, x_max]`` and divide by ``x_max``.  Both arms are evaluated on the same
interval so the numbers are comparable; E3's two-axis AUCs coincide because its trajectory axis
is a constant multiple of the step axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

E3_TRAJECTORIES_PER_STEP = 64 * 8
E3_TOTAL_STEPS = 200
E3_TOTAL_TRAJECTORIES = E3_TOTAL_STEPS * E3_TRAJECTORIES_PER_STEP


def interpolate(points: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear interpolation; ``x`` must lie within the curve's x-range."""
    if len(points) < 2:
        raise ValueError("interpolation needs at least two points")
    xs = [p[0] for p in points]
    if xs != sorted(xs) or len(set(xs)) != len(xs):
        raise ValueError("curve x values must be strictly increasing")
    if x < xs[0] or x > xs[-1]:
        raise ValueError(f"x={x} outside the curve range [{xs[0]}, {xs[-1]}]")
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise AssertionError("unreachable")


def normalized_auc(points: Iterable[tuple[float, float]], x_max: float, x_min: float = 0.0) -> float:
    """Trapezoid integral of the interpolated curve over ``[x_min, x_max]`` divided by its length."""
    curve = sorted((float(x), float(y)) for x, y in points)
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min")
    if curve[0][0] > x_min or curve[-1][0] < x_max:
        raise ValueError(f"curve [{curve[0][0]}, {curve[-1][0]}] does not cover [{x_min}, {x_max}]")
    knots = sorted({x for x, _ in curve if x_min < x < x_max} | {x_min, x_max})
    area = 0.0
    for left, right in zip(knots, knots[1:]):
        area += 0.5 * (interpolate(curve, left) + interpolate(curve, right)) * (right - left)
    return area / (x_max - x_min)


def step_curve(pass_at_1: dict[int, float]) -> list[tuple[float, float]]:
    return sorted((float(step), float(value)) for step, value in pass_at_1.items())


def trajectory_curve(pass_at_1: dict[int, float], trajectories_by_step: dict[int, int]) -> list[tuple[float, float]]:
    """Map validation steps to the cumulative-trajectory axis (step 0 -> 0 trajectories)."""
    points: list[tuple[float, float]] = []
    for step, value in pass_at_1.items():
        if step == 0:
            points.append((0.0, float(value)))
            continue
        if step not in trajectories_by_step:
            raise KeyError(f"no cumulative trajectory count for validation step {step}")
        points.append((float(trajectories_by_step[step]), float(value)))
    return sorted(points)


def e3_trajectories_by_step(max_step: int = E3_TOTAL_STEPS) -> dict[int, int]:
    return {step: step * E3_TRAJECTORIES_PER_STEP for step in range(1, max_step + 1)}


def read_validation_curve(run_dir: Path) -> dict[int, float]:
    validation = run_dir / "predictions/validation"
    curve: dict[int, float] = {}
    for path in sorted(validation.glob("*.jsonl"), key=lambda p: int(p.stem)):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty validation file {path}")
        curve[int(path.stem)] = sum(float(row["acc"]) for row in rows) / len(rows)
    return curve


def read_ledger_trajectories(run_dir: Path) -> dict[int, int]:
    ledger = run_dir / "e7_ledger.jsonl"
    out: dict[int, int] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["effective_step"])] = int(row["cumulative_trajectories"])
    return out


def compare_arms(
    e3_curve: dict[int, float],
    e7_curve: dict[int, float],
    e7_trajectories: dict[int, int],
    *,
    step_max: int = E3_TOTAL_STEPS,
    trajectory_max: int = E3_TOTAL_TRAJECTORIES,
) -> dict[str, Any]:
    e3_steps = step_curve(e3_curve)
    e7_steps = step_curve(e7_curve)
    e3_traj = trajectory_curve(e3_curve, e3_trajectories_by_step(max(e3_curve)))
    e7_traj = trajectory_curve(e7_curve, e7_trajectories)
    result = {
        "step_axis": {
            "x_max": step_max,
            "e3_auc": normalized_auc(e3_steps, step_max),
            "e7_auc": normalized_auc(e7_steps, step_max),
        },
        "trajectory_axis": {
            "x_max": trajectory_max,
            "e3_auc": normalized_auc(e3_traj, trajectory_max),
            "e7_auc": normalized_auc(e7_traj, trajectory_max),
        },
    }
    for axis in ("step_axis", "trajectory_axis"):
        result[axis]["delta_e7_minus_e3"] = result[axis]["e7_auc"] - result[axis]["e3_auc"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-run", type=Path, required=True)
    parser.add_argument("--e7-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = compare_arms(
        read_validation_curve(args.e3_run), read_validation_curve(args.e7_run), read_ledger_trajectories(args.e7_run)
    )
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
