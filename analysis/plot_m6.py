"""Render the preregistered M6 E3/E5 comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)


def plot_m6(analysis_path: Path, output_path: Path) -> None:
    payload: dict[str, Any] = json.loads(analysis_path.read_text(encoding="utf-8"))
    e3 = payload["runs"]["e3"]
    e5 = payload["runs"]["e5"]
    steps = [int(value) for value in e3["validation_curve"]]
    e3_curve = [e3["validation_curve"][str(step)] for step in steps]
    e5_curve = [e5["validation_curve"][str(step)] for step in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=160)
    ax1.plot(steps, e3_curve, color="#2a78d6", marker="o", linewidth=2, label="E3 trajectory GRPO")
    ax1.plot(steps, e5_curve, color="#d95f02", marker="o", linewidth=2, label="E5 turn credit")
    ax1.set_title("A. Fixed MATH500-100 pass@1")
    ax1.set_xlabel("actor update")
    ax1.set_ylabel("greedy pass@1")
    ax1.set_ylim(0.55, 0.80)
    ax1.legend(frameon=False, fontsize=9)
    _style(ax1)

    metric_specs = [
        ("mean_tool_calls", "mean calls"),
        ("repeated_code_trajectory_rate", "repeated code"),
        ("truncated_rate", "truncated"),
        ("invalid_rate", "invalid"),
    ]
    x = list(range(len(metric_specs)))
    width = 0.36
    e3_values = [e3["behavior"][key] for key, _ in metric_specs]
    e5_values = [e5["behavior"][key] for key, _ in metric_specs]
    ax2.bar([value - width / 2 for value in x], e3_values, width, color="#2a78d6", label="E3")
    ax2.bar([value + width / 2 for value in x], e5_values, width, color="#d95f02", label="E5")
    ax2.set_xticks(x, [label for _, label in metric_specs], rotation=18, ha="right")
    ax2.set_title("B. Training-trajectory behavior")
    ax2.set_ylabel("mean / trajectory rate")
    ax2.legend(frameon=False, fontsize=9)
    _style(ax2)

    fig.suptitle("M6: earlier learning, tied final accuracy, substantially more tool use", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    plot_m6(args.analysis_path, args.output_path)


if __name__ == "__main__":
    main()
