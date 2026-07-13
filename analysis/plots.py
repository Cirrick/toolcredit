"""Plotting for reports. M1: tool-gain probe figure (reports/assets/01_tool_gain.png)."""

import json
import os
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_DIR, "reports", "assets")

# dataviz palette: categorical slot 1/2 for the two arms, status red for error state
C_COT = "#2a78d6"
C_TIR = "#1baf7a"
C_ERR = "#e34948"
INK, INK2 = "#333333", "#666666"


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def style_axis(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)


def conditional_pass_rates(tir_rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """P(correct | tool usage outcome) per level: no_tool / tool_ok / tool_err."""
    acc: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in tir_rows:
        key = "no_tool" if r["n_tool_calls"] == 0 else ("tool_err" if r["n_tool_errors"] > 0 else "tool_ok")
        acc[r["level"]][key][1] += 1
        acc[r["level"]][key][0] += r["correct"]
    return {lv: {k: v[0] / v[1] for k, v in d.items()} for lv, d in acc.items()}


def plot_tool_gain(probe_dir: str, out_path: str) -> None:
    metrics = json.load(open(os.path.join(probe_dir, "metrics.json")))
    tir_rows = read_jsonl(os.path.join(probe_dir, "trajectories_tir.jsonl"))
    levels = sorted(int(lv) for lv in metrics["levels"])
    cot = [metrics["levels"][str(lv)]["cot"]["pass1"] for lv in levels]
    tir = [metrics["levels"][str(lv)]["tir"]["pass1"] for lv in levels]
    cond = conditional_pass_rates(tir_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    fig.patch.set_facecolor("white")

    # Panel A: zero-shot pass@1 per level, two arms
    ax1.plot(levels, cot, color=C_COT, linewidth=2, marker="o", markersize=7, label="CoT")
    ax1.plot(levels, tir, color=C_TIR, linewidth=2, marker="o", markersize=7, label="TIR (zero-shot)")
    for lv, y in zip(levels, cot):
        ax1.annotate(f"{y:.2f}", (lv, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8, color=INK)
    for lv, y in zip(levels, tir):
        ax1.annotate(f"{y:.2f}", (lv, y), textcoords="offset points", xytext=(0, -15),
                     ha="center", fontsize=8, color=INK)
    ax1.set_xticks(levels)
    ax1.set_xticklabels([f"L{lv}" for lv in levels])
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("pass@1 (temp 0.6, n=4)", color=INK, fontsize=10)
    ax1.set_title("A. Zero-shot tool gain: negative at every level", color=INK, fontsize=11)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    style_axis(ax1)

    # Panel B: TIR pass@1 decomposed by tool outcome
    keys = [("no_tool", "no tool use", "#9ec5f4"), ("tool_ok", "all tool calls ok", C_TIR), ("tool_err", "tool call errored", C_ERR)]
    width = 0.26
    for j, (key, label, color) in enumerate(keys):
        xs = [lv + (j - 1) * width for lv in levels]
        ys = [cond[lv].get(key, 0.0) for lv in levels]
        bars = ax2.bar(xs, ys, width=width * 0.92, color=color, label=label, zorder=2)
        for b, y in zip(bars, ys):
            ax2.annotate(f"{y:.2f}", (b.get_x() + b.get_width() / 2, y), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=7.5, color=INK)
    ax2.set_xticks(levels)
    ax2.set_xticklabels([f"L{lv}" for lv in levels])
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("P(correct | tool outcome)", color=INK, fontsize=10)
    ax2.set_title("B. Decomposition: successful tool use wins on L3-5", color=INK, fontsize=11)
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    style_axis(ax2)

    fig.suptitle("Qwen3-1.7B (non-thinking) - MATH train pool, 100 questions/level, temp 0.6, n=4", color=INK2, fontsize=10, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"figure -> {out_path}")


if __name__ == "__main__":
    plot_tool_gain(
        probe_dir=os.path.join(PROJECT_DIR, "data", "probe"),
        out_path=os.path.join(ASSETS_DIR, "01_tool_gain.png"),
    )
