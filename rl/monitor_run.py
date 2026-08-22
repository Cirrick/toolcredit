"""One-shot M4 run monitor and artifact summarizer.

This command does no polling.  Invoke it at sparse checkpoints while the tmux
job runs; it merges native TensorBoard scalars with trajectory-level metrics
from veRL's rollout JSONL and refreshes metrics.json / summary.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def read_native_scalars(tensorboard_dir: Path) -> dict[str, list[dict[str, float | int]]]:
    merged: dict[str, dict[int, float]] = defaultdict(dict)
    for event_file in sorted(tensorboard_dir.rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if tag.startswith("m4/"):
                continue
            for event in accumulator.Scalars(tag):
                merged[tag][event.step] = event.value
    return {
        tag: [{"step": step, "value": values[step]} for step in sorted(values)]
        for tag, values in sorted(merged.items())
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def read_rollout_metrics(prediction_dir: Path) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = {}
    for path in sorted(prediction_dir.glob("*.jsonl"), key=lambda item: int(item.stem)):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if not rows:
            continue
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups[str(row["sample_id"])].append(float(row["acc"]))
        step_metrics = {
            "reward/mean": _mean([float(row["score"]) for row in rows]),
            "n_tool_calls/mean": _mean([float(row["n_tool_calls"]) for row in rows]),
            "tool_error_rate": _mean([float(row["tool_error_rate"]) for row in rows]),
            "tool_parse_error_rate": _mean(
                [float(float(row.get("tool_parse_errors", 0.0)) > 0.0) for row in rows]
            ),
            "invalid_format_rate": 1.0 - _mean([float(row["format_ok"]) for row in rows]),
            "invalid_rate": _mean([float(row["invalid"]) for row in rows]),
            "truncated_rate": _mean([float(row["truncated"]) for row in rows]),
            "group_all_correct_frac": _mean([float(all(value == 1.0 for value in group)) for group in groups.values()]),
            "group_all_wrong_frac": _mean([float(all(value == 0.0 for value in group)) for group in groups.values()]),
        }
        if all("original_tool_return_token_count" in row for row in rows):
            policy_tokens = [float(row["original_policy_token_count"]) for row in rows]
            tool_tokens = [float(row["original_tool_return_token_count"]) for row in rows]
            loss_tokens = [float(row["nomask_loss_token_count"]) for row in rows]
            total_loss_tokens = sum(loss_tokens)
            step_metrics.update(
                {
                    "mask/original_policy_tokens_mean": _mean(policy_tokens),
                    "mask/tool_return_tokens_mean": _mean(tool_tokens),
                    "mask/nomask_loss_tokens_mean": _mean(loss_tokens),
                    "mask/tool_return_loss_fraction": (
                        sum(tool_tokens) / total_loss_tokens if total_loss_tokens else 0.0
                    ),
                }
            )
        if all("base_score" in row for row in rows):
            step_metrics.update(
                {
                    "reward/base_score_mean": _mean([float(row["base_score"]) for row in rows]),
                    "reward/exec_success_fraction_mean": _mean(
                        [float(row["exec_success_fraction"]) for row in rows]
                    ),
                    "reward/exec_bonus_mean": _mean([float(row["exec_bonus"]) for row in rows]),
                    "reward/budget_penalty_mean": _mean(
                        [float(row["budget_penalty"]) for row in rows]
                    ),
                    "n_tool_success/mean": _mean([float(row["n_tool_success"]) for row in rows]),
                }
            )
        metrics[int(path.stem)] = step_metrics
    return metrics


def _latest(native: dict[str, list[dict[str, float | int]]], tag: str) -> float | None:
    points = native.get(tag, [])
    return float(points[-1]["value"]) if points else None


def _latest_matching(native: dict[str, list[dict[str, float | int]]], needle: str) -> float | None:
    candidates = [tag for tag in native if needle in tag]
    if not candidates:
        return None
    return _latest(native, sorted(candidates)[0])


def write_aux_tensorboard(tensorboard_dir: Path, rollout: dict[int, dict[str, float]]) -> None:
    if not rollout:
        return
    with SummaryWriter(str(tensorboard_dir / "m4_aux")) as writer:
        for step, values in rollout.items():
            for key, value in values.items():
                writer.add_scalar(f"m4/{key}", value, step)


def summarize(run_dir: Path) -> dict[str, Any]:
    native = read_native_scalars(run_dir / "tensorboard")
    rollout = read_rollout_metrics(run_dir / "predictions/train")
    write_aux_tensorboard(run_dir / "tensorboard", rollout)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "unknown"}
    latest_step = max(rollout, default=int(_latest(native, "training/global_step") or 0))
    latest_rollout = rollout.get(latest_step, {})
    headline = {
        "step": latest_step,
        "reward_mean": latest_rollout.get("reward/mean", _latest(native, "critic/score/mean")),
        "eval_pass_at_1": _latest_matching(native, "val-core/toolcredit_math500/acc/mean@1"),
        "actor_kl": _latest(native, "actor/ppo_kl"),
        "actor_entropy": _latest(native, "actor/entropy"),
        "response_length_mean": _latest(native, "response_length/mean"),
        **latest_rollout,
    }
    payload = {"status": status, "headline": headline, "native": native, "rollout": rollout}
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# {run_dir.name}",
        "",
        f"- 状态：{status.get('state', 'unknown')}",
        f"- 最新训练 step：{latest_step}",
    ]
    for key, value in headline.items():
        if key != "step" and value is not None:
            rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"- `{key}`：{rendered}")
    lines += [
        "",
        "## 产物",
        "",
        "- `resolved_config.yaml`：实际训练配置",
        "- `checkpoints/`：veRL checkpoint 与恢复点",
        "- `predictions/`：逐 step 训练与验证 JSONL",
        "- `tensorboard/`：原生曲线与 `m4/*` 辅助监控指标",
        "- `metrics.json`：本摘要的机器可读版本",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.run_dir.resolve())
    print(json.dumps({"status": payload["status"], "headline": payload["headline"]}, indent=2))


if __name__ == "__main__":
    main()
