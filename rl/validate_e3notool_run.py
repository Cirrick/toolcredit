"""Fail-closed gates for M9 / E3-NoTool runs: smoke gate and formal completion (plans/M9.md §4, §7).

The E3-NoTool arm carries no turn-credit ledger, so there is no ``turn_credit_audit`` block
and no trajectory UID to check.  The structural guarantees are proved instead from the
rollout JSONL veRL persists for every trajectory:

* the decoded prompt contains no tool schema at all (plan §7 step 2 hard stop),
* every trajectory reports zero tool calls, zero tool errors and zero parse errors,
* per-step ``tool_call`` tag emission is recorded as a curve (plan §4 gate: > 5% after
  step 50 pauses the run),
* each step is 64 complete prompt groups x 8 rollouts.

Everything variant-agnostic (validation panel, checkpoint readability, SFT identity) is
reused from the frozen M6 validator ``rl/validate_e5_canonical_smoke.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf

from rl.analyze_e5 import _iter_rows
from rl.launch.e3_grpo_baseline import DEFAULT_CONFIG as E3_CONFIG
from rl.launch.e3_grpo_baseline import PROJECT_ROOT, compose_config
from rl.launch.e3notool_grpo import (
    AGENT_LOOP_NAME,
    E3_DIFF_PATHS,
    SMOKE_DIFF_PATHS,
    compose_e3notool_config,
    validate_e3notool_config,
    verl_source_hashes,
)
from rl.launch.e5v2_conserved_credit import diff_paths, resolved
from rl.monitor_run import read_native_scalars
from rl.validate_e5_canonical_smoke import (
    SFT_MODEL_SHA256,
    SFT_PATH,
    validate_checkpoint,
    validate_validation_files,
)

FORMAL_VALIDATION_STEPS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
ROLLOUTS_PER_STEP = 512
PROMPTS_PER_STEP = 64
ROLLOUTS_PER_PROMPT = 8
# Plan §4: the start checkpoint still emits hermes tags out of habit; RL is expected to
# wash them out.  Above this fraction after ``TOOL_TAG_GATE_FROM_STEP`` the run pauses.
TOOL_TAG_RATE_LIMIT = 0.05
TOOL_TAG_GATE_FROM_STEP = 50

TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
# Markers that can only appear in a prompt if a tool schema was injected.
PROMPT_TOOL_SCHEMA_MARKERS = ("<tools>", "</tools>", "code_interpreter", "# Tools", "<tool_call>")
OTHER_HERMES_TAGS = ("<tool_response>", "</tool_response>", "<tools>", "</tools>")


# --------------------------------------------------------------------------- #
# text-level detectors (plan §3 ``tool_tag_*`` fields, plan §6 fixture N)
# --------------------------------------------------------------------------- #


def tool_tag_stats(text: str) -> dict[str, Any]:
    """Count hermes ``tool_call`` tags in a response; detection never changes scoring.

    A complete ``<tool_call>...</tool_call>`` block counts once.  An unpaired opening or
    closing tag also counts once, so a truncated imitation is not silently missed.
    ``other_hermes_tag_count`` records ``<tool_response>``/``<tools>`` imitations
    separately; they are reported but excluded from the gated rate.
    """
    body = text or ""
    complete = len(TOOL_CALL_BLOCK_RE.findall(body))
    remainder = TOOL_CALL_BLOCK_RE.sub("", body)
    strays = remainder.count("<tool_call>") + remainder.count("</tool_call>")
    count = complete + strays
    return {
        "tool_tag_emitted": count > 0,
        "tool_tag_count": count,
        "complete_tool_call_blocks": complete,
        "unpaired_tool_call_tags": strays,
        "other_hermes_tag_count": sum(body.count(tag) for tag in OTHER_HERMES_TAGS),
    }


def prompt_tool_schema_markers(text: str) -> list[str]:
    """Every tool-schema marker present in a decoded prompt; empty means schema-free."""
    body = text or ""
    return [marker for marker in PROMPT_TOOL_SCHEMA_MARKERS if marker in body]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def _validate_postrun_config(config: Any, completed_run_dir: Path) -> None:
    if not completed_run_dir.is_dir():
        raise FileNotFoundError(f"completed run directory does not exist: {completed_run_dir}")
    sentinel = completed_run_dir.parent / f".{completed_run_dir.name}.postrun-config-validation-target"
    if sentinel.exists():
        raise FileExistsError(f"post-run validation sentinel already exists: {sentinel}")
    validate_e3notool_config(config, sentinel, resume=False)


def validate_resolved_config_notool(run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    actual_config = OmegaConf.load(run_dir / "resolved_config.yaml")
    _validate_postrun_config(actual_config, run_dir)
    actual = resolved(actual_config)
    expected = resolved(compose_e3notool_config(run_dir.name, smoke=smoke))
    drift = sorted(diff_paths(actual, expected))
    if drift:
        raise ValueError(f"resolved E3-NoTool config differs from the approved composition: {drift}")
    e3 = resolved(compose_config(E3_CONFIG, run_dir.name, smoke=False))
    observed = diff_paths(e3, actual)
    allowed = set(E3_DIFF_PATHS) | (SMOKE_DIFF_PATHS if smoke else set())
    if observed != allowed:
        raise ValueError(f"E3-control diff gate failed: {sorted(observed)} != {sorted(allowed)}")
    if Path(str(actual_config.actor_rollout_ref.model.path)).resolve() != SFT_PATH:
        raise ValueError("E3-NoTool did not start from the frozen SFT checkpoint")
    if actual_config.trainer.resume_from_path is not None:
        raise ValueError("fresh E3-NoTool run has a non-null resume path")
    return {"exact_config_match": True, "e3_control_diff_paths": sorted(observed)}


# --------------------------------------------------------------------------- #
# rollout-level gates
# --------------------------------------------------------------------------- #


def _step_paths(run_dir: Path, steps: Iterable[int]) -> list[Path]:
    paths = []
    for step in steps:
        path = run_dir / "predictions/train" / f"{step}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing train step file: {path}")
        paths.append(path)
    return paths


def validate_row_is_tool_free(row: dict[str, Any]) -> dict[str, Any]:
    """Every per-trajectory tool field must be absent or exactly zero (plan §4)."""
    for field in ("n_tool_calls", "tool_error_rate", "tool_parse_errors", "truncated"):
        value = float(row.get(field, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"NaN/Inf rollout field {field}")
        if value != 0.0:
            raise ValueError(f"E3-NoTool trajectory reports nonzero {field}={value}")
    for field in ("tool_call_counts", "tool_success_count", "tool_error_count", "tool_parse_error_count"):
        if field in row and float(row[field]) != 0.0:
            raise ValueError(f"E3-NoTool trajectory carries a nonzero {field}")
    for field in ("score", "acc", "invalid", "format_ok"):
        if field in row and not math.isfinite(float(row[field])):
            raise ValueError(f"NaN/Inf rollout field {field}")
    markers = prompt_tool_schema_markers(str(row.get("input", "")))
    if markers:
        raise ValueError(f"E3-NoTool prompt contains tool schema markers: {markers}")
    return tool_tag_stats(str(row.get("output", "")))


def validate_rollouts_notool(run_dir: Path, expected_steps: int) -> dict[str, Any]:
    """Completeness, group structure, tool-free invariants and the tag curve."""
    train_dir = run_dir / "predictions/train"
    paths = sorted(train_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    actual_steps = [int(path.stem) for path in paths]
    if actual_steps != list(range(1, expected_steps + 1)):
        raise ValueError(f"train step files are incomplete: {actual_steps}")

    trajectories = 0
    tagged = 0
    tags = 0
    other_tags = 0
    correct = 0
    invalid = 0
    tag_curve: dict[str, float] = {}
    for path in paths:
        prompt_counts: Counter[str] = Counter()
        rows_in_step = 0
        tagged_in_step = 0
        for _, row in _iter_rows([path]):
            stats = validate_row_is_tool_free(row)
            prompt_counts[str(row["sample_id"])] += 1
            rows_in_step += 1
            trajectories += 1
            tagged_in_step += int(stats["tool_tag_emitted"])
            tags += int(stats["tool_tag_count"])
            other_tags += int(stats["other_hermes_tag_count"])
            correct += int(float(row["acc"]) == 1.0)
            invalid += int(float(row.get("invalid", 0.0)) > 0.0)
        if rows_in_step != ROLLOUTS_PER_STEP:
            raise ValueError(f"step {path.stem} contains {rows_in_step} instead of {ROLLOUTS_PER_STEP} rollouts")
        if len(prompt_counts) != PROMPTS_PER_STEP or set(prompt_counts.values()) != {ROLLOUTS_PER_PROMPT}:
            raise ValueError(
                f"step {path.stem} is not {PROMPTS_PER_STEP} complete groups x {ROLLOUTS_PER_PROMPT} rollouts"
            )
        tagged += tagged_in_step
        tag_curve[path.stem] = tagged_in_step / rows_in_step

    native = read_native_scalars(run_dir / "tensorboard")
    scalar_count = 0
    for points in native.values():
        for point in points:
            scalar_count += 1
            if not math.isfinite(float(point["value"])):
                raise ValueError("NaN/Inf TensorBoard scalar")
    return {
        "steps": actual_steps,
        "trajectory_count": trajectories,
        "expected_trajectory_count": expected_steps * ROLLOUTS_PER_STEP,
        "prompt_groups_per_step": PROMPTS_PER_STEP,
        "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
        "tool_call_counts_all_zero": True,
        "prompts_free_of_tool_schema": True,
        "accuracy": correct / trajectories,
        "invalid_rate": invalid / trajectories,
        "tool_tag_trajectories": tagged,
        "tool_tag_trajectory_rate": tagged / trajectories,
        "tool_tag_total": tags,
        "other_hermes_tag_total": other_tags,
        "tool_tag_rate_by_step": tag_curve,
        "tensorboard_finite_scalar_count": scalar_count,
    }


def tool_tag_rate_by_step(run_dir: Path, steps: Iterable[int]) -> dict[str, float]:
    """Fraction of rollouts emitting a ``<tool_call>`` tag, per training step."""
    rows: Counter[str] = Counter()
    tagged: Counter[str] = Counter()
    for _, row in _iter_rows(_step_paths(run_dir, steps)):
        step = str(int(row["step"]))
        rows[step] += 1
        tagged[step] += int(tool_tag_stats(str(row.get("output", "")))["tool_tag_emitted"])
    return {step: tagged[step] / rows[step] for step in sorted(rows, key=int)}


def tool_tag_gate(tag_curve: dict[str, float]) -> dict[str, Any]:
    """Plan §4: after step 50 a tag rate above 5% pauses the run for an audit."""
    violations = {
        step: rate
        for step, rate in tag_curve.items()
        if int(step) > TOOL_TAG_GATE_FROM_STEP and rate > TOOL_TAG_RATE_LIMIT
    }
    return {
        "limit": TOOL_TAG_RATE_LIMIT,
        "applies_after_step": TOOL_TAG_GATE_FROM_STEP,
        "violating_steps": dict(sorted(violations.items(), key=lambda item: int(item[0]))),
        "pause_required": bool(violations),
    }


def validation_tool_tag_curve(run_dir: Path) -> dict[str, Any]:
    """Tag emission on the fixed MATH500-100 greedy panel, per validation step."""
    validation_dir = run_dir / "predictions/validation"
    curve: dict[str, float] = {}
    for path in sorted(validation_dir.glob("*.jsonl"), key=lambda item: int(item.stem)):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tagged = sum(int(tool_tag_stats(str(row.get("output", "")))["tool_tag_emitted"]) for row in rows)
        curve[path.stem] = tagged / len(rows)
    return curve


def behavior_metrics(run_dir: Path, steps: Iterable[int]) -> dict[str, Any]:
    """Accuracy / format / tag behaviour over a window of training steps."""
    steps = list(steps)
    count = correct = invalid = boxed_missing = tagged = 0
    output_chars = 0
    for _, row in _iter_rows(_step_paths(run_dir, steps)):
        count += 1
        correct += int(float(row["acc"]) == 1.0)
        invalid += int(float(row.get("invalid", 0.0)) > 0.0)
        boxed_missing += int(float(row.get("format_ok", 0.0)) == 0.0)
        output = str(row.get("output", ""))
        output_chars += len(output)
        tagged += int(tool_tag_stats(output)["tool_tag_emitted"])
    if count == 0:
        raise ValueError("no rollouts in the requested step window")
    return {
        "steps": [min(steps), max(steps)],
        "rows": count,
        "accuracy": correct / count,
        "invalid_rate": invalid / count,
        "boxed_missing_rate": boxed_missing / count,
        "tool_tag_trajectory_rate": tagged / count,
        "mean_response_characters": output_chars / count,
    }


def tensorboard_mechanism_metrics(run_dir: Path) -> dict[str, Any]:
    """Plan §4 mechanism scalars: length, truncation, entropy, KL."""
    native = read_native_scalars(run_dir / "tensorboard")
    wanted = (
        "response_length/mean",
        "response_length/clip_ratio",
        "actor/entropy",
        "actor/kl_loss",
        "critic/score/mean",
    )
    summary: dict[str, Any] = {}
    for tag in wanted:
        points = native.get(tag)
        if not points:
            continue
        summary[tag] = {
            "first": points[0],
            "last": points[-1],
            "point_count": len(points),
        }
    return summary


def zero_variance_group_fraction(run_dir: Path, steps: Iterable[int]) -> dict[str, Any]:
    """Plan §4: share of GRPO groups whose eight scores are identical (E3 reference 44%)."""
    groups: dict[tuple[int, str], list[float]] = {}
    for _, row in _iter_rows(_step_paths(run_dir, steps)):
        groups.setdefault((int(row["step"]), str(row["sample_id"])), []).append(float(row["score"]))
    if not groups:
        raise ValueError("no rollouts in the requested step window")
    zero = sum(1 for scores in groups.values() if len(set(scores)) == 1)
    return {"groups": len(groups), "zero_variance_groups": zero, "zero_variance_fraction": zero / len(groups)}


# --------------------------------------------------------------------------- #
# prompt audit (plan §7 step 2: ten decoded prompts, manually reviewed, persisted)
# --------------------------------------------------------------------------- #


def write_prompt_audit(run_dir: Path, steps: Iterable[int], count: int = 10) -> Path:
    """Persist ``count`` decoded prompts, stable-ordered by sample id, for manual review."""
    seen: dict[str, dict[str, Any]] = {}
    for _, row in _iter_rows(_step_paths(run_dir, steps)):
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            continue
        prompt = str(row.get("input", ""))
        seen[sample_id] = {
            "step": int(row["step"]),
            "sample_id": sample_id,
            "prompt_characters": len(prompt),
            "tool_schema_markers": prompt_tool_schema_markers(prompt),
            "prompt": prompt,
        }
    samples = [seen[key] for key in sorted(seen)][:count]
    if len(samples) < count:
        raise ValueError(f"only {len(samples)} distinct prompts available for the audit")
    offenders = [item["sample_id"] for item in samples if item["tool_schema_markers"]]
    if offenders:
        raise ValueError(f"decoded prompts still carry a tool schema: {offenders}")
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    jsonl = analysis_dir / "prompt_audit.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    lines = [
        "# E3-NoTool prompt audit",
        "",
        f"run: `{run_dir.name}` · {len(samples)} distinct decoded prompts · no tool schema markers found",
        "",
        "Markers searched for: " + ", ".join(f"`{marker}`" for marker in PROMPT_TOOL_SCHEMA_MARKERS),
        "",
    ]
    for index, sample in enumerate(samples, start=1):
        lines += [
            f"## Prompt {index} · step {sample['step']} · `{sample['sample_id']}` "
            f"({sample['prompt_characters']} chars)",
            "",
            "```",
            sample["prompt"],
            "```",
            "",
        ]
    (analysis_dir / "prompt_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return jsonl


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def smoke_gate(run_dir: Path, launch_ledger: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"smoke did not exit normally: {status}")
    ledger = json.loads(launch_ledger.read_text(encoding="utf-8"))
    if ledger.get("run_name") != run_dir.name or ledger.get("resumed") is not False:
        raise ValueError("fresh-launch ledger does not prove resumed=false")
    if Path(str(ledger.get("checkpoint"))).resolve() != SFT_PATH:
        raise ValueError("fresh-launch ledger does not identify the frozen SFT checkpoint")
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("checkpoint_model_sha256") != SFT_MODEL_SHA256:
        raise ValueError("smoke preflight SFT model hash drift")
    if preflight.get("agent_loop") != AGENT_LOOP_NAME:
        raise ValueError("smoke preflight did not record the native single-turn agent loop")

    config = validate_resolved_config_notool(run_dir, smoke=True)
    validation = validate_validation_files(run_dir, [0, 5])
    rollouts = validate_rollouts_notool(run_dir, 5)
    audit_path = write_prompt_audit(run_dir, range(1, 6))
    retained = (
        [path.name for path in (run_dir / "checkpoints").glob("global_step_*")]
        if (run_dir / "checkpoints").exists()
        else []
    )
    if retained:
        raise ValueError(f"smoke retained checkpoints although save_freq=-1: {retained}")

    report = {
        "schema_version": "toolcredit_e3notool_smoke_gate_v1",
        "run_name": run_dir.name,
        "gate_passed": True,
        "fresh_launch": {"resumed": False, "checkpoint": str(SFT_PATH), "model_sha256": SFT_MODEL_SHA256},
        "verl_source_hashes": verl_source_hashes(),
        "resolved_config": config,
        "validation": validation,
        "rollouts": rollouts,
        "tool_tag_gate": tool_tag_gate(rollouts["tool_tag_rate_by_step"]),
        "validation_tool_tag_rate_by_step": validation_tool_tag_curve(run_dir),
        "behavior": behavior_metrics(run_dir, range(1, 6)),
        "zero_variance_groups": zero_variance_group_fraction(run_dir, range(1, 6)),
        "tensorboard": tensorboard_mechanism_metrics(run_dir),
        "prompt_audit": str(audit_path.relative_to(run_dir)),
        "retained_checkpoints": retained,
        "disk_free_gib": round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 2),
        "gpu_visible": torch.cuda.is_available() and torch.cuda.device_count() == 1,
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    (analysis_dir / "smoke_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "evidence_classification.json").write_text(
        json.dumps(
            {
                "classification": "e3notool_five_step_smoke",
                "smoke_gate_passed": True,
                "gate_report": "analysis/smoke_gate.json",
                "resume_allowed": False,
                "checkpoint_retained": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def formal_completion(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"formal E3-NoTool run is not completed: {status}")
    config = validate_resolved_config_notool(run_dir, smoke=False)
    validation = validate_validation_files(run_dir, FORMAL_VALIDATION_STEPS)
    rollouts = validate_rollouts_notool(run_dir, 200)
    checkpoint = validate_checkpoint(run_dir, 200)
    recovery_dir = run_dir / "recovery"
    recovery_events = (
        sorted(str(path.relative_to(run_dir)) for path in recovery_dir.glob("resume_from_*") if path.is_dir())
        if recovery_dir.exists()
        else []
    )
    report = {
        "schema_version": "toolcredit_e3notool_formal_completion_gate_v1",
        "run_name": run_dir.name,
        "status": "completed",
        "effective_actor_updates": 200,
        "resolved_config": config,
        "verl_source_hashes": verl_source_hashes(),
        "fixed_panel_pass_at_1": validation["pass_at_1"],
        "rollouts": rollouts,
        "checkpoint": checkpoint,
        "tool_tag_gate": tool_tag_gate(rollouts["tool_tag_rate_by_step"]),
        "validation_tool_tag_rate_by_step": validation_tool_tag_curve(run_dir),
        "behavior_first_25": behavior_metrics(run_dir, range(1, 26)),
        "behavior_last_25": behavior_metrics(run_dir, range(176, 201)),
        "behavior_all_200": behavior_metrics(run_dir, range(1, 201)),
        "zero_variance_groups_first_25": zero_variance_group_fraction(run_dir, range(1, 26)),
        "zero_variance_groups_last_25": zero_variance_group_fraction(run_dir, range(176, 201)),
        "tensorboard": tensorboard_mechanism_metrics(run_dir),
        "recovery_events": recovery_events,
        "disk_free_gib": round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 2),
        "interpretation_performed": False,
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    (analysis_dir / "formal_completion_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("run_dir", type=Path)
    smoke.add_argument("--launch-ledger", type=Path, required=True)
    formal = sub.add_parser("formal")
    formal.add_argument("run_dir", type=Path)
    tags = sub.add_parser("tags")
    tags.add_argument("run_dir", type=Path)
    tags.add_argument("--start", type=int, default=1)
    tags.add_argument("--end", type=int, default=25)
    prompts = sub.add_parser("prompts")
    prompts.add_argument("run_dir", type=Path)
    prompts.add_argument("--start", type=int, default=1)
    prompts.add_argument("--end", type=int, default=5)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.command == "smoke":
        report = smoke_gate(run_dir, args.launch_ledger.resolve())
        print(
            json.dumps(
                {
                    "run_name": report["run_name"],
                    "gate_passed": True,
                    "tool_tag_trajectory_rate": report["rollouts"]["tool_tag_trajectory_rate"],
                },
                indent=2,
            )
        )
    elif args.command == "formal":
        report = formal_completion(run_dir)
        print(json.dumps({"run_name": report["run_name"], "status": report["status"]}, indent=2))
    elif args.command == "tags":
        print(json.dumps(tool_tag_rate_by_step(run_dir, range(args.start, args.end + 1)), indent=2))
    elif args.command == "prompts":
        print(str(write_prompt_audit(run_dir, range(args.start, args.end + 1))))


if __name__ == "__main__":
    main()
