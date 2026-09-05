"""Fail-closed gates for M8 / E5-v2 runs: smoke gate, ledger review samples, behavior gate,
per-step exposure and formal completion (plans/M8.md §7–§9).

Reuses the variant-agnostic v1 validators (validation panel, checkpoint, segmented wrapper
proofs, recovery overlays) and adds v2-specific audit checks: schema/adoption v2, exact
recomputation of the conserved corrections, per-trajectory conservation, zero-variance
identity and the §3.2 condition invariants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf

from rl.analyze_e4 import code_hash, extract_codes
from rl.analyze_e5 import _iter_rows
from rl.custom.turn_advantage_v2 import (
    ADOPTION_VERSION_V2,
    CONSERVATION_TOLERANCE,
    FROZEN_BETA,
    LEDGER_SCHEMA_VERSION_V2,
    VARIANT_CONSERVED_V2,
    conserved_turn_corrections,
)
from rl.launch.e3_grpo_baseline import DEFAULT_CONFIG as E3_CONFIG
from rl.launch.e3_grpo_baseline import PROJECT_ROOT, compose_config
from rl.launch.e5_turn_credit import validate_freeze_bundle
from rl.launch.e5v2_conserved_credit import (
    E3_CONTROL_DIFF_PATHS,
    PROOF_PREFIX,
    SMOKE_DIFF_PATHS,
    compose_e5v2_config,
    diff_paths,
    resolved,
    validate_e5v2_config,
)
from rl.validate_e5_canonical_smoke import (
    FREEZE_V2_SHA256,
    SFT_MODEL_SHA256,
    SFT_PATH,
    validate_checkpoint,
    validate_rollouts,
    validate_validation_files,
)

FORMAL_VALIDATION_STEPS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
EXPOSURE_FLOOR = 0.08
# Plan §8 step 5 / user stop rule: pause-and-audit thresholds at steps 50 and 100.
PAUSE_MEAN_CALLS = 2.0
PAUSE_FOUR_CALL = 0.15
# Plan §9.1 acceptance lines on the last 25 training steps.
PASS_LINES = {
    "mean_tool_calls": 1.4,
    "four_call_fraction": 0.06,
    "repeated_code_trajectory_rate": 0.04,
    "truncated_rate": 0.03,
}


# --------------------------------------------------------------------------- #
# audit-level checks
# --------------------------------------------------------------------------- #


def validate_audit_v2(audit: dict[str, Any]) -> dict[str, int]:
    required = {
        "schema_version",
        "adoption_version",
        "variant",
        "wrapper_proof",
        "prompt_uid",
        "trajectory_uid",
        "response_length_unpadded",
        "response_mask_sha256_uint8",
        "policy_token_count",
        "masked_tool_or_padding_token_count",
        "group_score_std",
        "informative_group",
        "eligible_turn_count",
        "gated",
        "treated",
        "conservation_residual",
        "turns",
    }
    missing = required - audit.keys()
    if missing:
        raise ValueError(f"E5-v2 audit missing fields: {sorted(missing)}")
    if audit["schema_version"] != LEDGER_SCHEMA_VERSION_V2:
        raise ValueError("unexpected ledger schema for E5-v2")
    if audit["adoption_version"] != ADOPTION_VERSION_V2:
        raise ValueError("unexpected adoption version for E5-v2")
    if audit["variant"] != VARIANT_CONSERVED_V2:
        raise ValueError("unexpected turn-credit variant")
    if not str(audit["wrapper_proof"]).startswith(PROOF_PREFIX):
        raise ValueError("E5-v2 Ray-driver wrapper proof is absent")

    response_length = int(audit["response_length_unpadded"])
    policy_count = int(audit["policy_token_count"])
    if policy_count <= 0 or response_length <= 0 or policy_count > response_length:
        raise ValueError("invalid response/policy token counts")
    informative = bool(audit["informative_group"])
    previous_end = 0
    policy_span_count = tool_span_count = 0
    base_advantage: float | None = None
    s_values: list[int] = []
    token_counts: list[int] = []
    eligible: list[bool] = []
    recorded: list[float] = []
    for expected_index, turn in enumerate(audit["turns"]):
        if int(turn["turn_index"]) != expected_index:
            raise ValueError("noncontiguous turn index")
        start, end = map(int, turn["policy_token_span"])
        if not 0 <= start < end <= response_length or start < previous_end:
            raise ValueError("invalid or overlapping policy token span")
        if int(turn["n_t"]) != end - start:
            raise ValueError("n_t does not equal the policy span length")
        policy_span_count += end - start
        previous_end = end
        tool_span = turn.get("tool_token_span")
        if tool_span is not None:
            tool_start, tool_end = map(int, tool_span)
            if not end <= tool_start < tool_end <= response_length:
                raise ValueError("invalid tool token span")
            tool_span_count += tool_end - tool_start
            previous_end = tool_end
        a_traj = float(turn["A_traj"])
        if base_advantage is None:
            base_advantage = a_traj
        elif not math.isclose(a_traj, base_advantage, abs_tol=1e-6):
            raise ValueError("native trajectory advantage differs across turns")
        correction = float(turn["correction"])
        if not math.isclose(float(turn["A_turn"]), a_traj + correction, abs_tol=1e-6):
            raise ValueError("A_turn does not equal A_traj + correction")
        s_t, s_v1 = int(turn["s_t"]), int(turn["s_t_v1"])
        if s_t not in {0, 1} or s_v1 not in {0, 1}:
            raise ValueError("s_t / s_t_v1 must be binary")
        if s_t == 1:
            if s_v1 != 1 or not turn["credit_eligible"] or turn["repeat_of_turn"] is not None or turn["after_boxed"]:
                raise ValueError("s_t=1 violates a v2 condition")
            if int(turn["execution_success"]) != 1 or int(turn["adopted"]) != 1:
                raise ValueError("s_t=1 without execution success and adoption")
        if s_v1 == 1 and s_t == 0 and turn["repeat_of_turn"] is None and not turn["after_boxed"]:
            raise ValueError("s_t dropped from v1 without a recorded v2 reason")
        if turn["repeat_of_turn"] is not None and not 0 <= int(turn["repeat_of_turn"]) < expected_index:
            raise ValueError("repeat_of_turn must point at an earlier turn")
        if not turn["credit_eligible"] and correction != 0.0:
            raise ValueError("ineligible turn received a nonzero correction")
        s_values.append(s_t)
        token_counts.append(end - start)
        eligible.append(bool(turn["credit_eligible"]))
        recorded.append(correction)
    if policy_span_count != policy_count:
        raise ValueError("policy spans do not conserve policy token count")
    if tool_span_count > int(audit["masked_tool_or_padding_token_count"]):
        raise ValueError("tool spans exceed masked token count")
    if int(audit["eligible_turn_count"]) != sum(eligible):
        raise ValueError("eligible_turn_count mismatch")

    expected, s_tilde, gated = conserved_turn_corrections(
        s_values, token_counts, eligible, beta=FROZEN_BETA, informative_group=informative
    )
    for value, reference in zip(recorded, expected, strict=True):
        if not math.isclose(value, reference, abs_tol=1e-6):
            raise ValueError("recorded correction does not match the frozen v2 formula")
    residual = sum(c * n for c, n in zip(recorded, token_counts, strict=True))
    if abs(residual) > CONSERVATION_TOLERANCE or abs(float(audit["conservation_residual"])) > CONSERVATION_TOLERANCE:
        raise ValueError("trajectory correction is not conserved")
    if bool(audit["gated"]) != gated:
        raise ValueError("gated flag mismatch")
    if bool(audit["treated"]) != any(value != 0.0 for value in recorded):
        raise ValueError("treated flag mismatch")
    if not informative and (base_advantage is None or not math.isclose(base_advantage, 0.0, abs_tol=1e-6)):
        raise ValueError("zero-variance group has nonzero native advantage")
    for turn in audit["turns"]:
        if turn["s_tilde"] is None and not gated:
            raise ValueError("ungated trajectory missing s_tilde")
        if not gated and not math.isclose(float(turn["s_tilde"]), float(s_tilde), abs_tol=1e-9):
            raise ValueError("s_tilde mismatch")
    return {"policy_span_tokens": policy_span_count, "tool_span_tokens": tool_span_count}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def _validate_postrun_config(config: Any, completed_run_dir: Path) -> None:
    if not completed_run_dir.is_dir():
        raise FileNotFoundError(f"completed run directory does not exist: {completed_run_dir}")
    sentinel = completed_run_dir.parent / f".{completed_run_dir.name}.postrun-config-validation-target"
    if sentinel.exists():
        raise FileExistsError(f"post-run validation sentinel already exists: {sentinel}")
    validate_e5v2_config(config, sentinel, resume=False)


def validate_resolved_config_v2(run_dir: Path, *, smoke: bool) -> dict[str, Any]:
    actual_config = OmegaConf.load(run_dir / "resolved_config.yaml")
    _validate_postrun_config(actual_config, run_dir)
    actual = resolved(actual_config)
    expected = resolved(compose_e5v2_config(run_dir.name, smoke=smoke))
    drift = sorted(diff_paths(actual, expected))
    if drift:
        raise ValueError(f"resolved E5-v2 config differs from the approved composition: {drift}")
    e3 = resolved(compose_config(E3_CONFIG, run_dir.name, smoke=False))
    observed = diff_paths(e3, actual)
    allowed = set(E3_CONTROL_DIFF_PATHS) | (SMOKE_DIFF_PATHS if smoke else set())
    if observed != allowed:
        raise ValueError(f"E3-control diff gate failed: {sorted(observed)} != {sorted(allowed)}")
    if Path(str(actual_config.actor_rollout_ref.model.path)).resolve() != SFT_PATH:
        raise ValueError("E5-v2 did not start from the frozen SFT checkpoint")
    if actual_config.trainer.resume_from_path is not None:
        raise ValueError("fresh E5-v2 run has a non-null resume path")
    return {"exact_config_match": True, "e3_control_diff_paths": sorted(observed)}


# --------------------------------------------------------------------------- #
# per-step behavior / exposure
# --------------------------------------------------------------------------- #


def _step_paths(run_dir: Path, steps: Iterable[int]) -> list[Path]:
    paths = []
    for step in steps:
        path = run_dir / "predictions/train" / f"{step}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing train step file: {path}")
        paths.append(path)
    return paths


def behavior_metrics(run_dir: Path, steps: Iterable[int]) -> dict[str, Any]:
    """Same definitions as rl/finalize_m6.py:_behavior_metrics, restricted to ``steps``."""
    steps = list(steps)
    count = calls = truncated = repeated = correct = invalid = 0
    histogram: Counter[int] = Counter()
    for _, row in _iter_rows(_step_paths(run_dir, steps)):
        count += 1
        row_calls = int(float(row["n_tool_calls"]))
        calls += row_calls
        histogram[min(row_calls, 4)] += 1
        truncated += int(float(row["truncated"]) > 0)
        invalid += int(float(row["invalid"]) > 0)
        correct += int(float(row["acc"]) == 1)
        hashes = [code_hash(code) for code in extract_codes(str(row["output"]))]
        repeated += int(len(set(hashes)) < len(hashes))
    if count == 0:
        raise ValueError("no rollouts in the requested step window")
    return {
        "steps": [min(steps), max(steps)],
        "rows": count,
        "accuracy": correct / count,
        "mean_tool_calls": calls / count,
        "four_call_fraction": histogram[4] / count,
        "call_fractions_0_1_2_3_4plus": {str(value): histogram[value] / count for value in range(5)},
        "repeated_code_trajectory_rate": repeated / count,
        "truncated_rate": truncated / count,
        "invalid_rate": invalid / count,
    }


def behavior_gate(run_dir: Path, upto_step: int, window: int = 25) -> dict[str, Any]:
    """Pause thresholds (mean calls > 2.0 or 4-call > 15%) plus §9.1 pass lines."""
    metrics = behavior_metrics(run_dir, range(max(1, upto_step - window + 1), upto_step + 1))
    pause = metrics["mean_tool_calls"] > PAUSE_MEAN_CALLS or metrics["four_call_fraction"] > PAUSE_FOUR_CALL
    pass_lines = {key: metrics[key] <= limit for key, limit in PASS_LINES.items()}
    return {
        "upto_step": upto_step,
        "metrics": metrics,
        "pause_thresholds": {"mean_tool_calls": PAUSE_MEAN_CALLS, "four_call_fraction": PAUSE_FOUR_CALL},
        "pause_required": pause,
        "pass_lines": PASS_LINES,
        "pass_line_results": pass_lines,
        "all_pass_lines_met": all(pass_lines.values()),
    }


def exposure(run_dir: Path, steps: Iterable[int]) -> dict[str, Any]:
    steps = list(steps)
    rows = treated = gated = informative_rows = multi = 0
    groups: dict[tuple[int, str], bool] = {}
    s1_v1 = s1_v2 = repeats = boxed = eligible = 0
    max_residual = 0.0
    per_step: dict[str, float] = {}
    step_rows: Counter[int] = Counter()
    step_treated: Counter[int] = Counter()
    for _, row in _iter_rows(_step_paths(run_dir, steps)):
        audit = row["turn_credit_audit"]
        step = int(row["step"])
        rows += 1
        step_rows[step] += 1
        treated += int(audit["treated"])
        step_treated[step] += int(audit["treated"])
        gated += int(audit["gated"])
        informative_rows += int(audit["informative_group"])
        multi += int(int(audit["eligible_turn_count"]) >= 2)
        groups[(step, str(audit["prompt_uid"]))] = bool(audit["informative_group"])
        max_residual = max(max_residual, abs(float(audit["conservation_residual"])))
        for turn in audit["turns"]:
            eligible += int(turn["credit_eligible"])
            s1_v1 += int(turn["s_t_v1"])
            s1_v2 += int(turn["s_t"])
            repeats += int(turn["repeat_of_turn"] is not None)
            boxed += int(bool(turn["after_boxed"]))
    for step in sorted(step_rows):
        per_step[str(step)] = step_treated[step] / step_rows[step]
    return {
        "steps": [min(steps), max(steps)],
        "trajectories": rows,
        "groups": len(groups),
        "informative_group_fraction": sum(groups.values()) / len(groups),
        "multi_call_trajectory_fraction": multi / rows,
        "treated_fraction": treated / rows,
        "gated_fraction": gated / rows,
        "treated_fraction_by_step": per_step,
        "eligible_turns": eligible,
        "s_t_v1_positive_turns": s1_v1,
        "s_t_v2_positive_turns": s1_v2,
        "turns_flagged_repeat": repeats,
        "turns_flagged_after_boxed": boxed,
        "max_abs_conservation_residual": max_residual,
        "exposure_floor": EXPOSURE_FLOOR,
        "exposure_above_floor": treated / rows >= EXPOSURE_FLOOR,
    }


# --------------------------------------------------------------------------- #
# ledger review samples (plan §8 step 3)
# --------------------------------------------------------------------------- #


def _stable_key(audit: dict[str, Any]) -> str:
    return hashlib.sha256(str(audit["trajectory_uid"]).encode("ascii")).hexdigest()


def _bucket(audit: dict[str, Any]) -> str:
    turns = audit["turns"]
    if any(turn["repeat_of_turn"] is not None for turn in turns):
        return "repeat"
    if any(turn["after_boxed"] and turn["credit_eligible"] for turn in turns):
        return "after_boxed"
    if audit["treated"]:
        return "treated_plain"
    if not audit["informative_group"] and int(audit["eligible_turn_count"]) >= 2:
        return "gated_zero_variance"
    return "single_or_uniform"


BUCKET_QUOTA = {"repeat": 3, "after_boxed": 2, "treated_plain": 3, "gated_zero_variance": 1, "single_or_uniform": 1}


def ledger_review_samples(run_dir: Path, steps: Iterable[int], count: int = 10) -> list[dict[str, Any]]:
    """Stable-hash, stratified selection of trajectories for manual §3.2 verification."""
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {name: [] for name in BUCKET_QUOTA}
    for _, row in _iter_rows(_step_paths(run_dir, steps)):
        audit = row["turn_credit_audit"]
        buckets[_bucket(audit)].append((_stable_key(audit), {"row": row, "audit": audit}))
    chosen: list[dict[str, Any]] = []
    leftovers: list[tuple[str, dict[str, Any]]] = []
    for name, quota in BUCKET_QUOTA.items():
        ordered = sorted(buckets[name], key=lambda item: item[0])
        for key, item in ordered[:quota]:
            chosen.append({"bucket": name, "stable_key": key, **item})
        leftovers.extend((key, {"bucket": name, **item}) for key, item in ordered[quota:])
    for key, item in sorted(leftovers, key=lambda pair: pair[0]):
        if len(chosen) >= count:
            break
        chosen.append({"stable_key": key, **item})
    samples = []
    for item in chosen[:count]:
        row, audit = item["row"], item["audit"]
        samples.append(
            {
                "bucket": item["bucket"],
                "step": int(row["step"]),
                "sample_id": row.get("sample_id"),
                "trajectory_uid": audit["trajectory_uid"],
                "prompt_uid": audit["prompt_uid"],
                "score": float(row["score"]),
                "acc": float(row["acc"]),
                "n_tool_calls": float(row["n_tool_calls"]),
                "group_score_std": audit["group_score_std"],
                "informative_group": audit["informative_group"],
                "eligible_turn_count": audit["eligible_turn_count"],
                "gated": audit["gated"],
                "treated": audit["treated"],
                "conservation_residual": audit["conservation_residual"],
                "A_traj": audit["turns"][0]["A_traj"],
                "turns": [
                    {
                        "turn_index": turn["turn_index"],
                        "n_t": turn["n_t"],
                        "execution_status": turn["execution_status"],
                        "adoption_status": turn["adoption_status"],
                        "credit_eligible": turn["credit_eligible"],
                        "s_t_v1": turn["s_t_v1"],
                        "repeat_of_turn": turn["repeat_of_turn"],
                        "after_boxed": turn["after_boxed"],
                        "s_t": turn["s_t"],
                        "s_tilde": turn["s_tilde"],
                        "correction": turn["correction"],
                        "A_turn": turn["A_turn"],
                        "codes": extract_codes(str(turn.get("assistant_text") or "")),
                        "code_hashes": [code_hash(c) for c in extract_codes(str(turn.get("assistant_text") or ""))],
                        "boxed_in_text": "\\boxed{" in str(turn.get("assistant_text") or ""),
                        "assistant_text": turn.get("assistant_text"),
                        "visible_tool_response_text": turn.get("visible_tool_response_text"),
                    }
                    for turn in audit["turns"]
                ],
            }
        )
    return samples


def write_review_samples(run_dir: Path, steps: Iterable[int], count: int = 10) -> Path:
    samples = ledger_review_samples(run_dir, steps, count)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    jsonl = analysis_dir / "ledger_review_samples.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    lines = [
        "# E5-v2 smoke ledger review samples",
        "",
        f"run: `{run_dir.name}` · {len(samples)} trajectories · stable-hash stratified by bucket",
        "",
    ]
    for index, sample in enumerate(samples, start=1):
        lines += [
            f"## Sample {index} · bucket `{sample['bucket']}` · step {sample['step']} · `{sample['trajectory_uid']}`",
            "",
            f"score {sample['score']:.2f} · acc {sample['acc']:.0f} · calls {sample['n_tool_calls']:.0f} · "
            f"group std {sample['group_score_std']:.4f} · informative {sample['informative_group']} · "
            f"eligible turns {sample['eligible_turn_count']} · gated {sample['gated']} · treated {sample['treated']} · "
            f"A_traj {sample['A_traj']:+.4f} · residual {sample['conservation_residual']:.1e}",
            "",
            "| t | n_t | exec | adoption | eligible | s_v1 | repeat_of | after_boxed | s_t | s̃ | c_t | code (first 70 chars) | boxed in text |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for turn in sample["turns"]:
            code = turn["codes"][0][:70].replace("\n", "⏎").replace("|", "\\|") if turn["codes"] else ""
            s_tilde = "" if turn["s_tilde"] is None else f"{turn['s_tilde']:.3f}"
            lines.append(
                f"| {turn['turn_index']} | {turn['n_t']} | {turn['execution_status']} | {turn['adoption_status']} | "
                f"{int(turn['credit_eligible'])} | {turn['s_t_v1']} | {turn['repeat_of_turn']} | {int(turn['after_boxed'])} | "
                f"{turn['s_t']} | {s_tilde} | {turn['correction']:+.4f} | `{code}` | {int(turn['boxed_in_text'])} |"
            )
        lines.append("")
    (analysis_dir / "ledger_review_samples.md").write_text("\n".join(lines), encoding="utf-8")
    return jsonl


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def _wrapper_installation(run_dir: Path) -> dict[str, Any]:
    wrapper = json.loads((run_dir / "wrapper_installation.json").read_text(encoding="utf-8"))
    if wrapper.get("boundary_version") != "e5v2_runtime_wrapper_v1" or wrapper.get("variant") != VARIANT_CONSERVED_V2:
        raise ValueError("wrapper installation proof is not the E5-v2 boundary")
    if not str(wrapper.get("proof_id", "")).startswith(PROOF_PREFIX):
        raise ValueError("wrapper proof prefix is not E5-v2")
    return wrapper


def smoke_gate(run_dir: Path, launch_ledger: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"smoke did not exit normally: {status}")
    ledger = json.loads(launch_ledger.read_text(encoding="utf-8"))
    if ledger.get("run_name") != run_dir.name or ledger.get("resumed") is not False:
        raise ValueError("fresh-launch ledger does not prove resumed=false")
    if Path(str(ledger.get("checkpoint"))).resolve() != SFT_PATH:
        raise ValueError("fresh-launch ledger does not identify the frozen SFT checkpoint")
    freeze = validate_freeze_bundle(PROJECT_ROOT / "eval/diagnostic_freeze_v2.sha256")
    if freeze["manifest_sha256"] != FREEZE_V2_SHA256:
        raise ValueError("freeze v2 manifest hash drift")
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("checkpoint_model_sha256") != SFT_MODEL_SHA256:
        raise ValueError("smoke preflight SFT model hash drift")
    config = validate_resolved_config_v2(run_dir, smoke=True)
    validation = validate_validation_files(run_dir, [0, 5])
    rollouts = validate_rollouts(run_dir, 5, audit_validator=validate_audit_v2)
    wrapper = _wrapper_installation(run_dir)
    retained = [path.name for path in (run_dir / "checkpoints").glob("global_step_*")] if (run_dir / "checkpoints").exists() else []
    if retained:
        raise ValueError(f"smoke retained checkpoints although save_freq=-1: {retained}")
    smoke_exposure = exposure(run_dir, range(1, 6))
    behavior = behavior_metrics(run_dir, range(1, 6))
    samples_path = write_review_samples(run_dir, range(1, 6))
    report = {
        "schema_version": "toolcredit_e5v2_smoke_gate_v1",
        "run_name": run_dir.name,
        "gate_passed": True,
        "fresh_launch": {"resumed": False, "checkpoint": str(SFT_PATH), "model_sha256": SFT_MODEL_SHA256},
        "freeze": freeze,
        "resolved_config": config,
        "validation": validation,
        "rollouts": rollouts,
        "wrapper_proof_id": wrapper["proof_id"],
        "retained_checkpoints": retained,
        "exposure": smoke_exposure,
        "behavior": behavior,
        "review_samples": str(samples_path.relative_to(run_dir)),
        "frozen_treatment": {
            "variant": VARIANT_CONSERVED_V2,
            "beta": FROZEN_BETA,
            "adoption_version": ADOPTION_VERSION_V2,
        },
        "disk_free_gib": round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 2),
        "gpu_visible": torch.cuda.is_available() and torch.cuda.device_count() == 1,
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    (analysis_dir / "smoke_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "evidence_classification.json").write_text(
        json.dumps(
            {
                "classification": "e5v2_five_step_smoke",
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
        raise ValueError(f"formal E5-v2 run is not completed: {status}")
    config = validate_resolved_config_v2(run_dir, smoke=False)
    validation = validate_validation_files(run_dir, FORMAL_VALIDATION_STEPS)
    rollouts = validate_rollouts(run_dir, 200, recovery_aware=True, audit_validator=validate_audit_v2)
    checkpoint = validate_checkpoint(run_dir, 200)
    wrapper = _wrapper_installation(run_dir)
    recovery_dir = run_dir / "recovery"
    recovery_events = (
        sorted(str(path.relative_to(run_dir)) for path in recovery_dir.glob("resume_from_*") if path.is_dir())
        if recovery_dir.exists()
        else []
    )
    report = {
        "schema_version": "toolcredit_e5v2_formal_completion_gate_v1",
        "run_name": run_dir.name,
        "status": "completed",
        "effective_actor_updates": 200,
        "resolved_config": config,
        "fixed_panel_pass_at_1": validation["pass_at_1"],
        "rollouts": rollouts,
        "checkpoint": checkpoint,
        "wrapper_proof_id": wrapper["proof_id"],
        "recovery_events": recovery_events,
        "exposure_first_25": exposure(run_dir, range(1, 26)),
        "exposure_last_25": exposure(run_dir, range(176, 201)),
        "behavior_last_25": behavior_gate(run_dir, 200),
        "behavior_all_200": behavior_metrics(run_dir, range(1, 201)),
        "behavior_step_50": behavior_gate(run_dir, 50),
        "behavior_step_100": behavior_gate(run_dir, 100),
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
    behavior = sub.add_parser("behavior")
    behavior.add_argument("run_dir", type=Path)
    behavior.add_argument("--upto", type=int, required=True)
    expo = sub.add_parser("exposure")
    expo.add_argument("run_dir", type=Path)
    expo.add_argument("--start", type=int, default=1)
    expo.add_argument("--end", type=int, default=25)
    samples = sub.add_parser("samples")
    samples.add_argument("run_dir", type=Path)
    samples.add_argument("--start", type=int, default=1)
    samples.add_argument("--end", type=int, default=5)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.command == "smoke":
        report = smoke_gate(run_dir, args.launch_ledger.resolve())
        print(json.dumps({"run_name": report["run_name"], "gate_passed": True, "exposure": report["exposure"]["treated_fraction"]}, indent=2))
    elif args.command == "formal":
        report = formal_completion(run_dir)
        print(json.dumps({"run_name": report["run_name"], "status": report["status"]}, indent=2))
    elif args.command == "behavior":
        print(json.dumps(behavior_gate(run_dir, args.upto), indent=2))
    elif args.command == "exposure":
        print(json.dumps(exposure(run_dir, range(args.start, args.end + 1)), indent=2))
    elif args.command == "samples":
        print(str(write_review_samples(run_dir, range(args.start, args.end + 1))))


if __name__ == "__main__":
    main()
