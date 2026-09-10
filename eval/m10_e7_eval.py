"""M10 unified evaluation of E7 under protocol v6 (plans/M10.md §8 step 5).

Only the two new roles are generated (2 x 760 x (1 + 4) = 7,600 trajectories).  Every other
role's trajectories, scores and coarse taxonomy proposals are reused byte-for-byte from the M7
canonical run after re-verifying its hash ledgers.  Both E7 roles are ``tir``, so generation runs
through the same frozen M7 metadata agent loop as E3 -- no new generation path exists in v6.

The plan §9 reading needs two axes, and this module computes both:

* ``Delta_step`` -- the normalized AUC over effective steps 0-200 of the fixed-100 greedy
  validation curve, E7 minus E3, with a question-level bootstrap CI (both arms score the same
  frozen 100-question panel at the same validation steps, so questions pair exactly);
* ``Delta_compute`` -- the paired FULL760 greedy delta between the equal-compute checkpoint and
  E3 step-200, with the frozen source-stratified question bootstrap.

Reporting only the first would turn "spent 2-3x the rollouts" into "the algorithm is faster",
which plan §2 explicitly forbids.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from rewards.verifier import verify_answer
from rl.custom.reward import compute_score
from rl.custom.turn_advantage import LEDGER_KEY

from analysis.badcase_taxonomy import enrich_with_coarse_proposal
from analysis.m10_compute_axes import (
    E3_TOTAL_TRAJECTORIES,
    e3_trajectories_by_step,
    normalized_auc,
    read_ledger_trajectories,
    step_curve,
    trajectory_curve,
)
from eval.build_diagnostic_freeze_v4 import M7_CANONICAL_RUN, verify_m7_canonical_ledger
from eval.build_diagnostic_freeze_v5 import verify_m8_canonical_ledger
from eval.build_diagnostic_freeze_v6 import (
    MANIFEST_PATH as MANIFEST_V6,
    PAIRINGS,
    PROTOCOL_PATH as PROTOCOL_V6,
    verify as verify_v6,
    verify_m9_canonical_ledger,
)
from eval.checkpoints import COMMON_TOKENIZER, ROOT, sha256
from eval.checkpoints_v4 import ROLE_V4
from eval.checkpoints_v5 import NOTOOL_ROLE_NAMES
from eval.checkpoints_v6 import E7_ROLE_NAMES, E7_RUN, ROLE_EQUAL_COMPUTE, ROLE_STEP_200, resolved_e7_roles
from eval.evaluate import validate_scored_row
from eval.generate import (
    EVALUATOR_VERSION,
    ROLE_NAMES,
    SOURCE_COUNTS,
    SGLangHTTPServer,
    _plain_metrics,
    _read_jsonl,
    _sampling_params,
    append_jsonl_fsync,
    build_m7_loop,
    canonical_key,
    derive_sample_seed,
    load_json,
    load_panel,
    load_panel_questions,
    setting_sample_indices,
    validate_frozen_runtime_config,
)
from eval.m8_e5v2_eval import (
    STATISTICS,
    _read_rows,
    _write_json,
    _write_jsonl,
    load_m7_labeled_rows,
    transitions_and_bootstrap,
)
from eval.metrics import pass_metrics

ROLE_NAMES_V6 = (*ROLE_NAMES, ROLE_V4, *NOTOOL_ROLE_NAMES, *E7_ROLE_NAMES)
PROTOCOL_VERSION_V6 = "toolcredit_diagnostic_v6"
SETTINGS = ("greedy", "sampled")
SOURCES = tuple(SOURCE_COUNTS)
RUN_ROOT = ROOT / "eval/runs"
E3_ROLE = "e3_grpo_baseline_step_200"
E3_RUN = ROOT / "rl/runs/e3_grpo_baseline_20260819_224555"
APPROVAL_ENV = "M10_EVAL_APPROVED"
VALIDATION_STEPS = tuple(range(0, 201, 25))
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 42
DELTA_THRESHOLD_PT = 2.0

if len(set(ROLE_NAMES_V6)) != len(ROLE_NAMES_V6):
    raise AssertionError("v6 role universe contains duplicates")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# key universe and raw-row validation for the extended role set
# --------------------------------------------------------------------------- #


def expected_keys_v6(roles: Iterable[str] = E7_ROLE_NAMES) -> list[tuple[str, str, str, int, int]]:
    keys: list[tuple[str, str, str, int, int]] = []
    panel = load_panel()
    for role in roles:
        if role not in ROLE_NAMES_V6:
            raise ValueError(f"unknown role in v6 key universe: {role}")
        for setting in SETTINGS:
            for row in panel:
                for sample_index in setting_sample_indices(setting):
                    question_id = str(row["question_id"])
                    keys.append((role, setting, question_id, sample_index, derive_sample_seed(question_id, sample_index)))
    if len(keys) != len(set(keys)):
        raise AssertionError("v6 expected key universe is not unique")
    return keys


def validate_raw_generation_row_v6(row: dict[str, Any], question: dict[str, Any]) -> None:
    """Mirror of the frozen validator with the role universe extended to the two E7 roles."""
    required = {
        "protocol_version", "schema_version", "evaluator_version", "freeze_manifest_sha256",
        "checkpoint_role", "checkpoint_identity", "evaluation_setting", "question_id", "source",
        "split", "stratum", "question_hash", "sample_index", "sample_seed", "prompt", "prompt_hash",
        "messages", "response", "response_token_count", "termination_reason", "tool_metadata",
        "infrastructure_status", "raw_output_sha256",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"raw generation row is missing fields: {missing}")
    if row["infrastructure_status"] != "ok":
        raise ValueError("infrastructure failures cannot be accepted as model trajectories")
    if row["question_id"] != question["question_id"] or row["source"] != question["source"]:
        raise ValueError("raw generation row question identity mismatch")
    if row["question_hash"] != question["question_sha256_nfkc_whitespace"]:
        raise ValueError("raw generation row question hash mismatch")
    setting = str(row["evaluation_setting"])
    if int(row["sample_index"]) not in setting_sample_indices(setting):
        raise ValueError("sample index is outside its frozen setting")
    if int(row["sample_seed"]) != derive_sample_seed(str(row["question_id"]), int(row["sample_index"])):
        raise ValueError("raw generation row sample seed drifted")
    if row["checkpoint_role"] not in ROLE_NAMES_V6:
        raise ValueError("raw generation row has unknown checkpoint role")
    response = str(row["response"])
    if row["raw_output_sha256"] != hashlib.sha256(response.encode("utf-8")).hexdigest():
        raise ValueError("raw generation response hash mismatch")


def validated_resume_rows_v6(
    path: Path, questions: dict[str, dict[str, Any]]
) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    accepted: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id"))
        if question_id not in questions:
            raise ValueError(f"resume shard contains unknown question ID: {question_id}")
        validate_raw_generation_row_v6(row, questions[question_id])
        key = canonical_key(row)
        if key in accepted:
            raise ValueError(f"duplicate resume key: {key}")
        accepted[key] = row
    return accepted


# --------------------------------------------------------------------------- #
# generation (frozen loop, frozen server adapter, frozen seeds)
# --------------------------------------------------------------------------- #


async def generate_one_v6(
    loop: Any,
    tokenizer: Any,
    role: str,
    question: dict[str, Any],
    setting: str,
    sample_index: int,
    checkpoint_identity: dict[str, Any],
    manifest_sha256: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Mirror of ``eval.generate.generate_one`` for one E7 role under protocol v6."""
    seed = derive_sample_seed(question["question_id"], sample_index)
    prompt = str(question["question"]) + config["generation"]["prompt_suffix"]
    raw_prompt = [{"role": "user", "content": prompt}]
    try:
        output = await loop.run(sampling_params=_sampling_params(config, setting, seed), raw_prompt=raw_prompt)
    except Exception as exc:
        raise RuntimeError(
            f"infrastructure/model-serving failure for {(role, setting, question['question_id'], sample_index)}"
        ) from exc
    ledger = output.extra_fields.get(LEDGER_KEY)
    if not isinstance(ledger, dict):
        raise RuntimeError("M7 metadata loop returned no structured audit ledger")
    response = tokenizer.decode(output.response_ids, skip_special_tokens=True)
    prompt_ids_hash = hashlib.sha256(
        b"".join(int(item).to_bytes(4, "big", signed=False) for item in output.prompt_ids)
    ).hexdigest()
    messages: list[dict[str, Any]] = []
    for turn in ledger["assistant_turns"]:
        messages.append({"role": "assistant", "content": turn.get("assistant_text", ""), "turn_audit": turn})
        if turn.get("visible_tool_response_text") is not None:
            messages.append({"role": "tool", "content": turn["visible_tool_response_text"]})
    row = {
        "protocol_version": PROTOCOL_VERSION_V6,
        "schema_version": "toolcredit_m7_trajectory_schema_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "freeze_manifest_sha256": manifest_sha256,
        "checkpoint_role": role,
        "checkpoint_identity": checkpoint_identity,
        "evaluation_setting": setting,
        "question_id": question["question_id"],
        "source": question["source"],
        "split": question["split"],
        "stratum": question["stratum"],
        "question_hash": question["question_sha256_nfkc_whitespace"],
        "sample_index": sample_index,
        "sample_seed": seed,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": prompt_ids_hash,
        "messages": messages,
        "response": response,
        "response_token_count": len(output.response_ids),
        "response_mask": output.response_mask,
        "termination_reason": ledger.get("termination_reason"),
        "tool_metadata": {**output.extra_fields, "agent_loop_metrics": _plain_metrics(output.metrics)},
        "infrastructure_status": "ok",
        "raw_output_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
    }
    validate_raw_generation_row_v6(row, question)
    return row


def _require_approval() -> None:
    if os.environ.get(APPROVAL_ENV) != "YES":
        raise PermissionError(f"{APPROVAL_ENV}=YES is required (plans/M10.md §8 step 5 authorization)")


def _protocol_role_entry(role: str) -> dict[str, Any]:
    if role not in E7_ROLE_NAMES:
        raise ValueError(f"{role} is not one of the v6 generated roles")
    protocol = load_json(PROTOCOL_V6)
    entry = protocol["checkpoints"][role]
    live = resolved_e7_roles()[role]
    live["state"], live["role"] = "validated", role
    live["generation_mode"] = "tir"
    if live != entry:
        raise ValueError(f"live {role} binding differs from the frozen v6 protocol")
    return entry


async def generate_shard_v6(
    role: str, setting: str, source: str, endpoint: str, output: Path, limit_questions: int | None = None
) -> dict[str, Any]:
    """Fill one (role, setting, source) shard sequentially, exactly like the frozen M7/M8 tir path.

    ``limit_questions`` exists only for the smoke run (one question per source, M9 precedent); the
    canonical run never passes it, and the completion assertion is against the limited panel.
    """
    if setting not in SETTINGS or source not in SOURCE_COUNTS:
        raise ValueError("unknown setting or source")
    if limit_questions is not None and limit_questions < 1:
        raise ValueError("limit_questions must be positive")
    _require_approval()
    config = validate_frozen_runtime_config()
    verify_v6()
    entry = _protocol_role_entry(role)
    server = SGLangHTTPServer(endpoint)
    await server.validate_model_path(Path(entry["inference_locator"]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(COMMON_TOKENIZER)
    loop = build_m7_loop(tokenizer, server, config)
    questions = [row for row in load_panel_questions() if row["source"] == source]
    if limit_questions is not None:
        questions = questions[:limit_questions]
    question_map = {row["question_id"]: row for row in questions}
    existing = validated_resume_rows_v6(output, question_map)
    if any(key[0] != role for key in existing):
        raise ValueError(f"resume shard for {role} contains another role's rows")
    manifest_sha256 = sha256(MANIFEST_V6)
    generated = 0
    for question in questions:
        for sample_index in setting_sample_indices(setting):
            key = (
                role,
                setting,
                question["question_id"],
                sample_index,
                derive_sample_seed(question["question_id"], sample_index),
            )
            if key in existing:
                continue
            row = await generate_one_v6(
                loop, tokenizer, role, question, setting, sample_index, entry, manifest_sha256, config
            )
            append_jsonl_fsync(output, row)
            existing[key] = row
            generated += 1
    expected_count = len(questions) * len(setting_sample_indices(setting))
    if len(existing) != expected_count:
        raise AssertionError(f"shard completion mismatch: {len(existing)} != {expected_count}")
    return {
        "status": "completed",
        "role": role,
        "setting": setting,
        "source": source,
        "expected": expected_count,
        "existing_before": expected_count - generated,
        "generated": generated,
        "limit_questions": limit_questions,
        "output": str(output),
        "output_sha256": sha256(output),
    }


# --------------------------------------------------------------------------- #
# scoring (mirror of eval.evaluate.score_trajectory with the v6 validator)
# --------------------------------------------------------------------------- #


def score_trajectory_v6(raw: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    validate_raw_generation_row_v6(raw, question)
    if raw.get("infrastructure_status") != "ok":
        raise ValueError("infrastructure failures must be retried, not scored as wrong")
    tool_metadata = raw.get("tool_metadata")
    if not isinstance(tool_metadata, dict):
        raise TypeError("tool_metadata must be a mapping")
    score = compute_score(
        data_source="toolcredit_math",
        solution_str=str(raw["response"]),
        ground_truth=str(question["answer"]),
        extra_info={"id": raw["question_id"], **tool_metadata},
    )
    strict = verify_answer(str(raw["response"]), str(question["answer"]), strict_boxed=True)
    if bool(score["acc"]) != bool(strict["correct"]):
        raise AssertionError("project reward and strict verifier correctness diverged")
    enriched = dict(raw)
    enriched.update(
        {
            "reward": score,
            "verifier_outcome": strict,
            "correct": bool(strict["correct"]),
            "format_ok": bool(score["format_ok"]),
            "invalid": bool(score["invalid"]),
            "truncated": bool(score["truncated"])
            or bool(tool_metadata.get("turn_credit_ledger", {}).get("trajectory_truncated", False)),
            "budget_exhausted": bool(float(score["n_tool_calls"]) >= 4 and not strict["correct"]),
            "primary_failure": None,
            "secondary_labels": [],
            "label_source": "pending_frozen_taxonomy_v1",
            "adjudication_rationale": None,
            "raw_evidence_pointer": {
                "canonical_key": list(canonical_key(raw)),
                "raw_output_sha256": raw["raw_output_sha256"],
            },
        }
    )
    return enriched


def score_shard(input_path: Path, output_path: Path) -> dict[str, Any]:
    questions = {row["question_id"]: row for row in load_panel_questions()}
    rows = _read_jsonl(input_path)
    scored = [score_trajectory_v6(row, questions[row["question_id"]]) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(scored),
        "output_sha256": sha256(output_path),
    }


def completeness_v6(rows: list[dict[str, Any]], roles: Iterable[str] = E7_ROLE_NAMES) -> dict[str, Any]:
    observed: set[tuple[str, str, str, int, int]] = set()
    for row in rows:
        validate_scored_row(row)
        key = canonical_key(row)
        if key in observed:
            raise ValueError(f"duplicate canonical scored key: {key}")
        observed.add(key)
    expected = set(expected_keys_v6(roles))
    missing, unexpected = sorted(expected - observed), sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(f"E7 key universe incomplete: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return {
        "expected": len(expected),
        "observed": len(observed),
        "complete": True,
        "exact_key_digest": hashlib.sha256(
            json.dumps(sorted(observed), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# the two compute axes (plan §9)
# --------------------------------------------------------------------------- #


def per_question_validation_curves(run_dir: Path, steps: Iterable[int] = VALIDATION_STEPS) -> dict[str, dict[int, float]]:
    """``question_id -> {step: acc}`` over the frozen fixed-100 panel, for the paired AUC bootstrap."""
    curves: dict[str, dict[int, float]] = {}
    for step in steps:
        path = run_dir / "predictions/validation" / f"{step}.jsonl"
        rows = _read_rows(path)
        if len(rows) != 100:
            raise ValueError(f"fixed panel must have 100 rows at {path}")
        for row in rows:
            curves.setdefault(str(row["sample_id"]), {})[int(step)] = float(row["acc"])
    incomplete = [qid for qid, curve in curves.items() if set(curve) != set(steps)]
    if len(curves) != 100 or incomplete:
        raise ValueError(f"fixed-100 panel is not complete across {sorted(steps)}: {incomplete[:3]}")
    return curves


def _auc_from_question_curves(curves: dict[str, dict[int, float]], question_ids: list[str], step_max: int = 200) -> float:
    mean_curve = {
        step: sum(curves[qid][step] for qid in question_ids) / len(question_ids) for step in VALIDATION_STEPS
    }
    return normalized_auc(step_curve(mean_curve), step_max)


def auc_delta_bootstrap(
    e3_curves: dict[str, dict[int, float]],
    e7_curves: dict[str, dict[int, float]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Question-level paired bootstrap of the step-axis AUC difference (plan §9 Delta_step CI).

    Both arms evaluate the same frozen 100-question panel at the same validation steps, so a
    replicate resamples question IDs once and recomputes both curves from that same resample.
    """
    question_ids = sorted(e3_curves)
    if question_ids != sorted(e7_curves):
        raise ValueError("the two arms did not evaluate the same fixed-100 panel")
    point = _auc_from_question_curves(e7_curves, question_ids) - _auc_from_question_curves(e3_curves, question_ids)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sample = [question_ids[rng.randrange(len(question_ids))] for _ in question_ids]
        deltas.append(
            _auc_from_question_curves(e7_curves, sample) - _auc_from_question_curves(e3_curves, sample)
        )
    deltas.sort()
    lower = deltas[int(0.025 * (replicates - 1))]
    upper = deltas[int(0.975 * (replicates - 1))]
    return {
        "point_estimate": point,
        "point_estimate_pt": point * 100,
        "lower": lower,
        "upper": upper,
        "ci_pt": [lower * 100, upper * 100],
        "replicates": replicates,
        "seed": seed,
        "confidence_interval": "two-sided percentile 95%",
        "resampling": "question_id over the frozen fixed-100 validation panel; both arms resampled together",
    }


def compute_axes(e7_run: Path = E7_RUN, e3_run: Path = E3_RUN) -> dict[str, Any]:
    """Both x-axes of the fixed-100 greedy curve plus the secondary compute axes (plan §9)."""
    e3_curves = per_question_validation_curves(e3_run)
    e7_curves = per_question_validation_curves(e7_run)
    question_ids = sorted(e3_curves)
    e3_mean = {step: sum(e3_curves[q][step] for q in question_ids) / 100 for step in VALIDATION_STEPS}
    e7_mean = {step: sum(e7_curves[q][step] for q in question_ids) / 100 for step in VALIDATION_STEPS}
    e7_trajectories = read_ledger_trajectories(e7_run)
    ledger_rows = [json.loads(line) for line in (e7_run / "e7_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    last = ledger_rows[-1]
    step_axis = {
        "x_max": 200,
        "e3_auc": normalized_auc(step_curve(e3_mean), 200),
        "e7_auc": normalized_auc(step_curve(e7_mean), 200),
    }
    step_axis["delta_e7_minus_e3"] = step_axis["e7_auc"] - step_axis["e3_auc"]
    trajectory_axis = {
        "x_max": E3_TOTAL_TRAJECTORIES,
        "e3_auc": normalized_auc(trajectory_curve(e3_mean, e3_trajectories_by_step()), E3_TOTAL_TRAJECTORIES),
        "e7_auc": normalized_auc(trajectory_curve(e7_mean, e7_trajectories), E3_TOTAL_TRAJECTORIES),
    }
    trajectory_axis["delta_e7_minus_e3"] = trajectory_axis["e7_auc"] - trajectory_axis["e3_auc"]
    return {
        "validation_steps": list(VALIDATION_STEPS),
        "e3_fixed100_curve": {str(k): v for k, v in sorted(e3_mean.items())},
        "e7_fixed100_curve": {str(k): v for k, v in sorted(e7_mean.items())},
        "step_axis": step_axis,
        "trajectory_axis": trajectory_axis,
        "delta_step_bootstrap": auc_delta_bootstrap(e3_curves, e7_curves),
        "secondary_compute_axes": {
            "e7_total_rollout_trajectories": int(last["cumulative_trajectories"]),
            "e3_total_rollout_trajectories": E3_TOTAL_TRAJECTORIES,
            "trajectory_ratio": int(last["cumulative_trajectories"]) / E3_TOTAL_TRAJECTORIES,
            "e7_total_rollout_tokens": int(last["cumulative_rollout_tokens"]),
            "e7_total_generation_wall_time_hours": float(last["cumulative_generation_wall_time"]) / 3600,
            "e7_mean_generation_batches": sum(int(r["generation_batches_used"]) for r in ledger_rows) / len(ledger_rows),
            "e7_equal_compute_step": next(
                (int(r["effective_step"]) for r in ledger_rows if r.get("equal_compute_checkpoint")), None
            ),
        },
        "informative_set_drift": {
            "zero_std_frac_by_quartile": [
                sum(
                    int(r["zero_std_groups"]) / (int(r["informative_groups"]) + int(r["zero_std_groups"]))
                    for r in ledger_rows[lo:hi]
                )
                / (hi - lo)
                for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200))
            ],
            "scope": "descriptive only (plan §9); never a decision criterion",
        },
    }


def _direction(point_pt: float | None, lower_pt: float, upper_pt: float) -> str:
    """Plan §9: positive/negative need both a 2pt effect and a CI that excludes zero."""
    if point_pt is None:
        return "undetermined"
    crosses_zero = lower_pt <= 0 <= upper_pt
    if point_pt >= DELTA_THRESHOLD_PT and not crosses_zero:
        return "positive"
    if point_pt <= -DELTA_THRESHOLD_PT and not crosses_zero:
        return "negative"
    return "unchanged"


def classify_preregistered_case(delta_step: dict[str, Any], delta_compute: dict[str, Any]) -> dict[str, Any]:
    """Mechanical reading of the plan §9 four-cell matrix (Delta_step x Delta_compute)."""
    step_dir = _direction(delta_step["point_estimate_pt"], *delta_step["ci_pt"])
    compute_point = delta_compute["point_estimate"]
    compute_dir = _direction(
        None if compute_point is None else compute_point * 100,
        delta_compute["lower"] * 100,
        delta_compute["upper"] * 100,
    )
    if compute_dir == "negative":
        case = "D"
    elif step_dir == "positive" and compute_dir == "positive":
        case = "B"
    elif step_dir == "positive" and compute_dir == "unchanged":
        case = "A"
    elif step_dir == "unchanged" and compute_dir == "unchanged":
        case = "C"
    else:
        case = "outside_preregistered_table"
    writeups = {
        "A": "filtering converted no-gradient rollout compute into more informative groups: higher per-update sample efficiency, but no free lunch per unit of rollout compute",
        "B": "filtering has a net benefit: E7 is more accurate at the same 102,400 rollout trajectories",
        "C": "E3's zero-variance groups had limited effect on the learning curve; no difference detected (a CI crossing 0 is not absence)",
        "D": "the difficulty drift introduced by resampling outweighed the sample-efficiency gain: worse at equal compute",
        "outside_preregistered_table": "record under the nearest cell and state the deviation explicitly",
    }
    return {
        "case": case,
        "delta_step_direction": step_dir,
        "delta_compute_direction": compute_dir,
        "delta_step_pt": delta_step["point_estimate_pt"],
        "delta_step_ci_pt": delta_step["ci_pt"],
        "delta_compute_pt": None if compute_point is None else compute_point * 100,
        "delta_compute_ci_pt": [delta_compute["lower"] * 100, delta_compute["upper"] * 100],
        "threshold_pt": DELTA_THRESHOLD_PT,
        "preregistered_writeup": writeups[case],
        "guardrail": "both axes must be reported together; the step axis alone would present 2-3x the rollout spend as a speedup (plan §2)",
    }


# --------------------------------------------------------------------------- #
# downstream
# --------------------------------------------------------------------------- #


def _m7_behavior_function() -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("m7_produce_raw_metrics", M7_CANONICAL_RUN / "produce_raw_metrics.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M7 produce_raw_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.behavior


def downstream(run_dir: Path) -> dict[str, Any]:
    verify_v6()
    reuse_m7 = verify_m7_canonical_ledger()
    reuse_m8 = verify_m8_canonical_ledger()
    reuse_m9 = verify_m9_canonical_ledger()
    questions = {row["question_id"]: row for row in load_panel_questions()}

    scored: dict[str, list[dict[str, Any]]] = {}
    for role in E7_ROLE_NAMES:
        rows: list[dict[str, Any]] = []
        for setting in SETTINGS:
            for source in SOURCES:
                rows.extend(_read_rows(run_dir / "scored" / role / setting / f"{source}.jsonl"))
        scored[role] = rows
    completeness = completeness_v6([row for rows in scored.values() for row in rows])

    labeled: dict[str, list[dict[str, Any]]] = {
        role: [enrich_with_coarse_proposal(row, str(questions[row["question_id"]]["answer"])) for row in rows]
        for role, rows in scored.items()
    }
    taxonomy_dir = run_dir / "taxonomy"
    _write_jsonl(
        taxonomy_dir / "coarse_proposals.jsonl",
        (
            {
                key: row[key]
                for key in (
                    "checkpoint_role", "evaluation_setting", "source", "question_id", "sample_index",
                    "sample_seed", "correct", "primary_failure", "secondary_labels", "label_source",
                    "adjudication_rationale", "raw_evidence_pointer",
                )
            }
            for role in E7_ROLE_NAMES
            for row in labeled[role]
        ),
    )
    coarse_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for role in E7_ROLE_NAMES:
        for row in labeled[role]:
            label = "correct" if row["correct"] else str(row["primary_failure"])
            coarse_counts[f"{role}:{row['evaluation_setting']}:{row['source']}"][label] += 1
            coarse_counts[f"{role}:{row['evaluation_setting']}:FULL760"][label] += 1
    _write_json(
        taxonomy_dir / "coarse_metrics.json",
        {
            "scope": "deterministic coarse proposals only; no blinded audit for E7 (plan §1 range: no new human audit)",
            "taxonomy_version": "toolcredit_failure_taxonomy_v1",
            "trajectory_count": sum(len(rows) for rows in labeled.values()),
            "groups": {key: dict(sorted(counter.items())) for key, counter in sorted(coarse_counts.items())},
        },
    )

    reused = load_m7_labeled_rows((E3_ROLE,))
    transitions_dir = run_dir / "transitions"
    comparisons: dict[str, Any] = {}
    for pairing in PAIRINGS:
        before_role, after_role = pairing["before_role"], pairing["after_role"]
        before = reused[E3_ROLE] if before_role == E3_ROLE else labeled[before_role]
        after = labeled[after_role]
        name = f"{before_role}__to__{after_role}"
        comparisons[name] = transitions_and_bootstrap(before, after, before_role, after_role, transitions_dir, name)

    behavior = _m7_behavior_function()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for role in E7_ROLE_NAMES:
        for row in scored[role]:
            grouped[f"{role}:{row['evaluation_setting']}:{row['source']}"].append(row)
            grouped[f"{role}:{row['evaluation_setting']}:FULL760"].append(row)
    all_scored = [row for rows in scored.values() for row in rows]
    metrics_dir = run_dir / "metrics"
    _write_json(metrics_dir / "pass_at_k.json", {"protocol_version": PROTOCOL_VERSION_V6, "metrics": pass_metrics(all_scored)})
    _write_json(
        metrics_dir / "tool_behavior.json",
        {"protocol_version": PROTOCOL_VERSION_V6, "groups": {key: behavior(values) for key, values in sorted(grouped.items())}},
    )
    _write_json(metrics_dir / "completeness.json", completeness)

    axes = compute_axes()
    _write_json(metrics_dir / "compute_axes.json", axes)

    m7_pass = load_json(M7_CANONICAL_RUN / "metrics/pass_at_k.json")["metrics"]
    m7_behavior = load_json(M7_CANONICAL_RUN / "metrics/tool_behavior.json")["groups"]
    v6_pass = pass_metrics(all_scored)
    equal_compute_pairing = f"{E3_ROLE}__to__{ROLE_EQUAL_COMPUTE}"
    delta_compute = comparisons[equal_compute_pairing]["bootstrap"]["settings"]["greedy"]["FULL760"]["accuracy_delta"]
    headline = {
        "protocol_version": PROTOCOL_VERSION_V6,
        "reused_m7_canonical": reuse_m7,
        "reused_m8_canonical": reuse_m8,
        "reused_m9_canonical": reuse_m9,
        "greedy_full760_pass_at_1": {
            E3_ROLE: m7_pass[f"{E3_ROLE}:greedy:FULL760"]["pass@1"],
            **{role: v6_pass[f"{role}:greedy:FULL760"]["pass@1"] for role in E7_ROLE_NAMES},
        },
        "sampled_full760_pass_at_k": {
            E3_ROLE: m7_pass[f"{E3_ROLE}:sampled:FULL760"],
            **{role: v6_pass[f"{role}:sampled:FULL760"] for role in E7_ROLE_NAMES},
        },
        "greedy_full760_behavior": {
            E3_ROLE: {
                key: m7_behavior[f"{E3_ROLE}:greedy:FULL760"][key]
                for key in ("tool_calls", "tool_call_count_fractions", "truncation_rate", "per_call_execution_success", "behavior_candidates")
            },
            **{
                role: {
                    key: behavior(grouped[f"{role}:greedy:FULL760"])[key]
                    for key in ("tool_calls", "tool_call_count_fractions", "truncation_rate", "per_call_execution_success", "behavior_candidates")
                }
                for role in E7_ROLE_NAMES
            },
        },
        "paired_greedy_full760": {
            name: {
                key: value["summary"]["settings"]["greedy"]["FULL760"][key]
                for key in ("denominator", "fixed_failures", "new_failures", "unchanged_correct", "unchanged_failed", "accuracy_delta")
            }
            for name, value in comparisons.items()
        },
        "paired_greedy_full760_bootstrap": {name: value["bootstrap"]["settings"]["greedy"]["FULL760"] for name, value in comparisons.items()},
        "paired_sampled_full760_bootstrap": {name: value["bootstrap"]["settings"]["sampled"]["FULL760"] for name, value in comparisons.items()},
        "compute_axes": axes,
        "preregistered_case": classify_preregistered_case(axes["delta_step_bootstrap"], delta_compute),
        "scientific_interpretation_performed": False,
    }
    _write_json(metrics_dir / "m10_headline.json", headline)
    return headline


def write_closure(run_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "status.json"} and "server_logs" not in path.parts and "stage_logs" not in path.parts:
            entries.append(f"{sha256(path)}  {path.relative_to(run_dir)}")
    (run_dir / "hashes.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    status = {
        "run_name": run_dir.name,
        "protocol_version": PROTOCOL_VERSION_V6,
        "freeze_manifest_sha256": sha256(MANIFEST_V6),
        "status": "e7_evaluation_completed",
        "completed_at": utc_now(),
        "hashes_sha256": sha256(run_dir / "hashes.sha256"),
        "artifact_count": len(entries),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def preflight(run_dir: Path) -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {
        "protocol": verify_v6(),
        "m7_canonical": verify_m7_canonical_ledger(),
        "m8_canonical": verify_m8_canonical_ledger(),
        "m9_canonical": verify_m9_canonical_ledger(),
        "roles": {role: _protocol_role_entry(role)["materialization"] for role in E7_ROLE_NAMES},
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E7 evaluation requires exactly one visible CUDA GPU")
    free_gpu, total_gpu = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    if disk_free < 8 * 1024**3:
        raise RuntimeError("E7 evaluation disk gate failed: < 8 GiB free")
    report.update(
        {
            "gpu_free_gib": round(free_gpu / 2**30, 2),
            "gpu_total_gib": round(total_gpu / 2**30, 2),
            "disk_free_gib": round(disk_free / 2**30, 2),
            "expected_new_keys": len(expected_keys_v6()),
            "run_dir": str(run_dir),
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--role", choices=E7_ROLE_NAMES, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--frozen-at", required=True)
    sub.add_parser("verify")
    pre = sub.add_parser("preflight")
    pre.add_argument("--run-dir", type=Path, required=True)
    gen = sub.add_parser("generate-shard")
    gen.add_argument("--role", choices=E7_ROLE_NAMES, required=True)
    gen.add_argument("--setting", choices=SETTINGS, required=True)
    gen.add_argument("--source", choices=SOURCES, required=True)
    gen.add_argument("--endpoint", default="http://127.0.0.1:30000")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--limit-questions", type=int, default=None)
    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    axes = sub.add_parser("compute-axes")
    axes.add_argument("--output", type=Path)
    down = sub.add_parser("downstream")
    down.add_argument("--run-dir", type=Path, required=True)
    closure = sub.add_parser("closure")
    closure.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        from eval.checkpoints_v6 import materialize_e7

        result: Any = materialize_e7(args.role)
    elif args.command == "freeze":
        from eval.build_diagnostic_freeze_v6 import build

        build(args.frozen_at)
        from eval.verify_diagnostic_freeze_v6_semantics import verify_semantic_diff

        result = verify_semantic_diff()
    elif args.command == "verify":
        from eval.verify_diagnostic_freeze_v6_semantics import verify_semantic_diff

        result = verify_semantic_diff()
    elif args.command == "preflight":
        result = preflight(args.run_dir.resolve())
    elif args.command == "generate-shard":
        result = asyncio.run(
            generate_shard_v6(args.role, args.setting, args.source, args.endpoint, args.output, args.limit_questions)
        )
    elif args.command == "score":
        result = score_shard(args.input, args.output)
    elif args.command == "compute-axes":
        result = compute_axes()
        if args.output:
            _write_json(args.output, result)
    elif args.command == "downstream":
        result = downstream(args.run_dir.resolve())
    else:
        result = write_closure(args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
