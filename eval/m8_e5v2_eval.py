"""M8 unified evaluation of E5-v2 under protocol v4 (plans/M8.md §8 step 6).

Only the new role is generated (760 x (1 + 4) = 3,800 trajectories).  raw/SFT/E3/E5-v1
trajectories, scores and coarse taxonomy proposals are reused byte-for-byte from the M7
canonical run after re-verifying its hash ledgers.  Generation, scoring, pairing and
bootstrap reuse the frozen v3 code (imported, not copied) wherever that code does not
hard-code the four-role universe; the few functions that do (raw-row validator,
``generate_one``, ``score_trajectory``) are mirrored here with the role universe extended.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from rewards.verifier import verify_answer
from rl.custom.reward import compute_score
from rl.custom.turn_advantage import LEDGER_KEY

from analysis.badcase_taxonomy import enrich_with_coarse_proposal
from analysis.paired_transitions import (
    accuracy_delta_statistic,
    conditional_fix_statistic,
    new_failure_statistic,
    paired_transition,
    source_stratified_question_bootstrap,
)
from eval.build_diagnostic_freeze_v4 import (
    M7_CANONICAL_RUN,
    MANIFEST_PATH as MANIFEST_V4,
    PAIRINGS,
    PROTOCOL_PATH as PROTOCOL_V4,
    verify as verify_v4,
    verify_m7_canonical_ledger,
)
from eval.checkpoints import COMMON_TOKENIZER, ROOT, sha256
from eval.checkpoints_v4 import E5V2_RUN, ROLE_V4, resolved_e5v2_role
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
from eval.metrics import pass_metrics

ROLE_NAMES_V4 = (*ROLE_NAMES, ROLE_V4)
PROTOCOL_VERSION_V4 = "toolcredit_diagnostic_v4"
SETTINGS = ("greedy", "sampled")
SOURCES = tuple(SOURCE_COUNTS)
RUN_ROOT = ROOT / "eval/runs"
E3_ROLE = "e3_grpo_baseline_step_200"
E5V1_ROLE = "e5_turn_credit_step_200"
E3_RUN = ROOT / "rl/runs/e3_grpo_baseline_20260819_224555"
E5V1_RUN = ROOT / "rl/runs/e5_turn_credit_20260823_082012"
APPROVAL_ENV = "M8_EVAL_APPROVED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# key universe and raw-row validation for the extended role set
# --------------------------------------------------------------------------- #


def expected_keys_v4(roles: Iterable[str] = (ROLE_V4,)) -> list[tuple[str, str, str, int, int]]:
    keys: list[tuple[str, str, str, int, int]] = []
    panel = load_panel()
    for role in roles:
        if role not in ROLE_NAMES_V4:
            raise ValueError(f"unknown role in v4 key universe: {role}")
        for setting in SETTINGS:
            for row in panel:
                for sample_index in setting_sample_indices(setting):
                    keys.append(
                        (role, setting, str(row["question_id"]), sample_index, derive_sample_seed(str(row["question_id"]), sample_index))
                    )
    if len(keys) != len(set(keys)):
        raise AssertionError("v4 expected key universe is not unique")
    return keys


def validate_raw_generation_row_v4(row: dict[str, Any], question: dict[str, Any]) -> None:
    """Mirror of the frozen validator with the role universe extended to E5-v2."""
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
    if row["checkpoint_role"] not in ROLE_NAMES_V4:
        raise ValueError("raw generation row has unknown checkpoint role")
    response = str(row["response"])
    if row["raw_output_sha256"] != hashlib.sha256(response.encode("utf-8")).hexdigest():
        raise ValueError("raw generation response hash mismatch")


def validated_resume_rows_v4(path: Path, questions: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    accepted: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id"))
        if question_id not in questions:
            raise ValueError(f"resume shard contains unknown question ID: {question_id}")
        validate_raw_generation_row_v4(row, questions[question_id])
        key = canonical_key(row)
        if key in accepted:
            raise ValueError(f"duplicate resume key: {key}")
        accepted[key] = row
    return accepted


# --------------------------------------------------------------------------- #
# generation (frozen loop, frozen server adapter, frozen seeds)
# --------------------------------------------------------------------------- #


async def generate_one_v4(
    loop: Any,
    tokenizer: Any,
    question: dict[str, Any],
    setting: str,
    sample_index: int,
    checkpoint_identity: dict[str, Any],
    manifest_sha256: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Mirror of ``eval.generate.generate_one`` for the E5-v2 role under protocol v4."""
    seed = derive_sample_seed(question["question_id"], sample_index)
    prompt = str(question["question"]) + config["generation"]["prompt_suffix"]
    raw_prompt = [{"role": "user", "content": prompt}]
    try:
        output = await loop.run(sampling_params=_sampling_params(config, setting, seed), raw_prompt=raw_prompt)
    except Exception as exc:
        raise RuntimeError(
            f"infrastructure/model-serving failure for {(ROLE_V4, setting, question['question_id'], sample_index)}"
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
        "protocol_version": PROTOCOL_VERSION_V4,
        "schema_version": "toolcredit_m7_trajectory_schema_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "freeze_manifest_sha256": manifest_sha256,
        "checkpoint_role": ROLE_V4,
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
    validate_raw_generation_row_v4(row, question)
    return row


def _require_approval() -> None:
    if os.environ.get(APPROVAL_ENV) != "YES":
        raise PermissionError(f"{APPROVAL_ENV}=YES is required (plans/M8.md §8 step 6 authorization)")


def _protocol_role_entry() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_V4)
    entry = protocol["checkpoints"][ROLE_V4]
    live = resolved_e5v2_role()
    live["state"], live["role"] = "validated", ROLE_V4
    if live != entry:
        raise ValueError("live E5-v2 role binding differs from the frozen v4 protocol")
    return entry


async def generate_shard_v4(setting: str, source: str, endpoint: str, output: Path) -> dict[str, Any]:
    if setting not in SETTINGS or source not in SOURCE_COUNTS:
        raise ValueError("unknown setting or source")
    _require_approval()
    config = validate_frozen_runtime_config()
    verify_v4()
    entry = _protocol_role_entry()
    server = SGLangHTTPServer(endpoint)
    await server.validate_model_path(Path(entry["inference_locator"]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(COMMON_TOKENIZER)
    loop = build_m7_loop(tokenizer, server, config)
    questions = [row for row in load_panel_questions() if row["source"] == source]
    question_map = {row["question_id"]: row for row in questions}
    existing = validated_resume_rows_v4(output, question_map)
    manifest_sha256 = sha256(MANIFEST_V4)
    generated = 0
    for question in questions:
        for sample_index in setting_sample_indices(setting):
            key = (ROLE_V4, setting, question["question_id"], sample_index, derive_sample_seed(question["question_id"], sample_index))
            if key in existing:
                continue
            row = await generate_one_v4(loop, tokenizer, question, setting, sample_index, entry, manifest_sha256, config)
            append_jsonl_fsync(output, row)
            existing[key] = row
            generated += 1
    expected_count = len(questions) * len(setting_sample_indices(setting))
    if len(existing) != expected_count:
        raise AssertionError(f"shard completion mismatch: {len(existing)} != {expected_count}")
    return {
        "status": "completed",
        "role": ROLE_V4,
        "setting": setting,
        "source": source,
        "expected": expected_count,
        "existing_before": expected_count - generated,
        "generated": generated,
        "output": str(output),
        "output_sha256": sha256(output),
    }


# --------------------------------------------------------------------------- #
# scoring (mirror of eval.evaluate.score_trajectory with the v4 validator)
# --------------------------------------------------------------------------- #


def score_trajectory_v4(raw: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    validate_raw_generation_row_v4(raw, question)
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
            "truncated": bool(score["truncated"]) or bool(tool_metadata.get("turn_credit_ledger", {}).get("trajectory_truncated", False)),
            "budget_exhausted": bool(float(score["n_tool_calls"]) >= 4 and not strict["correct"]),
            "primary_failure": None,
            "secondary_labels": [],
            "label_source": "pending_frozen_taxonomy_v1",
            "adjudication_rationale": None,
            "raw_evidence_pointer": {"canonical_key": list(canonical_key(raw)), "raw_output_sha256": raw["raw_output_sha256"]},
        }
    )
    return enriched


def score_shard(input_path: Path, output_path: Path) -> dict[str, Any]:
    questions = {row["question_id"]: row for row in load_panel_questions()}
    rows = _read_jsonl(input_path)
    scored = [score_trajectory_v4(row, questions[row["question_id"]]) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"input": str(input_path), "output": str(output_path), "rows": len(scored), "output_sha256": sha256(output_path)}


# --------------------------------------------------------------------------- #
# downstream: completeness, taxonomy proposals, pairing, bootstrap, metrics
# --------------------------------------------------------------------------- #


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"partial/blank JSONL at {path}:{line_number}")
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def completeness_v4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed: set[tuple[str, str, str, int, int]] = set()
    for row in rows:
        validate_scored_row(row)
        key = canonical_key(row)
        if key in observed:
            raise ValueError(f"duplicate canonical scored key: {key}")
        observed.add(key)
    expected = set(expected_keys_v4())
    missing, unexpected = sorted(expected - observed), sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(f"E5-v2 key universe incomplete: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return {
        "expected": len(expected),
        "observed": len(observed),
        "complete": True,
        "exact_key_digest": hashlib.sha256(json.dumps(sorted(observed), separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def load_m7_labeled_rows(roles: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    """Reused canonical scored rows joined with M7's frozen coarse taxonomy proposals."""
    proposals: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for row in _read_rows(M7_CANONICAL_RUN / "taxonomy/coarse_proposals.jsonl"):
        key = (str(row["checkpoint_role"]), str(row["evaluation_setting"]), str(row["question_id"]), int(row["sample_index"]), int(row["sample_seed"]))
        proposals[key] = row
    result: dict[str, list[dict[str, Any]]] = {}
    for role in roles:
        rows: list[dict[str, Any]] = []
        for setting in SETTINGS:
            for source in SOURCES:
                for row in _read_rows(M7_CANONICAL_RUN / "scored" / role / setting / f"{source}.jsonl"):
                    validate_scored_row(row)
                    proposal = proposals[canonical_key(row)]
                    labeled = dict(row)
                    for field in ("primary_failure", "secondary_labels", "label_source", "adjudication_rationale"):
                        labeled[field] = proposal[field]
                    if bool(labeled["correct"]) != (labeled["primary_failure"] is None):
                        raise ValueError("M7 coarse proposal violates the correct-iff-null invariant")
                    rows.append(labeled)
        if len(rows) != 3800:
            raise ValueError(f"reused role {role} does not have 3,800 canonical rows")
        result[role] = rows
    return result


def failure_category_migration_statistic(rows: list[dict[str, Any]]) -> float | None:
    """Among before-failures, the share that land in a *different failure* category after."""
    failed = [row for row in rows if row["before_state"] != "correct"]
    if not failed:
        return None
    migrated = sum(row["after_state"] not in {"correct", row["before_state"]} for row in failed)
    return migrated / len(failed)


STATISTICS: dict[str, Callable[[list[dict[str, Any]]], float | None]] = {
    "accuracy_delta": accuracy_delta_statistic,
    "conditional_fix_rate": conditional_fix_statistic,
    "new_failure_rate": new_failure_statistic,
    "failure_category_migration_rate": failure_category_migration_statistic,
}


def transitions_and_bootstrap(
    before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]], before_role: str, after_role: str, out_dir: Path, comparison: str
) -> dict[str, Any]:
    summary: dict[str, Any] = {"comparison": comparison, "before_role": before_role, "after_role": after_role, "label_scope": "deterministic coarse toolcredit_failure_taxonomy_v1", "settings": {}}
    bootstrap: dict[str, Any] = {"comparison": comparison, "replicates": 10000, "seed": 42, "confidence_interval": "two-sided percentile 95%", "resampling": "question_id within frozen source strata; sampled indices retained as a cluster", "settings": {}}
    paired_rows_all: list[dict[str, Any]] = []
    for setting in SETTINGS:
        before = [row for row in before_rows if row["evaluation_setting"] == setting]
        after = [row for row in after_rows if row["evaluation_setting"] == setting]
        full = paired_transition(before, after, setting, before_role, after_role)
        exact_rows = full.pop("exact_rows")
        for row in exact_rows:
            paired_rows_all.append({**row, "setting": setting})
        per_source = {}
        for source in SOURCES:
            result = paired_transition(
                [row for row in before if row["source"] == source], [row for row in after if row["source"] == source], setting, before_role, after_role
            )
            result.pop("exact_rows")
            per_source[source] = result
        summary["settings"][setting] = {"FULL760": full, "sources": per_source}
        bootstrap["settings"][setting] = {
            "FULL760": {name: source_stratified_question_bootstrap(exact_rows, statistic) for name, statistic in STATISTICS.items()},
            **{
                source: {
                    name: source_stratified_question_bootstrap([row for row in exact_rows if row["source"] == source], statistic)
                    for name, statistic in STATISTICS.items()
                }
                for source in SOURCES
            },
        }
    _write_json(out_dir / f"{comparison}.json", summary)
    _write_json(out_dir / f"{comparison}.bootstrap.json", bootstrap)
    _write_jsonl(out_dir / f"{comparison}.paired_rows.jsonl", paired_rows_all)
    return {"summary": summary, "bootstrap": bootstrap}


def _m7_behavior_function() -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Load the exact behavior-metric definition that produced M7's tool_behavior.json."""
    spec = importlib.util.spec_from_file_location("m7_produce_raw_metrics", M7_CANONICAL_RUN / "produce_raw_metrics.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M7 produce_raw_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.behavior


def training_curves() -> dict[str, Any]:
    from rl.compare_e4 import early_metrics, final_metrics, validation_curve

    result: dict[str, Any] = {}
    for name, run_dir in (("e3", E3_RUN), ("e5_v1", E5V1_RUN), ("e5_v2", E5V2_RUN)):
        curve = validation_curve(run_dir)
        points = sorted(curve.items())
        auc_0_200 = sum((r - l) * (lv + rv) / 2 for (l, lv), (r, rv) in zip(points, points[1:])) / 200
        result[name] = {
            "run_name": run_dir.name,
            "validation_curve": {str(step): value for step, value in points},
            "early": early_metrics(curve),
            "auc_0_200_normalized": auc_0_200,
            "final": final_metrics(curve),
        }
    return result


def classify_preregistered_case(
    greedy_delta: dict[str, Any], early_auc_delta_pt: float, behavior_pass: bool, exposure_first_25: float, exposure_floor: float = 0.08
) -> dict[str, Any]:
    """Mechanical reading of plans/M8.md §9.2 (A/B/C) with the §4 exposure clause on top."""
    point = greedy_delta["point_estimate"]
    lower, upper = greedy_delta["lower"], greedy_delta["upper"]
    ci_crosses_zero = lower <= 0 <= upper
    if point is not None and point * 100 >= 2 and not ci_crosses_zero:
        case = "C"
    elif ci_crosses_zero and early_auc_delta_pt >= 2:
        case = "B"
    elif ci_crosses_zero and abs(early_auc_delta_pt) < 2:
        case = "A"
    else:
        case = "outside_preregistered_table"
    if exposure_first_25 < exposure_floor and case == "A":
        case = "A_untested_exposure_below_floor"
    return {
        "case": case,
        "greedy_delta_point_pt": None if point is None else point * 100,
        "greedy_delta_ci_pt": [lower * 100, upper * 100],
        "ci_crosses_zero": ci_crosses_zero,
        "early_auc_0_100_delta_pt": early_auc_delta_pt,
        "behavior_pass_lines_met": behavior_pass,
        "exposure_first_25": exposure_first_25,
        "exposure_floor": exposure_floor,
        "interpretation_guardrail": "behavior lines must pass before accuracy is interpreted (plan §9.1); case C additionally requires the per-question migration evidence (plan §9.2)",
    }


def downstream(run_dir: Path) -> dict[str, Any]:
    verify_v4()
    reuse = verify_m7_canonical_ledger()
    questions = {row["question_id"]: row for row in load_panel_questions()}
    scored_rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        for source in SOURCES:
            scored_rows.extend(_read_rows(run_dir / "scored" / ROLE_V4 / setting / f"{source}.jsonl"))
    completeness = completeness_v4(scored_rows)
    labeled_v2 = [enrich_with_coarse_proposal(row, str(questions[row["question_id"]]["answer"])) for row in scored_rows]
    taxonomy_dir = run_dir / "taxonomy"
    _write_jsonl(
        taxonomy_dir / "coarse_proposals.jsonl",
        (
            {key: row[key] for key in ("checkpoint_role", "evaluation_setting", "source", "question_id", "sample_index", "sample_seed", "correct", "primary_failure", "secondary_labels", "label_source", "adjudication_rationale", "raw_evidence_pointer")}
            for row in labeled_v2
        ),
    )
    coarse_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in labeled_v2:
        label = "correct" if row["correct"] else str(row["primary_failure"])
        coarse_counts[f"{ROLE_V4}:{row['evaluation_setting']}:{row['source']}"][label] += 1
        coarse_counts[f"{ROLE_V4}:{row['evaluation_setting']}:FULL760"][label] += 1
    _write_json(
        taxonomy_dir / "coarse_metrics.json",
        {
            "scope": "deterministic coarse proposals only; no blinded audit for E5-v2 (plan §1 range: no new human audit)",
            "taxonomy_version": "toolcredit_failure_taxonomy_v1",
            "trajectory_count": len(labeled_v2),
            "groups": {key: dict(sorted(counter.items())) for key, counter in sorted(coarse_counts.items())},
        },
    )
    reused = load_m7_labeled_rows((E3_ROLE, E5V1_ROLE))
    transitions_dir = run_dir / "transitions"
    comparisons = {
        "e3_to_e5v2": transitions_and_bootstrap(reused[E3_ROLE], labeled_v2, E3_ROLE, ROLE_V4, transitions_dir, "e3_to_e5v2"),
        "e5v1_to_e5v2": transitions_and_bootstrap(reused[E5V1_ROLE], labeled_v2, E5V1_ROLE, ROLE_V4, transitions_dir, "e5v1_to_e5v2"),
    }
    behavior = _m7_behavior_function()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[f"{ROLE_V4}:{row['evaluation_setting']}:{row['source']}"].append(row)
        grouped[f"{ROLE_V4}:{row['evaluation_setting']}:FULL760"].append(row)
    metrics_dir = run_dir / "metrics"
    _write_json(metrics_dir / "pass_at_k.json", {"protocol_version": PROTOCOL_VERSION_V4, "metrics": pass_metrics(scored_rows)})
    _write_json(metrics_dir / "tool_behavior.json", {"protocol_version": PROTOCOL_VERSION_V4, "groups": {key: behavior(values) for key, values in sorted(grouped.items())}})
    _write_json(metrics_dir / "completeness.json", completeness)

    m7_pass = load_json(M7_CANONICAL_RUN / "metrics/pass_at_k.json")["metrics"]
    m7_behavior = load_json(M7_CANONICAL_RUN / "metrics/tool_behavior.json")["groups"]
    v2_pass = pass_metrics(scored_rows)
    v2_behavior = behavior(grouped[f"{ROLE_V4}:greedy:FULL760"])
    curves = training_curves()
    gate = load_json(E5V2_RUN / "analysis/formal_completion_gate.json")
    greedy_delta = comparisons["e3_to_e5v2"]["bootstrap"]["settings"]["greedy"]["FULL760"]["accuracy_delta"]
    early_delta_pt = (curves["e5_v2"]["early"]["auc_0_100_normalized"] - curves["e3"]["early"]["auc_0_100_normalized"]) * 100
    headline = {
        "protocol_version": PROTOCOL_VERSION_V4,
        "reused_m7_canonical": reuse,
        "greedy_full760_pass_at_1": {
            **{role: m7_pass[f"{role}:greedy:FULL760"]["pass@1"] for role in ROLE_NAMES},
            ROLE_V4: v2_pass[f"{ROLE_V4}:greedy:FULL760"]["pass@1"],
        },
        "sampled_full760_pass_at_k": {
            **{role: m7_pass[f"{role}:sampled:FULL760"] for role in ROLE_NAMES},
            ROLE_V4: v2_pass[f"{ROLE_V4}:sampled:FULL760"],
        },
        "greedy_full760_behavior": {
            **{
                role: {
                    key: m7_behavior[f"{role}:greedy:FULL760"][key]
                    for key in ("tool_calls", "tool_call_count_fractions", "truncation_rate", "per_call_execution_success", "behavior_candidates")
                }
                for role in (E3_ROLE, E5V1_ROLE)
            },
            ROLE_V4: {key: v2_behavior[key] for key in ("tool_calls", "tool_call_count_fractions", "truncation_rate", "per_call_execution_success", "behavior_candidates")},
        },
        "paired_greedy_full760": {
            name: {key: value["settings"]["greedy"]["FULL760"][key] for key in ("denominator", "fixed_failures", "new_failures", "unchanged_correct", "unchanged_failed", "accuracy_delta")}
            for name, value in ((k, v["summary"]) for k, v in comparisons.items())
        },
        "paired_greedy_full760_bootstrap": {name: value["bootstrap"]["settings"]["greedy"]["FULL760"] for name, value in comparisons.items()},
        "paired_sampled_full760_bootstrap": {name: value["bootstrap"]["settings"]["sampled"]["FULL760"] for name, value in comparisons.items()},
        "training_curves": curves,
        "training_behavior_last_25": gate["behavior_last_25"],
        "training_exposure_first_25": gate["exposure_first_25"],
        "training_exposure_last_25": gate["exposure_last_25"],
        "preregistered_case": classify_preregistered_case(
            greedy_delta,
            early_delta_pt,
            bool(gate["behavior_last_25"]["all_pass_lines_met"]),
            float(gate["exposure_first_25"]["treated_fraction"]),
        ),
        "scientific_interpretation_performed": False,
    }
    _write_json(metrics_dir / "m8_headline.json", headline)
    return headline


def write_closure(run_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "status.json"} and "server_logs" not in path.parts and "stage_logs" not in path.parts:
            entries.append(f"{sha256(path)}  {path.relative_to(run_dir)}")
    (run_dir / "hashes.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    status = {
        "run_name": run_dir.name,
        "protocol_version": PROTOCOL_VERSION_V4,
        "freeze_manifest_sha256": sha256(MANIFEST_V4),
        "status": "e5v2_evaluation_completed",
        "completed_at": utc_now(),
        "hashes_sha256": sha256(run_dir / "hashes.sha256"),
        "artifact_count": len(entries),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def preflight(run_dir: Path) -> dict[str, Any]:
    import torch

    report = {"protocol": verify_v4(), "m7_canonical": verify_m7_canonical_ledger(), "role": _protocol_role_entry()["materialization"]}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E5-v2 evaluation requires exactly one visible CUDA GPU")
    free_gpu, total_gpu = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    if disk_free < 8 * 1024**3:
        raise RuntimeError("E5-v2 evaluation disk gate failed: < 8 GiB free")
    report.update({"gpu_free_gib": round(free_gpu / 2**30, 2), "gpu_total_gib": round(total_gpu / 2**30, 2), "disk_free_gib": round(disk_free / 2**30, 2), "expected_new_keys": len(expected_keys_v4()), "run_dir": str(run_dir)})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--frozen-at", required=True)
    sub.add_parser("verify")
    pre = sub.add_parser("preflight")
    pre.add_argument("--run-dir", type=Path, required=True)
    gen = sub.add_parser("generate-shard")
    gen.add_argument("--setting", choices=SETTINGS, required=True)
    gen.add_argument("--source", choices=SOURCES, required=True)
    gen.add_argument("--endpoint", default="http://127.0.0.1:30000")
    gen.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    down = sub.add_parser("downstream")
    down.add_argument("--run-dir", type=Path, required=True)
    closure = sub.add_parser("closure")
    closure.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        from eval.checkpoints_v4 import materialize_e5v2

        result: Any = materialize_e5v2()
    elif args.command == "freeze":
        from eval.build_diagnostic_freeze_v4 import build

        build(args.frozen_at)
        from eval.verify_diagnostic_freeze_v4_semantics import verify_semantic_diff

        result = verify_semantic_diff()
    elif args.command == "verify":
        from eval.verify_diagnostic_freeze_v4_semantics import verify_semantic_diff

        result = verify_semantic_diff()
    elif args.command == "preflight":
        result = preflight(args.run_dir.resolve())
    elif args.command == "generate-shard":
        result = asyncio.run(generate_shard_v4(args.setting, args.source, args.endpoint, args.output))
    elif args.command == "score":
        result = score_shard(args.input, args.output)
    elif args.command == "downstream":
        result = downstream(args.run_dir.resolve())
    else:
        result = write_closure(args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
