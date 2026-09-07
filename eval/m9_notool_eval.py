"""M9 unified evaluation of the four ``notool`` roles under protocol v5 (plans/M9.md §7 step 4).

Only the notool roles are generated (4 x 760 x (1 + 4) = 15,200 trajectories).  The five
tir roles (raw/SFT/E3/E5-v1 from M7, E5-v2 from M8) are reused byte-for-byte after
re-verifying their hash ledgers.  Scoring, pairing and bootstrap reuse the frozen v3/v4 code
(imported, not copied).  The notool generation path is deliberately tiny: the prompt is the
chat template applied with ``tools=None`` (token-identical to what veRL's native
``single_turn_agent`` produced during E3-NoTool training, plans/M9.md §6 Fixture J), one
SGLang ``/generate`` call, no tool loop.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rewards.verifier import verify_answer
from rl.custom.reward import compute_score
from rl.validate_e3notool_run import tool_tag_stats

from analysis.badcase_taxonomy import enrich_with_coarse_proposal
from eval.build_diagnostic_freeze_v5 import (
    E3_ROLE,
    M7_CANONICAL_RUN,
    M8_CANONICAL_RUN,
    MANIFEST_PATH as MANIFEST_V5,
    PAIRINGS,
    PROTOCOL_PATH as PROTOCOL_V5,
    RAW_ROLE,
    verify as verify_v5,
    verify_m7_canonical_ledger,
    verify_m8_canonical_ledger,
)
from eval.checkpoints import COMMON_TOKENIZER, ROOT, sha256
from eval.checkpoints_v4 import ROLE_V4
from eval.checkpoints_v5 import (
    GENERATION_MODE_NOTOOL,
    GENERATION_MODE_TIR,
    NOTOOL_ROLE_NAMES,
    NOTOOL_RUN,
    NOTOOL_WEIGHT_ROLES,
    ROLE_V5,
    resolved_notool_roles,
)
from eval.evaluate import validate_scored_row
from eval.generate import (
    EVALUATOR_VERSION,
    ROLE_NAMES,
    SOURCE_COUNTS,
    SGLangHTTPServer,
    _read_jsonl,
    _sampling_params,
    append_jsonl_fsync,
    canonical_key,
    derive_sample_seed,
    load_json,
    load_panel,
    load_panel_questions,
    setting_sample_indices,
    to_sglang_sampling_params,
    validate_frozen_runtime_config,
)
from eval.m8_e5v2_eval import _read_rows, _write_json, _write_jsonl, load_m7_labeled_rows, transitions_and_bootstrap
from eval.metrics import pass_metrics

ROLE_NAMES_V5 = (*ROLE_NAMES, ROLE_V4, *NOTOOL_ROLE_NAMES)
PROTOCOL_VERSION_V5 = "toolcredit_diagnostic_v5"
NOTOOL_SCHEMA_VERSION = "toolcredit_m9_notool_trajectory_schema_v1"
SETTINGS = ("greedy", "sampled")
SOURCES = tuple(SOURCE_COUNTS)
RUN_ROOT = ROOT / "eval/runs"
E3_RUN = ROOT / "rl/runs/e3_grpo_baseline_20260819_224555"
APPROVAL_ENV = "M9_EVAL_APPROVED"
NOTOOL_TERMINATION_REASONS = ("final_answer", "length")
TOOL_META_ZERO_FIELDS = ("tool_call_counts", "tool_success_count", "tool_error_count", "tool_parse_error_count")
PREREGISTERED_EFFECT_PT = 2.0
# User-authorized (2026-09-07) exploratory sensitivity roles.  They share the notool row
# invariants but are never part of the v5 protocol, key universe, pairings or closure.
SENSITIVITY_ROLES = ("e3notool_grpo_step_125_sensitivity",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_notool_role(role: str) -> bool:
    return role in NOTOOL_ROLE_NAMES or role in SENSITIVITY_ROLES


# --------------------------------------------------------------------------- #
# key universe and raw-row validation for the extended role set
# --------------------------------------------------------------------------- #


def expected_keys_v5(roles: Iterable[str] = NOTOOL_ROLE_NAMES) -> list[tuple[str, str, str, int, int]]:
    keys: list[tuple[str, str, str, int, int]] = []
    panel = load_panel()
    for role in roles:
        if role not in ROLE_NAMES_V5:
            raise ValueError(f"unknown role in v5 key universe: {role}")
        for setting in SETTINGS:
            for row in panel:
                for sample_index in setting_sample_indices(setting):
                    keys.append(
                        (role, setting, str(row["question_id"]), sample_index, derive_sample_seed(str(row["question_id"]), sample_index))
                    )
    if len(keys) != len(set(keys)):
        raise AssertionError("v5 expected key universe is not unique")
    return keys


def validate_raw_generation_row_v5(row: dict[str, Any], question: dict[str, Any]) -> None:
    """Frozen validator mirrored over the nine-role universe, plus the notool invariants."""
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
    role = str(row["checkpoint_role"])
    if role not in ROLE_NAMES_V5 and role not in SENSITIVITY_ROLES:
        raise ValueError("raw generation row has unknown checkpoint role")
    response = str(row["response"])
    if row["raw_output_sha256"] != hashlib.sha256(response.encode("utf-8")).hexdigest():
        raise ValueError("raw generation response hash mismatch")
    metadata = row["tool_metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("tool_metadata must be a mapping")
    if is_notool_role(role):
        if row.get("generation_mode") != GENERATION_MODE_NOTOOL:
            raise ValueError("notool row does not declare generation_mode=notool")
        if row.get("schema_version") != NOTOOL_SCHEMA_VERSION:
            raise ValueError("notool row has the wrong trajectory schema version")
        for field in TOOL_META_ZERO_FIELDS:
            if int(float(metadata.get(field, 0))) != 0:
                raise ValueError(f"notool row reports nonzero {field}")
        if "turn_credit_ledger" in metadata:
            raise ValueError("notool row carries a tool-loop ledger")
        messages = row["messages"]
        if any(str(message.get("role")) == "tool" for message in messages):
            raise ValueError("notool row carries a tool turn")
        if len(messages) != 1 or messages[0].get("role") != "assistant" or messages[0].get("content") != response:
            raise ValueError("notool row must contain exactly one assistant message equal to the response")
        if row["termination_reason"] not in NOTOOL_TERMINATION_REASONS:
            raise ValueError(f"notool termination reason is not in {NOTOOL_TERMINATION_REASONS}")
        stats = tool_tag_stats(response)
        if bool(row.get("tool_tag_emitted")) != stats["tool_tag_emitted"] or int(row.get("tool_tag_count", -1)) != stats["tool_tag_count"]:
            raise ValueError("notool tool-tag fields disagree with the frozen detector")
    elif row.get("generation_mode", GENERATION_MODE_TIR) != GENERATION_MODE_TIR:
        raise ValueError("tir role row declares a foreign generation mode")


def validated_resume_rows_v5(path: Path, questions: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    accepted: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id"))
        if question_id not in questions:
            raise ValueError(f"resume shard contains unknown question ID: {question_id}")
        validate_raw_generation_row_v5(row, questions[question_id])
        key = canonical_key(row)
        if key in accepted:
            raise ValueError(f"duplicate resume key: {key}")
        accepted[key] = row
    return accepted


# --------------------------------------------------------------------------- #
# notool generation: chat template without tools, one generation, no loop
# --------------------------------------------------------------------------- #


def notool_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    """Exactly what veRL's ``single_turn_agent`` tokenized during E3-NoTool training."""
    from verl.utils.chat_template import apply_chat_template

    ids = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        tools=None,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    return [int(item) for item in ids]


def classify_notool_termination(finish_reason: Any, response_token_count: int, budget: int) -> str:
    reason_type = finish_reason.get("type") if isinstance(finish_reason, dict) else str(finish_reason)
    if reason_type == "length" or response_token_count >= budget:
        return "length"
    return "final_answer"


async def generate_notool_tokens(endpoint: str, prompt_ids: list[int], sampling_params: dict[str, Any]) -> tuple[list[int], Any]:
    """One native SGLang ``/generate`` call; returns output ids and the raw finish reason."""
    import aiohttp

    params = to_sglang_sampling_params(sampling_params)
    payload = {"input_ids": prompt_ids, "sampling_params": params}
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint.rstrip("/") + "/generate", json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"SGLang infrastructure error HTTP {response.status}: {body[:2000]}")
    value = json.loads(body)
    token_ids = value.get("output_ids")
    if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
        raise RuntimeError("SGLang response did not contain integer output_ids")
    return token_ids, value.get("meta_info", {}).get("finish_reason")


async def generate_one_v5(
    tokenizer: Any,
    endpoint: str,
    question: dict[str, Any],
    role: str,
    setting: str,
    sample_index: int,
    checkpoint_identity: dict[str, Any],
    manifest_sha256: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not is_notool_role(role):
        raise ValueError(f"v5 generates notool roles only, got {role}")
    seed = derive_sample_seed(question["question_id"], sample_index)
    prompt = str(question["question"]) + config["generation"]["prompt_suffix"]
    budget = int(config["generation"]["max_response_tokens_total"])
    prompt_ids = notool_prompt_ids(tokenizer, prompt)
    try:
        token_ids, finish_reason = await generate_notool_tokens(endpoint, prompt_ids, _sampling_params(config, setting, seed))
    except Exception as exc:
        raise RuntimeError(f"infrastructure/model-serving failure for {(role, setting, question['question_id'], sample_index)}") from exc
    response_ids = token_ids[:budget]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    stats = tool_tag_stats(response)
    row = {
        "protocol_version": PROTOCOL_VERSION_V5,
        "schema_version": NOTOOL_SCHEMA_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "freeze_manifest_sha256": manifest_sha256,
        "checkpoint_role": role,
        "checkpoint_identity": checkpoint_identity,
        "generation_mode": GENERATION_MODE_NOTOOL,
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
        "prompt_token_ids_sha256": hashlib.sha256(b"".join(int(item).to_bytes(4, "big", signed=False) for item in prompt_ids)).hexdigest(),
        "prompt_token_count": len(prompt_ids),
        "messages": [{"role": "assistant", "content": response}],
        "response": response,
        "response_token_count": len(response_ids),
        "response_mask": [1] * len(response_ids),
        "termination_reason": classify_notool_termination(finish_reason, len(response_ids), budget),
        "tool_tag_emitted": bool(stats["tool_tag_emitted"]),
        "tool_tag_count": int(stats["tool_tag_count"]),
        "tool_metadata": {
            "tool_call_counts": 0,
            "tool_success_count": 0,
            "tool_error_count": 0,
            "tool_parse_error_count": 0,
            "num_turns": 2,
            "generation_mode": GENERATION_MODE_NOTOOL,
            "sglang_finish_reason": finish_reason,
            "raw_output_token_count": len(token_ids),
            "tool_tag_stats": stats,
        },
        "infrastructure_status": "ok",
        "raw_output_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
    }
    validate_raw_generation_row_v5(row, question)
    return row


def _require_approval() -> None:
    if os.environ.get(APPROVAL_ENV) != "YES":
        raise PermissionError(f"{APPROVAL_ENV}=YES is required (plans/M9.md §7 step 4 authorization)")


def _protocol_role_entries(live: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    protocol = load_json(PROTOCOL_V5)
    live = resolved_notool_roles() if live is None else live
    entries: dict[str, dict[str, Any]] = {}
    for role in NOTOOL_ROLE_NAMES:
        expected = dict(live[role])
        expected["state"], expected["role"], expected["generation_mode"] = "validated", role, GENERATION_MODE_NOTOOL
        entry = protocol["checkpoints"][role]
        if expected != entry:
            raise ValueError(f"live {role} binding differs from the frozen v5 protocol")
        entries[role] = entry
    return entries


DEFAULT_CONCURRENCY = 16


async def generate_shard_rows(
    tokenizer: Any,
    endpoint: str,
    role: str,
    setting: str,
    questions: list[dict[str, Any]],
    checkpoint_identity: dict[str, Any],
    manifest_sha256: str,
    config: dict[str, Any],
    output: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    """Fill one (role, setting, source) shard with bounded request concurrency.

    Requests overlap only while awaiting SGLang; every row is validated and appended with
    O_APPEND + fsync from the single event-loop thread, so the file never interleaves and the
    resume semantics (key-based, order-free) are unchanged.  Missing keys are still walked in
    the frozen panel order; completed rows are appended in completion order.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    question_map = {row["question_id"]: row for row in questions}
    existing = validated_resume_rows_v5(output, question_map)
    pending = [
        (question, sample_index)
        for question in questions
        for sample_index in setting_sample_indices(setting)
        if (role, setting, question["question_id"], sample_index, derive_sample_seed(question["question_id"], sample_index)) not in existing
    ]
    semaphore = asyncio.Semaphore(concurrency)
    generated = 0

    async def one(question: dict[str, Any], sample_index: int) -> None:
        nonlocal generated
        async with semaphore:
            row = await generate_one_v5(tokenizer, endpoint, question, role, setting, sample_index, checkpoint_identity, manifest_sha256, config)
        append_jsonl_fsync(output, row)
        existing[canonical_key(row)] = row
        generated += 1

    tasks = [asyncio.create_task(one(question, sample_index)) for question, sample_index in pending]
    try:
        for task in asyncio.as_completed(tasks):
            await task
    finally:
        for task in tasks:
            task.cancel()
    expected_count = len(questions) * len(setting_sample_indices(setting))
    if len(existing) != expected_count:
        raise AssertionError(f"shard completion mismatch: {len(existing)} != {expected_count}")
    return {
        "expected": expected_count,
        "existing_before": expected_count - generated,
        "generated": generated,
        "concurrency": concurrency,
        "output": str(output),
        "output_sha256": sha256(output),
    }


async def generate_role_v5(
    role: str, endpoint: str, run_dir: Path, limit_questions: int | None = None, concurrency: int = DEFAULT_CONCURRENCY
) -> dict[str, Any]:
    """Generate every (setting, source) shard of one notool role against one served model."""
    if not is_notool_role(role):
        raise ValueError(f"unknown notool role: {role}")
    _require_approval()
    config = validate_frozen_runtime_config()
    verify_v5()
    entry = _protocol_role_entries()[role]
    server = SGLangHTTPServer(endpoint)
    await server.validate_model_path(Path(entry["inference_locator"]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(COMMON_TOKENIZER)
    manifest_sha256 = sha256(MANIFEST_V5)
    panel = load_panel_questions()
    shards: dict[str, Any] = {}
    for setting in SETTINGS:
        for source in SOURCES:
            questions = [row for row in panel if row["source"] == source]
            if limit_questions is not None:
                questions = questions[:limit_questions]
            output = run_dir / "generation" / role / setting / f"{source}.jsonl"
            shards[f"{setting}/{source}"] = await generate_shard_rows(
                tokenizer, endpoint, role, setting, questions, entry, manifest_sha256, config, output, concurrency
            )
    return {"status": "completed", "role": role, "generation_mode": GENERATION_MODE_NOTOOL, "limit_questions": limit_questions, "shards": shards}


# --------------------------------------------------------------------------- #
# scoring (mirror of eval.evaluate.score_trajectory with the v5 validator)
# --------------------------------------------------------------------------- #


def score_trajectory_v5(raw: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    validate_raw_generation_row_v5(raw, question)
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
    if is_notool_role(str(raw["checkpoint_role"])):
        if float(score["n_tool_calls"]) != 0.0 or float(score["truncated"]) != 0.0:
            raise AssertionError("notool reward reports tool activity")
        truncated = raw["termination_reason"] == "length"
    else:
        truncated = bool(score["truncated"]) or bool(tool_metadata.get("turn_credit_ledger", {}).get("trajectory_truncated", False))
    enriched = dict(raw)
    enriched.update(
        {
            "reward": score,
            "verifier_outcome": strict,
            "correct": bool(strict["correct"]),
            "format_ok": bool(score["format_ok"]),
            "invalid": bool(score["invalid"]),
            "truncated": truncated,
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
    scored = [score_trajectory_v5(row, questions[row["question_id"]]) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"input": str(input_path), "output": str(output_path), "rows": len(scored), "output_sha256": sha256(output_path)}


# --------------------------------------------------------------------------- #
# downstream: completeness, taxonomy proposals, pairing, bootstrap, metrics
# --------------------------------------------------------------------------- #


def completeness_v5(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed: set[tuple[str, str, str, int, int]] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        validate_scored_row(row)
        key = canonical_key(row)
        if key in observed:
            raise ValueError(f"duplicate canonical scored key: {key}")
        observed.add(key)
        counts[f"{row['checkpoint_role']}:{row['evaluation_setting']}:{row['source']}"] += 1
    expected = set(expected_keys_v5())
    missing, unexpected = sorted(expected - observed), sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(f"notool key universe incomplete: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return {
        "expected": len(expected),
        "observed": len(observed),
        "counts": dict(sorted(counts.items())),
        "complete": True,
        "exact_key_digest": hashlib.sha256(json.dumps(sorted(observed), separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator if denominator else None}


def notool_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mechanism metrics for notool rows (plan §3/§4): no tool ledger exists, so no tool fields."""
    values = list(rows)
    total = len(values)
    if any(int(float(row["reward"]["n_tool_calls"])) != 0 for row in values):
        raise ValueError("notool behavior called on rows with tool calls")
    tokens = sum(int(row["response_token_count"]) for row in values)
    return {
        "trajectory_count": total,
        "generation_mode": GENERATION_MODE_NOTOOL,
        "strict_correct": _rate(sum(bool(row["correct"]) for row in values), total),
        "response_tokens": {"sum": tokens, "denominator": total, "mean": tokens / total if total else None},
        "truncation_rate": _rate(sum(bool(row["truncated"]) for row in values), total),
        "boxed_format_failure_rate": _rate(sum(not bool(row["format_ok"]) for row in values), total),
        "invalid_rate": _rate(sum(bool(row["invalid"]) for row in values), total),
        "tool_tag_emitted_rate": _rate(sum(bool(row.get("tool_tag_emitted")) for row in values), total),
        "tool_tag_total": sum(int(row.get("tool_tag_count", 0)) for row in values),
        "tool_calls": {"sum": 0, "denominator": total, "value": 0.0 if total else None},
    }


def paired_delta_by_stratum(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Greedy paired accuracy by frozen panel stratum (MATH level / source); plan §8 case A (i)."""
    before = {row["question_id"]: row for row in before_rows if row["evaluation_setting"] == "greedy"}
    after = {row["question_id"]: row for row in after_rows if row["evaluation_setting"] == "greedy"}
    if set(before) != set(after) or len(before) != 760:
        raise ValueError("stratum decomposition requires the complete paired greedy panel")
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for question_id in sorted(before):
        stratum = str(before[question_id]["stratum"])
        groups[stratum].append((bool(before[question_id]["correct"]), bool(after[question_id]["correct"])))
    result: dict[str, Any] = {}
    for stratum, pairs in sorted(groups.items()):
        n = len(pairs)
        before_correct = sum(left for left, _ in pairs)
        after_correct = sum(right for _, right in pairs)
        result[stratum] = {
            "denominator": n,
            "before_correct": before_correct,
            "after_correct": after_correct,
            "before_accuracy": before_correct / n,
            "after_accuracy": after_correct / n,
            "fixed_failures": sum((not left) and right for left, right in pairs),
            "new_failures": sum(left and (not right) for left, right in pairs),
            "accuracy_delta_pt": (after_correct - before_correct) / n * 100,
        }
    return result


def training_curves() -> dict[str, Any]:
    from rl.compare_e4 import early_metrics, final_metrics, validation_curve

    result: dict[str, Any] = {}
    for name, run_dir in (("e3", E3_RUN), ("e3_notool", NOTOOL_RUN)):
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
    result["note"] = "fixed-100 greedy MATH500 panel, n=100 (SE ~4.4pt at p=0.73); plan §4 mechanism metric, not a verdict"
    return result


def classify_preregistered_case_m9(greedy_delta: dict[str, Any], d_start: float | None) -> dict[str, Any]:
    """Mechanical reading of plans/M9.md §8 with Δ = E3-NoTool − E3 (greedy FULL760)."""
    point, lower, upper = greedy_delta["point_estimate"], greedy_delta["lower"], greedy_delta["upper"]
    if point is None or lower is None or upper is None:
        return {"case": "undefined", "reason": "zero-denominator bootstrap"}
    point_pt, lower_pt, upper_pt = point * 100, lower * 100, upper * 100
    ci_crosses_zero = lower <= 0 <= upper
    if point_pt <= -PREREGISTERED_EFFECT_PT and not ci_crosses_zero:
        case = "A"
    elif point_pt >= PREREGISTERED_EFFECT_PT and not ci_crosses_zero:
        case = "C"
    elif ci_crosses_zero:
        case = "B"
    else:
        case = "outside_preregistered_table"
    d_start_pt = None if d_start is None else d_start * 100
    start_debt_binding = False
    if case == "A" and d_start_pt is not None and abs(point_pt) <= d_start_pt:
        case = "A_indistinguishable_from_start_point_debt"
        start_debt_binding = True
    return {
        "case": case,
        "delta_definition": "acc(e3notool_grpo_step_200) - acc(e3_grpo_baseline_step_200), greedy FULL760 paired",
        "greedy_delta_point_pt": point_pt,
        "greedy_delta_ci_pt": [lower_pt, upper_pt],
        "ci_crosses_zero": ci_crosses_zero,
        "effect_threshold_pt": PREREGISTERED_EFFECT_PT,
        "d_start_pt": d_start_pt,
        "d_start_definition": "acc(raw_base_model_notool) - acc(sft_start_notool), greedy FULL760",
        "start_point_debt_rule_binding": start_debt_binding,
        "interpretation_guardrail": (
            "plan §8: case A must be decomposed by level/source and may not be written as a net tool contribution "
            "when |Δ| ≤ D_start; §11 (2026-09-06) forbids citing a 'conservative direction' for cases B/C because the "
            "smoke showed the start point favours NoTool on the training distribution"
        ),
    }


def _tool_tag_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['checkpoint_role']}:{row['evaluation_setting']}"].append(row)
    return {
        key: {
            "trajectory_count": len(values),
            "tool_tag_emitted_rate": _rate(sum(bool(row.get("tool_tag_emitted")) for row in values), len(values)),
            "tool_tag_total": sum(int(row.get("tool_tag_count", 0)) for row in values),
        }
        for key, values in sorted(grouped.items())
    }


def downstream(run_dir: Path) -> dict[str, Any]:
    verify_v5()
    reuse_m7 = verify_m7_canonical_ledger()
    reuse_m8 = verify_m8_canonical_ledger()
    questions = {row["question_id"]: row for row in load_panel_questions()}
    scored_rows: list[dict[str, Any]] = []
    for role in NOTOOL_ROLE_NAMES:
        for setting in SETTINGS:
            for source in SOURCES:
                scored_rows.extend(_read_rows(run_dir / "scored" / role / setting / f"{source}.jsonl"))
    completeness = completeness_v5(scored_rows)
    labeled = [enrich_with_coarse_proposal(row, str(questions[row["question_id"]]["answer"])) for row in scored_rows]
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled:
        by_role[str(row["checkpoint_role"])].append(row)
    taxonomy_dir = run_dir / "taxonomy"
    _write_jsonl(
        taxonomy_dir / "coarse_proposals.jsonl",
        (
            {key: row[key] for key in ("checkpoint_role", "evaluation_setting", "source", "question_id", "sample_index", "sample_seed", "correct", "primary_failure", "secondary_labels", "label_source", "adjudication_rationale", "raw_evidence_pointer")}
            for row in labeled
        ),
    )
    coarse_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in labeled:
        label = "correct" if row["correct"] else str(row["primary_failure"])
        coarse_counts[f"{row['checkpoint_role']}:{row['evaluation_setting']}:{row['source']}"][label] += 1
        coarse_counts[f"{row['checkpoint_role']}:{row['evaluation_setting']}:FULL760"][label] += 1
    _write_json(
        taxonomy_dir / "coarse_metrics.json",
        {
            "scope": "deterministic coarse proposals only; no blinded audit for notool roles (plan §3); tool categories are structurally empty",
            "taxonomy_version": "toolcredit_failure_taxonomy_v1",
            "trajectory_count": len(labeled),
            "groups": {key: dict(sorted(counter.items())) for key, counter in sorted(coarse_counts.items())},
        },
    )
    reused = load_m7_labeled_rows((E3_ROLE, RAW_ROLE))
    universe = {**reused, **by_role}
    transitions_dir = run_dir / "transitions"
    comparisons: dict[str, Any] = {}
    for pairing in PAIRINGS:
        name = pairing["question"]
        comparisons[name] = {
            **transitions_and_bootstrap(universe[pairing["before_role"]], universe[pairing["after_role"]], pairing["before_role"], pairing["after_role"], transitions_dir, name),
            "priority": pairing["priority"],
            "before_role": pairing["before_role"],
            "after_role": pairing["after_role"],
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[f"{row['checkpoint_role']}:{row['evaluation_setting']}:{row['source']}"].append(row)
        grouped[f"{row['checkpoint_role']}:{row['evaluation_setting']}:FULL760"].append(row)
    metrics_dir = run_dir / "metrics"
    notool_pass = pass_metrics(scored_rows)
    behavior = {key: notool_behavior(values) for key, values in sorted(grouped.items())}
    _write_json(metrics_dir / "pass_at_k.json", {"protocol_version": PROTOCOL_VERSION_V5, "metrics": notool_pass})
    _write_json(metrics_dir / "notool_behavior.json", {"protocol_version": PROTOCOL_VERSION_V5, "groups": behavior})
    _write_json(metrics_dir / "tool_tag_rates.json", {"protocol_version": PROTOCOL_VERSION_V5, "groups": _tool_tag_rates(scored_rows)})
    _write_json(metrics_dir / "completeness.json", completeness)

    m7_pass = load_json(M7_CANONICAL_RUN / "metrics/pass_at_k.json")["metrics"]
    m8_pass = load_json(M8_CANONICAL_RUN / "metrics/pass_at_k.json")["metrics"]
    m7_behavior = load_json(M7_CANONICAL_RUN / "metrics/tool_behavior.json")["groups"]
    greedy_acc = {
        **{role: m7_pass[f"{role}:greedy:FULL760"]["pass@1"] for role in ROLE_NAMES},
        ROLE_V4: m8_pass[f"{ROLE_V4}:greedy:FULL760"]["pass@1"],
        **{role: notool_pass[f"{role}:greedy:FULL760"]["pass@1"] for role in NOTOOL_ROLE_NAMES},
    }
    d_start = greedy_acc["raw_base_model_notool"]["value"] - greedy_acc["sft_start_notool"]["value"]
    primary = comparisons["tool_net_contribution_at_rl_endpoint"]
    greedy_delta = primary["bootstrap"]["settings"]["greedy"]["FULL760"]["accuracy_delta"]
    gate = load_json(NOTOOL_RUN / "analysis/formal_completion_gate.json")
    headline = {
        "protocol_version": PROTOCOL_VERSION_V5,
        "reused_m7_canonical": reuse_m7,
        "reused_m8_canonical": reuse_m8,
        "completeness": completeness,
        "greedy_full760_pass_at_1": greedy_acc,
        "sampled_full760_pass_at_k": {
            **{role: m7_pass[f"{role}:sampled:FULL760"] for role in ROLE_NAMES},
            ROLE_V4: m8_pass[f"{ROLE_V4}:sampled:FULL760"],
            **{role: notool_pass[f"{role}:sampled:FULL760"] for role in NOTOOL_ROLE_NAMES},
        },
        "d_start": {
            "definition": "acc(raw_base_model_notool) - acc(sft_start_notool), greedy FULL760",
            "value": d_start,
            "value_pt": d_start * 100,
            "paired_bootstrap": comparisons["start_point_debt_D_start"]["bootstrap"]["settings"]["greedy"]["FULL760"]["accuracy_delta"],
        },
        "paired_greedy_full760": {
            name: {
                "priority": value["priority"],
                "before_role": value["before_role"],
                "after_role": value["after_role"],
                **{key: value["summary"]["settings"]["greedy"]["FULL760"][key] for key in ("denominator", "fixed_failures", "new_failures", "unchanged_correct", "unchanged_failed", "accuracy_delta")},
            }
            for name, value in comparisons.items()
        },
        "paired_greedy_full760_bootstrap": {name: value["bootstrap"]["settings"]["greedy"]["FULL760"] for name, value in comparisons.items()},
        "paired_sampled_full760_bootstrap": {name: value["bootstrap"]["settings"]["sampled"]["FULL760"] for name, value in comparisons.items()},
        "primary_by_stratum": paired_delta_by_stratum(universe[E3_ROLE], universe[ROLE_V5]),
        "primary_by_source_bootstrap": {source: primary["bootstrap"]["settings"]["greedy"][source]["accuracy_delta"] for source in SOURCES},
        "notool_behavior_greedy_full760": {role: behavior[f"{role}:greedy:FULL760"] for role in NOTOOL_ROLE_NAMES},
        "e3_tir_behavior_greedy_full760": {
            key: m7_behavior[f"{E3_ROLE}:greedy:FULL760"][key]
            for key in ("tool_calls", "tool_call_count_fractions", "truncation_rate", "per_call_execution_success", "response_tokens", "no_tool")
        },
        "tool_tag_rates": _tool_tag_rates(scored_rows),
        "training_curves": training_curves(),
        "training_behavior_first_25": gate["behavior_first_25"],
        "training_behavior_last_25": gate["behavior_last_25"],
        "training_zero_variance_groups": {"first_25": gate["zero_variance_groups_first_25"], "last_25": gate["zero_variance_groups_last_25"]},
        "training_recovery_events": gate["recovery_events"],
        "preregistered_case": classify_preregistered_case_m9(greedy_delta, d_start),
        "scientific_interpretation_performed": False,
    }
    _write_json(metrics_dir / "m9_headline.json", headline)
    return headline


def write_closure(run_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "status.json"} and "server_logs" not in path.parts and "stage_logs" not in path.parts:
            entries.append(f"{sha256(path)}  {path.relative_to(run_dir)}")
    (run_dir / "hashes.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    status = {
        "run_name": run_dir.name,
        "protocol_version": PROTOCOL_VERSION_V5,
        "freeze_manifest_sha256": sha256(MANIFEST_V5),
        "status": "notool_evaluation_completed",
        "completed_at": utc_now(),
        "hashes_sha256": sha256(run_dir / "hashes.sha256"),
        "artifact_count": len(entries),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def preflight(run_dir: Path) -> dict[str, Any]:
    import torch

    entries = _protocol_role_entries()
    report = {
        "protocol": verify_v5(),
        "m7_canonical": verify_m7_canonical_ledger(),
        "m8_canonical": verify_m8_canonical_ledger(),
        "roles": {role: {"inference_locator": entry["inference_locator"], "weight_role": entry["weight_role"]} for role, entry in entries.items()},
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("notool evaluation requires exactly one visible CUDA GPU")
    free_gpu, total_gpu = torch.cuda.mem_get_info(0)
    disk_free = shutil.disk_usage(ROOT).free
    if disk_free < 20 * 1024**3:
        raise RuntimeError("notool evaluation disk gate failed: < 20 GiB free")
    report.update(
        {
            "gpu_free_gib": round(free_gpu / 2**30, 2),
            "gpu_total_gib": round(total_gpu / 2**30, 2),
            "disk_free_gib": round(disk_free / 2**30, 2),
            "expected_new_keys": len(expected_keys_v5()),
            "run_dir": str(run_dir),
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def decoded_prompt(question_id: str) -> dict[str, Any]:
    """Debug aid: the exact decoded notool prompt for one panel question."""
    from transformers import AutoTokenizer

    config = validate_frozen_runtime_config()
    question = next(row for row in load_panel_questions() if row["question_id"] == question_id)
    tokenizer = AutoTokenizer.from_pretrained(COMMON_TOKENIZER)
    ids = notool_prompt_ids(tokenizer, str(question["question"]) + config["generation"]["prompt_suffix"])
    return {"question_id": question_id, "prompt_token_count": len(ids), "decoded": tokenizer.decode(ids, skip_special_tokens=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--frozen-at", required=True)
    sub.add_parser("verify")
    pre = sub.add_parser("preflight")
    pre.add_argument("--run-dir", type=Path, required=True)
    gen = sub.add_parser("generate-role")
    gen.add_argument("--role", choices=NOTOOL_ROLE_NAMES, required=True)
    gen.add_argument("--endpoint", default="http://127.0.0.1:30000")
    gen.add_argument("--run-dir", type=Path, required=True)
    gen.add_argument("--limit-questions", type=int, default=None)
    gen.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    down = sub.add_parser("downstream")
    down.add_argument("--run-dir", type=Path, required=True)
    closure = sub.add_parser("closure")
    closure.add_argument("--run-dir", type=Path, required=True)
    prompt = sub.add_parser("prompt")
    prompt.add_argument("--question-id", required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        from eval.checkpoints_v5 import materialize_e3notool

        result: Any = materialize_e3notool()
    elif args.command == "freeze":
        from eval.build_diagnostic_freeze_v5 import build

        build(args.frozen_at)
        from eval.verify_diagnostic_freeze_v5_semantics import verify_semantic_diff

        result = verify_semantic_diff()
    elif args.command == "verify":
        from eval.verify_diagnostic_freeze_v5_semantics import verify_semantic_diff

        result = verify_semantic_diff()
    elif args.command == "preflight":
        result = preflight(args.run_dir.resolve())
    elif args.command == "generate-role":
        result = asyncio.run(generate_role_v5(args.role, args.endpoint, args.run_dir.resolve(), args.limit_questions, args.concurrency))
    elif args.command == "score":
        result = score_shard(args.input, args.output)
    elif args.command == "downstream":
        result = downstream(args.run_dir.resolve())
    elif args.command == "prompt":
        result = decoded_prompt(args.question_id)
    else:
        result = write_closure(args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
