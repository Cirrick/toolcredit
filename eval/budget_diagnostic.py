"""E3-only longer-budget diagnostic derived from the frozen M7 canonical run.

This module never changes the M7 denominator and never exposes the selected test
questions to a training path.  It selects the 70 canonical greedy E3 failures whose
frozen coarse primary was budget/truncation, records exact stopping mechanisms and
repeated-code tags, and reruns only those questions with a larger response/turn budget.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rewards.verifier import verify_answer
from rl.analyze_e4 import code_hash
from rl.custom.reward import compute_score
from rl.custom.turn_advantage import LEDGER_KEY

from eval.checkpoints import COMMON_TOKENIZER, ROOT, resolved_checkpoint_roles, sha256
from eval.generate import (
    SGLangHTTPServer,
    append_jsonl_fsync,
    build_m7_loop,
    canonical_key,
    derive_sample_seed,
    generate_one,
    load_json,
    load_panel_questions,
    validate_raw_generation_row,
)

CONFIG_PATH = ROOT / "eval/e3_budget_diagnostic_config.json"
ROLE = "e3_grpo_baseline_step_200"
SETTING = "greedy"
CANONICAL_RUN = ROOT / "eval/runs/m7_unified_eval_20260824_201032"
COARSE_PATH = CANONICAL_RUN / "taxonomy/coarse_proposals.jsonl"
EVALUATOR_VERSION = "toolcredit_e3_budget_diagnostic_v1"

TOKEN_TERMINATIONS = {"response_length", "tool_response_would_exceed_response_length"}
TURN_TERMINATIONS = {"assistant_turn_budget", "tool_turn_budget"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"partial or blank JSONL at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _replace_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale atomic-write temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _ledger(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("tool_metadata", {}).get(LEDGER_KEY)
    if not isinstance(value, dict):
        raise TypeError(f"trajectory has no structured ledger: {row.get('question_id')}")
    return value


def parsed_code_hashes(row: dict[str, Any]) -> list[str]:
    hashes: list[str] = []
    for turn in _ledger(row).get("assistant_turns", []):
        for call in turn.get("parsed_calls", []):
            if call.get("name") != "code_interpreter":
                continue
            try:
                arguments = json.loads(call.get("arguments", ""))
            except (json.JSONDecodeError, TypeError):
                continue
            code = arguments.get("code") if isinstance(arguments, dict) else None
            if isinstance(code, str):
                hashes.append(code_hash(code))
    return hashes


def stopping_mechanism(row: dict[str, Any]) -> str:
    reason = str(_ledger(row).get("termination_reason") or row.get("termination_reason"))
    if reason in TOKEN_TERMINATIONS:
        return "response_token_limit"
    if reason in TURN_TERMINATIONS:
        return "assistant_or_tool_turn_limit"
    raise ValueError(f"selected row has a non-budget termination {reason!r}: {row['question_id']}")


def diagnostic_class(row: dict[str, Any]) -> dict[str, Any]:
    mechanism = stopping_mechanism(row)
    hashes = parsed_code_hashes(row)
    repeated = bool(hashes) and len(set(hashes)) < len(hashes)
    exclusive = "repeated_normalized_code" if repeated else mechanism
    tags = [mechanism, *(["repeated_normalized_code"] if repeated else [])]
    return {
        "stopping_mechanism": mechanism,
        "repeated_normalized_code": repeated,
        "exclusive_class": exclusive,
        "mechanism_tags": tags,
        "normalized_code_hashes": hashes,
    }


def _canonical_scored_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    base = CANONICAL_RUN / f"scored/{ROLE}/{SETTING}"
    for path in sorted(base.glob("*.jsonl")):
        for row in _read_jsonl(path):
            question_id = str(row["question_id"])
            if question_id in rows:
                raise ValueError(f"duplicate canonical E3 greedy question: {question_id}")
            rows[question_id] = row
    if len(rows) != 760:
        raise ValueError(f"canonical E3 greedy cardinality drifted: {len(rows)} != 760")
    return rows


def selected_original_rows(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = load_json(CONFIG_PATH) if config is None else config
    selected_ids = {
        str(row["question_id"])
        for row in _read_jsonl(COARSE_PATH)
        if row.get("checkpoint_role") == ROLE
        and row.get("evaluation_setting") == SETTING
        and row.get("primary_failure") == config["selection"]["parent_primary_failure"]
    }
    expected = int(config["selection"]["expected_count"])
    if len(selected_ids) != expected:
        raise ValueError(f"budget diagnostic selection drifted: {len(selected_ids)} != {expected}")
    canonical = _canonical_scored_rows()
    rows = [canonical[question_id] for question_id in sorted(selected_ids)]
    for row in rows:
        if bool(row["correct"]):
            raise ValueError(f"selected canonical row is correct: {row['question_id']}")
        if not bool(row["truncated"]):
            raise ValueError(f"selected canonical row is not truly truncated: {row['question_id']}")
        diagnostic_class(row)
    return rows


def validate_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_json(CONFIG_PATH) if config is None else config
    parent = load_json(ROOT / config["parent_eval_config"])
    if config.get("checkpoint_role") != ROLE or config.get("evaluation_setting") != SETTING:
        raise ValueError("budget diagnostic must remain E3 greedy only")
    if config.get("parent_canonical_run") != str(CANONICAL_RUN.relative_to(ROOT)):
        raise ValueError("budget diagnostic parent run drifted")
    if config.get("selection", {}).get("training_use_forbidden") is not True:
        raise ValueError("diagnostic test questions must be forbidden from training use")
    generation = config["generation"]
    parent_generation = parent["generation"]
    unchanged = (
        "enable_thinking",
        "prompt_suffix",
        "max_parallel_tool_calls",
        "max_tool_response_length",
        "tool_response_truncate_side",
        "sandbox_timeout_seconds",
        "greedy",
    )
    drift = {key: (generation.get(key), parent_generation.get(key)) for key in unchanged if generation.get(key) != parent_generation.get(key)}
    if drift:
        raise ValueError(f"non-budget generation semantics drifted: {drift}")
    if int(generation["max_response_tokens_total"]) != 6144:
        raise ValueError("diagnostic response budget must be exactly 6144")
    if int(generation["max_user_tool_turns"]) != 6 or int(generation["max_assistant_turns"]) != 7:
        raise ValueError("diagnostic turn budgets must be exactly 6 tool / 7 assistant turns")
    if int(parent_generation["max_response_tokens_total"]) != 3072:
        raise ValueError("parent response budget drifted from canonical M7")
    if int(parent_generation["max_user_tool_turns"]) != 4 or int(parent_generation["max_assistant_turns"]) != 5:
        raise ValueError("parent turn budgets drifted from canonical M7")
    if config["classification"]["exclusive_precedence"] != [
        "repeated_normalized_code",
        "assistant_or_tool_turn_limit",
        "response_token_limit",
    ]:
        raise ValueError("diagnostic class precedence drifted")
    return config


def _verify_parent_integrity() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("hashes.sha256", "analysis_hashes.sha256"):
        path = CANONICAL_RUN / name
        completed = subprocess.run(
            ["sha256sum", "--quiet", "-c", name],
            cwd=CANONICAL_RUN,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"parent canonical integrity failed for {name}: {completed.stdout}{completed.stderr}")
        result[name] = sha256(path)
    return result


def _gpu_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi preflight failed: {completed.stderr}")
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        index, name, total_mib, free_mib = [part.strip() for part in line.split(",", maxsplit=3)]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "total_mib": int(total_mib),
                "free_mib": int(free_mib),
                "free_bytes": int(free_mib) * 1024 * 1024,
            }
        )
    if len(gpus) != 1:
        raise ValueError(f"diagnostic expects exactly one visible GPU, got {len(gpus)}")
    return {"gpus": gpus}


def selection_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for row in rows:
        classification = diagnostic_class(row)
        manifest.append(
            {
                "question_id": row["question_id"],
                "source": row["source"],
                "sample_index": 0,
                "sample_seed": int(row["sample_seed"]),
                "parent_canonical_key": list(canonical_key(row)),
                "parent_raw_output_sha256": row["raw_output_sha256"],
                "parent_response_token_count": int(row["response_token_count"]),
                "parent_termination_reason": row["termination_reason"],
                "parent_tool_call_count": int(float(row["reward"]["n_tool_calls"])),
                "parent_truncated": bool(row["truncated"]),
                "parent_correct": bool(row["correct"]),
                "training_use_forbidden": True,
                **classification,
            }
        )
    return manifest


def prepare(run_dir: Path) -> dict[str, Any]:
    config = validate_config()
    if run_dir.exists():
        raise FileExistsError(f"diagnostic run directory already exists: {run_dir}")
    rows = selected_original_rows(config)
    manifest = selection_manifest(rows)
    parent_integrity = _verify_parent_integrity()
    roles = resolved_checkpoint_roles(require_materialized=True)
    gpu = _gpu_snapshot()
    disk_free = shutil.disk_usage(ROOT).free
    minimum_gpu = int(config["resource_budget"]["minimum_free_gpu_bytes"])
    minimum_disk = int(config["resource_budget"]["minimum_free_disk_bytes"])
    if gpu["gpus"][0]["free_bytes"] < minimum_gpu:
        raise RuntimeError("insufficient free GPU memory for E3 budget diagnostic")
    if disk_free < minimum_disk:
        raise RuntimeError("insufficient free disk for E3 budget diagnostic")
    run_dir.mkdir(parents=True)
    panel_path = run_dir / "selection.jsonl"
    _write_jsonl_exclusive(panel_path, manifest)
    resolved = {
        "diagnostic_config": config,
        "diagnostic_config_sha256": sha256(CONFIG_PATH),
        "selection_sha256": sha256(panel_path),
        "parent_integrity": parent_integrity,
        "checkpoint_identity": roles[ROLE],
        "training_use_forbidden": True,
    }
    _write_json_exclusive(run_dir / "resolved_config.json", resolved)
    stopping = Counter(item["stopping_mechanism"] for item in manifest)
    exclusive = Counter(item["exclusive_class"] for item in manifest)
    repeated = sum(bool(item["repeated_normalized_code"]) for item in manifest)
    preflight = {
        "status": "ready_for_generation",
        "selected": len(manifest),
        "stopping_mechanisms": dict(sorted(stopping.items())),
        "exclusive_classes": dict(sorted(exclusive.items())),
        "repeated_normalized_code": repeated,
        "resource_snapshot": {**gpu, "disk_free_bytes": disk_free},
        "parent_canonical_integrity_verified": True,
        "gpu_generation_started": False,
    }
    _write_json_exclusive(run_dir / "preflight.json", preflight)
    _write_json_exclusive(run_dir / "status.json", {"state": "prepared", "generated": 0, "expected": len(manifest)})
    return preflight


def _load_run_inputs(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = load_json(run_dir / "resolved_config.json")
    config = validate_config(resolved["diagnostic_config"])
    if resolved.get("diagnostic_config_sha256") != sha256(CONFIG_PATH):
        raise ValueError("diagnostic config changed after preflight")
    manifest = _read_jsonl(run_dir / "selection.jsonl")
    if resolved.get("selection_sha256") != sha256(run_dir / "selection.jsonl"):
        raise ValueError("diagnostic selection changed after preflight")
    if len(manifest) != int(config["selection"]["expected_count"]):
        raise ValueError("diagnostic selection cardinality changed after preflight")
    return config, manifest


def validate_diagnostic_raw_row(row: dict[str, Any], selected: dict[str, dict[str, Any]], config_hash: str) -> None:
    questions = {item["question_id"]: item for item in load_panel_questions()}
    question_id = str(row.get("question_id"))
    if question_id not in selected or question_id not in questions:
        raise ValueError(f"diagnostic raw row is outside the fixed selection: {question_id}")
    validate_raw_generation_row(row, questions[question_id])
    if row.get("protocol_version") != EVALUATOR_VERSION or row.get("diagnostic_scope") != "e3_budget_only":
        raise ValueError("diagnostic raw row protocol/scope mismatch")
    if row.get("diagnostic_config_sha256") != config_hash:
        raise ValueError("diagnostic raw row config identity mismatch")
    if row.get("parent_canonical_key") != selected[question_id]["parent_canonical_key"]:
        raise ValueError("diagnostic raw row parent key mismatch")
    if row.get("diagnostic_generation_budget") != {
        "max_response_tokens_total": 6144,
        "max_user_tool_turns": 6,
        "max_assistant_turns": 7,
    }:
        raise ValueError("diagnostic raw row budget mismatch")


async def generate(run_dir: Path, endpoint: str) -> dict[str, Any]:
    if os.environ.get("E3_BUDGET_DIAGNOSTIC_APPROVED") != "YES":
        raise PermissionError("explicit E3 budget diagnostic approval guard is absent")
    if not os.environ.get("TMUX"):
        raise RuntimeError("GPU diagnostic must run inside tmux")
    config, manifest = _load_run_inputs(run_dir)
    selected = {row["question_id"]: row for row in manifest}
    roles = resolved_checkpoint_roles(require_materialized=True)
    server = SGLangHTTPServer(endpoint)
    await server.validate_model_path(Path(roles[ROLE]["inference_locator"]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(COMMON_TOKENIZER)
    loop = build_m7_loop(tokenizer, server, config)
    questions = {row["question_id"]: row for row in load_panel_questions()}
    raw_path = run_dir / "raw.jsonl"
    config_hash = sha256(CONFIG_PATH)
    existing: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        for row in _read_jsonl(raw_path):
            validate_diagnostic_raw_row(row, selected, config_hash)
            question_id = str(row["question_id"])
            if question_id in existing:
                raise ValueError(f"duplicate diagnostic resume row: {question_id}")
            existing[question_id] = row
    for selection in manifest:
        question_id = selection["question_id"]
        if question_id in existing:
            continue
        question = questions[question_id]
        row = await generate_one(
            loop=loop,
            tokenizer=tokenizer,
            question=question,
            role=ROLE,
            setting=SETTING,
            sample_index=0,
            checkpoint_identity=roles[ROLE],
            protocol_version=EVALUATOR_VERSION,
            manifest_sha256=config_hash,
            config=config,
        )
        row.update(
            {
                "diagnostic_scope": "e3_budget_only",
                "diagnostic_config_sha256": config_hash,
                "parent_canonical_key": selection["parent_canonical_key"],
                "training_use_forbidden": True,
                "diagnostic_generation_budget": {
                    "max_response_tokens_total": 6144,
                    "max_user_tool_turns": 6,
                    "max_assistant_turns": 7,
                },
            }
        )
        validate_diagnostic_raw_row(row, selected, config_hash)
        append_jsonl_fsync(raw_path, row)
        existing[question_id] = row
        _replace_json(
            run_dir / "status.json",
            {"state": "generating", "generated": len(existing), "expected": len(manifest)},
        )
    if len(existing) != len(manifest):
        raise AssertionError(f"diagnostic generation incomplete: {len(existing)} != {len(manifest)}")
    result = {
        "status": "generation_completed",
        "expected": len(manifest),
        "observed": len(existing),
        "raw_sha256": sha256(raw_path),
    }
    _replace_json(run_dir / "status.json", {"state": "generated", **result})
    return result


def _score_long_row(row: dict[str, Any], answer: str) -> dict[str, Any]:
    ledger = _ledger(row)
    score = compute_score(
        data_source="toolcredit_math",
        solution_str=str(row["response"]),
        ground_truth=answer,
        extra_info={"id": row["question_id"], **row["tool_metadata"]},
    )
    strict = verify_answer(str(row["response"]), answer, strict_boxed=True)
    truncated = bool(score["truncated"]) or bool(ledger.get("trajectory_truncated", False))
    result = dict(row)
    result.update(
        {
            "reward": score,
            "verifier_outcome": strict,
            "strict_answer_correct": bool(strict["correct"]),
            "correct": bool(score["acc"]),
            "format_ok": bool(score["format_ok"]),
            "invalid": bool(score["invalid"]),
            "truncated": truncated,
            "correct_despite_truncation": bool(score["acc"]) and truncated,
            "long_budget_exhausted": str(ledger.get("termination_reason")) in TOKEN_TERMINATIONS | TURN_TERMINATIONS,
        }
    )
    return result


def residual_subtype(row: dict[str, Any]) -> str:
    reason = str(_ledger(row).get("termination_reason") or row.get("termination_reason"))
    if reason == "final_answer":
        return "completed_wrong_semantic"
    if reason in TOKEN_TERMINATIONS:
        return "persistent_response_token_limit"
    if reason in TURN_TERMINATIONS:
        return "persistent_assistant_or_tool_turn_limit"
    return "other_wrong_termination"


def _count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    values = Counter(str(row[key]) for row in rows)
    return dict(sorted(values.items()))


def _stable_examples(rows: list[dict[str, Any]], key: str, limit: int = 3) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(str(row["question_id"]))
    return {
        group: sorted(values, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:limit]
        for group, values in sorted(grouped.items())
    }


def _summary_markdown(metrics: dict[str, Any]) -> str:
    original = metrics["original_70"]
    outcome = metrics["long_budget_outcome"]
    lines = [
        "# E3 longer-budget diagnostic",
        "",
        "This run is diagnostic-only. Its 70 held-out questions are forbidden from training use and do not alter the M7 canonical denominator.",
        "",
        "## Original selection",
        "",
        f"- Selected canonical E3 greedy failures: **{original['count']}**",
        f"- Stopping mechanisms: `{json.dumps(original['stopping_mechanisms'], sort_keys=True)}`",
        f"- Exclusive classes: `{json.dumps(original['exclusive_classes'], sort_keys=True)}`",
        f"- Repeated normalized code tag: **{original['repeated_normalized_code']}**",
        "",
        "## Longer-budget result",
        "",
        "Budget changed from 3072 tokens / 4 tool turns / 5 assistant turns to 6144 / 6 / 7; all other generation and scoring semantics were held fixed.",
        "",
        f"- Strict project-correct after expansion: **{outcome['correct']}/{outcome['count']} ({outcome['correct_rate']:.4f})**",
        f"- Correct answer reached before a later budget stop: **{outcome['correct_despite_truncation']}**",
        f"- Still wrong: **{outcome['wrong']}**",
        f"- Residual subtypes: `{json.dumps(outcome['residual_subtypes'], sort_keys=True)}`",
        f"- Long-run termination reasons: `{json.dumps(outcome['termination_reasons'], sort_keys=True)}`",
        "",
        "A wrong-to-correct transition supports budget as the direct cause for that trajectory. A residual wrong is reported both under the requested binary reclassification and under a mechanistic subtype; persistent re-truncation is not silently called a completed semantic answer.",
        "",
    ]
    return "\n".join(lines)


def finalize(run_dir: Path) -> dict[str, Any]:
    config, manifest = _load_run_inputs(run_dir)
    selected = {row["question_id"]: row for row in manifest}
    raw_rows = _read_jsonl(run_dir / "raw.jsonl")
    config_hash = sha256(CONFIG_PATH)
    if len(raw_rows) != len(manifest):
        raise ValueError(f"diagnostic raw cardinality mismatch: {len(raw_rows)} != {len(manifest)}")
    observed: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        validate_diagnostic_raw_row(row, selected, config_hash)
        question_id = str(row["question_id"])
        if question_id in observed:
            raise ValueError(f"duplicate diagnostic question: {question_id}")
        observed[question_id] = row
    if set(observed) != set(selected):
        raise ValueError("diagnostic raw keys do not exactly match selection")
    questions = {row["question_id"]: row for row in load_panel_questions()}
    original = {row["question_id"]: row for row in selected_original_rows(config)}
    scored = [_score_long_row(observed[item["question_id"]], str(questions[item["question_id"]]["answer"])) for item in manifest]
    _write_jsonl_exclusive(run_dir / "scored.jsonl", scored)
    comparisons: list[dict[str, Any]] = []
    for row in scored:
        question_id = str(row["question_id"])
        parent = original[question_id]
        selection = selected[question_id]
        long_hashes = parsed_code_hashes(row)
        fixed = bool(row["correct"])
        subtype = "budget_direct_wrong_to_correct" if fixed else residual_subtype(row)
        comparisons.append(
            {
                "question_id": question_id,
                "source": row["source"],
                "original_exclusive_class": selection["exclusive_class"],
                "original_stopping_mechanism": selection["stopping_mechanism"],
                "original_repeated_normalized_code": selection["repeated_normalized_code"],
                "original_response_token_count": parent["response_token_count"],
                "original_tool_call_count": int(float(parent["reward"]["n_tool_calls"])),
                "original_termination_reason": parent["termination_reason"],
                "original_raw_output_sha256": parent["raw_output_sha256"],
                "long_correct": fixed,
                "long_strict_answer_correct": bool(row["strict_answer_correct"]),
                "long_response_token_count": row["response_token_count"],
                "long_tool_call_count": int(float(row["reward"]["n_tool_calls"])),
                "long_termination_reason": row["termination_reason"],
                "long_truncated": bool(row["truncated"]),
                "long_correct_despite_truncation": bool(row["correct_despite_truncation"]),
                "long_repeated_normalized_code": bool(long_hashes) and len(set(long_hashes)) < len(long_hashes),
                "long_raw_output_sha256": row["raw_output_sha256"],
                "original_text_is_long_prefix": str(row["response"]).startswith(str(parent["response"])),
                "budget_direct_supported": fixed,
                "requested_binary_reclassification": "budget_direct" if fixed else "semantic_or_policy_failure_after_budget_increase",
                "mechanistic_outcome": subtype,
                "training_use_forbidden": True,
            }
        )
    _write_jsonl_exclusive(run_dir / "comparisons.jsonl", comparisons)
    correct = sum(row["long_correct"] for row in comparisons)
    residual = [row for row in comparisons if not row["long_correct"]]
    metrics = {
        "scope": "diagnostic_only_not_m7_canonical",
        "training_use_forbidden": True,
        "budget_change": {
            "response_tokens": [3072, 6144],
            "max_user_tool_turns": [4, 6],
            "max_assistant_turns": [5, 7],
        },
        "original_70": {
            "count": len(manifest),
            "stopping_mechanisms": _count_by(manifest, "stopping_mechanism"),
            "exclusive_classes": _count_by(manifest, "exclusive_class"),
            "repeated_normalized_code": sum(bool(row["repeated_normalized_code"]) for row in manifest),
            "source_counts": _count_by(manifest, "source"),
        },
        "long_budget_outcome": {
            "count": len(comparisons),
            "correct": correct,
            "wrong": len(comparisons) - correct,
            "correct_rate": correct / len(comparisons),
            "correct_despite_truncation": sum(bool(row["long_correct_despite_truncation"]) for row in comparisons),
            "fixed_by_original_exclusive_class": dict(sorted(Counter(row["original_exclusive_class"] for row in comparisons if row["long_correct"]).items())),
            "fixed_by_source": dict(sorted(Counter(row["source"] for row in comparisons if row["long_correct"]).items())),
            "residual_subtypes": dict(sorted(Counter(row["mechanistic_outcome"] for row in residual).items())),
            "termination_reasons": dict(sorted(Counter(str(row["long_termination_reason"]) for row in comparisons).items())),
            "prefix_identity": {
                "matching": sum(bool(row["original_text_is_long_prefix"]) for row in comparisons),
                "denominator": len(comparisons),
            },
        },
        "stable_hash_examples": {
            "by_mechanistic_outcome": _stable_examples(comparisons, "mechanistic_outcome"),
            "by_original_exclusive_class": _stable_examples(comparisons, "original_exclusive_class"),
        },
    }
    _write_json_exclusive(run_dir / "metrics.json", metrics)
    summary = _summary_markdown(metrics)
    summary_path = run_dir / "summary.md"
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(summary)
    completed_status = {
        "state": "completed",
        "expected": len(manifest),
        "generated": len(raw_rows),
        "scored": len(scored),
        "correct_after_long_budget": correct,
        "wrong_after_long_budget": len(scored) - correct,
    }
    _replace_json(run_dir / "status.json", completed_status)
    hashed = (
        "resolved_config.json",
        "selection.jsonl",
        "preflight.json",
        "raw.jsonl",
        "scored.jsonl",
        "comparisons.jsonl",
        "metrics.json",
        "summary.md",
        "status.json",
    )
    with (run_dir / "hashes.sha256").open("x", encoding="utf-8") as handle:
        for name in hashed:
            handle.write(f"{sha256(run_dir / name)}  {name}\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare", "generate", "finalize"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    if args.command == "preflight":
        config = validate_config()
        rows = selected_original_rows(config)
        manifest = selection_manifest(rows)
        payload: Any = {
            "status": "cpu_preflight_passed",
            "selected": len(rows),
            "stopping_mechanisms": _count_by(manifest, "stopping_mechanism"),
            "exclusive_classes": _count_by(manifest, "exclusive_class"),
            "repeated_normalized_code": sum(bool(row["repeated_normalized_code"]) for row in manifest),
            "training_use_forbidden": True,
        }
    else:
        if args.run_dir is None:
            parser.error(f"{args.command} requires --run-dir")
        if args.command == "prepare":
            payload = prepare(args.run_dir)
        elif args.command == "generate":
            payload = asyncio.run(generate(args.run_dir, args.endpoint))
        else:
            payload = finalize(args.run_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
