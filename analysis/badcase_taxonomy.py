"""Frozen-v1 deterministic coarse proposals and checkpoint-blinded audit packets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "eval/diagnostic_taxonomy_v1.json"


def _taxonomy() -> dict[str, Any]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def _ledger(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("tool_metadata", {}).get("turn_credit_ledger", {})
    return value if isinstance(value, dict) else {}


def propose_primary(row: dict[str, Any], gold: str) -> dict[str, Any]:
    """Apply the frozen decision-tree order; proposals are not human semantic truth."""
    labels = set(_taxonomy()["labels"])
    secondary: set[str] = set()
    calls = int(float(row["reward"]["n_tool_calls"]))
    parser_errors = int(float(row["reward"]["tool_parse_errors"]))
    execution_error = float(row["reward"]["tool_error_rate"]) > 0
    response = str(row["response"])
    ledger = _ledger(row)
    turns = ledger.get("assistant_turns", []) if isinstance(ledger.get("assistant_turns", []), list) else []

    if calls == 0:
        secondary.add("no_tool_correct" if row["correct"] else "no_tool_wrong_tool_likely_helpful")
    if calls >= 3:
        secondary.add("tool_overuse")
    if row["correct"]:
        primary = None
        rationale = "strict verifier correct; frozen taxonomy requires a null primary"
    elif row.get("budget_exhausted") or row.get("truncated"):
        primary = "budget_exhaustion_or_truncation"
        rationale = "frozen rule 1: budget exhaustion/truncation directly prevented completion"
    else:
        from rewards.verifier import verify_answer

        lenient = verify_answer(response, gold, strict_boxed=False)
        if not row.get("format_ok") and lenient["correct"]:
            primary = "boxed_or_verifier_format_error"
            rationale = "frozen rule 2: lenient verifier is correct but strict boxed format failed"
        elif calls == 0:
            primary = "no_tool_wrong_tool_likely_helpful"
            rationale = "frozen rule 3: wrong no-tool trajectory"
        elif parser_errors:
            primary = "tool_json_or_parser_failure"
            rationale = "frozen rule 4: recorded parser failure"
        elif any(
            re.search(r"(?:NameError|not defined|referenced before assignment)", str(turn.get("visible_tool_response_text", "")), re.I)
            for turn in turns
        ):
            primary = "stateless_variable_reuse"
            rationale = "frozen rule 5: tool output records stateless variable reuse"
        elif execution_error:
            later_success = any(
                int(turn.get("execution_success", 0)) == 1
                for turn in turns[1:]
            )
            if not later_success:
                primary = "no_recovery_after_tool_error"
                rationale = "frozen rule 6: execution error had no later successful recovery"
                secondary.add("python_or_sandbox_execution_error")
            else:
                primary = "python_or_sandbox_execution_error"
                rationale = "frozen rule 5: Python/sandbox execution failed before recovery"
        elif any(
            int(turn.get("execution_success", 0)) == 1
            and str(turn.get("adoption_status")) not in {"adopted", "no_tool"}
            for turn in turns
        ):
            primary = "tool_result_not_adopted_or_misread"
            rationale = "frozen rule 7: a visible successful result was not audibly adopted"
        elif re.search(r"\\boxed\{[^{}]*(?:\.\d{3,}|\\approx)[^{}]*\}", response):
            primary = "exact_to_low_precision_approximation"
            rationale = "frozen rule 9: boxed answer contains a potentially harmful approximation"
        else:
            primary = "tool_correct_semantic_or_constraint_miss"
            rationale = "frozen rule 8 fallback: tooling completed but final semantics/constraints are wrong"
    if primary is not None and primary not in labels:
        raise AssertionError(f"proposal escaped frozen taxonomy: {primary}")
    if not secondary.issubset(labels):
        raise AssertionError("secondary proposal escaped frozen taxonomy")
    return {
        "primary_failure": primary,
        "secondary_labels": sorted(secondary - ({primary} if primary else set())),
        "label_source": "deterministic_coarse_toolcredit_failure_taxonomy_v1",
        "adjudication_rationale": rationale,
    }


def enrich_with_coarse_proposal(row: dict[str, Any], gold: str) -> dict[str, Any]:
    result = dict(row)
    result.update(propose_primary(row, gold))
    if bool(result["correct"]) != (result["primary_failure"] is None):
        raise ValueError("taxonomy invariant violated: correct iff primary is null")
    return result


def _stable_rank(row: dict[str, Any]) -> str:
    identity = "\0".join(
        str(row[key]) for key in ("checkpoint_role", "evaluation_setting", "source", "question_id", "sample_index")
    )
    return hashlib.sha256(("toolcredit-m7-blind-audit-v1\0" + identity).encode("utf-8")).hexdigest()


def build_blind_packet(
    rows: Iterable[dict[str, Any]],
    gold_by_question: dict[str, str],
    limit_per_role_setting: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select stable, source/label-balanced wrong cases and separate their private identity."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["correct"]:
            groups[(row["checkpoint_role"], row["evaluation_setting"])].append(row)
    packet: list[dict[str, Any]] = []
    private: dict[str, dict[str, Any]] = {}
    forbidden = {"checkpoint_role", "stage", "transition_direction", "sample_index"}
    for group_rows in groups.values():
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            strata[(row["source"], row["primary_failure"])].append(row)
        ordered: list[dict[str, Any]] = []
        for values in strata.values():
            values.sort(key=_stable_rank)
        while len(ordered) < min(limit_per_role_setting, len(group_rows)):
            progressed = False
            for key in sorted(strata):
                if strata[key]:
                    ordered.append(strata[key].pop(0))
                    progressed = True
                    if len(ordered) == min(limit_per_role_setting, len(group_rows)):
                        break
            if not progressed:
                break
        for row in ordered:
            audit_id = "m7_audit_" + _stable_rank(row)[:20]
            public = {
                "audit_id": audit_id,
                "question_id": row["question_id"],
                "source": row["source"],
                "question": row["prompt"],
                "gold": gold_by_question[row["question_id"]],
                "messages": row["messages"],
                "response": row["response"],
                "strict_verifier_facts": row["verifier_outcome"],
                "coarse_primary_proposal": row["primary_failure"],
                "coarse_rationale": row["adjudication_rationale"],
            }
            if forbidden & public.keys():
                raise AssertionError("blind packet leaked checkpoint/stage/private sample identity")
            packet.append(public)
            private[audit_id] = {
                "checkpoint_role": row["checkpoint_role"],
                "evaluation_setting": row["evaluation_setting"],
                "question_id": row["question_id"],
                "sample_index": row["sample_index"],
                "sample_seed": row["sample_seed"],
            }
    if len(packet) > 192 or len(packet) != len(private):
        raise AssertionError("blind audit packet size/identity invariant failed")
    return packet, private
