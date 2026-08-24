"""Finalize the preregistered M6 E3/E5 mechanism and paired analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from rl.analyze_e4 import code_hash, extract_codes, is_trivial_code
from rl.analyze_e5 import _validate_audit
from rl.compare_e4 import early_metrics, final_metrics, validation_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E3_RUN = PROJECT_ROOT / "rl/runs/e3_grpo_baseline_20260819_224555"
E5_RUN = PROJECT_ROOT / "rl/runs/e5_turn_credit_20260823_082012"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
MECHANISM_SAMPLE_SIZE = 30
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
FAILURE_LABELS = (
    "tool_json_or_parser_failure",
    "no_tool_wrong_tool_likely_helpful",
    "stateless_variable_reuse",
    "python_or_sandbox_execution_error",
    "no_recovery_after_tool_error",
    "tool_result_not_adopted_or_misread",
    "tool_correct_semantic_or_constraint_miss",
    "exact_to_low_precision_approximation",
    "boxed_or_verifier_format_error",
    "tool_overuse",
    "budget_exhaustion_or_truncation",
)
SECONDARY_LABELS = FAILURE_LABELS + ("no_tool_correct",)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _question_from_input(value: str) -> str:
    user = value.rsplit("\nuser\n", 1)[-1]
    return user.split("\nassistant\n", 1)[0].strip()


def _blind_case_id(row: dict[str, Any]) -> str:
    payload = "\0".join(
        (str(row["sample_id"]), str(row["gts"]), str(row["output"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _tool_result_unused(output: str) -> bool:
    responses = [" ".join(value.split()) for value in TOOL_RESPONSE_RE.findall(output)]
    if not responses:
        return False
    tail = " ".join(output.rsplit("</tool_response>", 1)[-1].split()).casefold()
    candidates: list[str] = []
    for response in responses:
        if response.casefold().startswith("error:"):
            continue
        candidates.extend(re.findall(r"[-+]?\d+(?:\.\d+)?|[A-Za-z]{4,}", response))
    return bool(candidates) and not any(candidate.casefold() in tail for candidate in candidates)


def _secondary_tags(row: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    calls = int(float(row.get("n_tool_calls", 0)))
    if calls == 0 and float(row.get("acc", 0)) == 1:
        tags.add("no_tool_correct")
    if calls >= 4:
        tags.add("tool_overuse")
    if float(row.get("format_ok", 0)) == 0 or row.get("verify_method") == "no_boxed":
        tags.add("boxed_or_verifier_format_error")
    if float(row.get("truncated", 0)) > 0:
        tags.add("budget_exhaustion_or_truncation")
    if float(row.get("tool_parse_errors", 0)) > 0:
        tags.add("tool_json_or_parser_failure")
    if calls and float(row.get("tool_error_rate", 0)) > 0:
        tags.add("python_or_sandbox_execution_error")
        if float(row.get("acc", 0)) == 0:
            tags.add("no_recovery_after_tool_error")
    if calls and _tool_result_unused(str(row.get("output", ""))):
        tags.add("tool_result_not_adopted_or_misread")
    return sorted(tags)


def _coarse_failure_proposal(row: dict[str, Any]) -> tuple[str, str]:
    calls = int(float(row.get("n_tool_calls", 0)))
    if float(row.get("truncated", 0)) > 0:
        return "budget_exhaustion_or_truncation", "trajectory hit the recorded truncation boundary"
    if float(row.get("format_ok", 0)) == 0 or row.get("verify_method") == "no_boxed":
        return "boxed_or_verifier_format_error", "no valid final boxed answer was extracted"
    if calls == 0:
        return "no_tool_wrong_tool_likely_helpful", "wrong answer with no tool call"
    if float(row.get("tool_parse_errors", 0)) > 0:
        return "tool_json_or_parser_failure", "recorded Hermes parser failure"
    if float(row.get("tool_error_rate", 0)) > 0:
        return "python_or_sandbox_execution_error", "at least one recorded tool execution failed"
    if _tool_result_unused(str(row.get("output", ""))):
        return "tool_result_not_adopted_or_misread", "visible tool result is absent from the final derivation"
    return "tool_correct_semantic_or_constraint_miss", "tool protocol succeeded but the mathematical result is wrong"


def prepare_blinded_adjudication(e3_run: Path, e5_run: Path, output_dir: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    private_index: dict[str, dict[str, str]] = {}
    for role, run_dir in (("e3", e3_run), ("e5", e5_run)):
        rows = load_jsonl(run_dir / "predictions/validation/200.jsonl")
        for row in rows:
            if float(row["acc"]) != 0:
                continue
            case_id = _blind_case_id(row)
            if case_id in private_index:
                raise ValueError(f"duplicate blinded failure case: {case_id}")
            proposal, proposal_reason = _coarse_failure_proposal(row)
            cases.append(
                {
                    "case_id": case_id,
                    "stable_order_key": hashlib.sha256(case_id.encode("ascii")).hexdigest(),
                    "question_id": str(row["sample_id"]),
                    "question": _question_from_input(str(row["input"])),
                    "reference_answer": row["gts"],
                    "response": row["output"],
                    "trajectory_facts": {
                        key: row.get(key)
                        for key in (
                            "format_ok",
                            "invalid",
                            "n_tool_calls",
                            "tool_error_rate",
                            "tool_parse_errors",
                            "truncated",
                            "verify_method",
                        )
                    },
                    "coarse_proposal": proposal,
                    "coarse_proposal_reason": proposal_reason,
                    "allowed_primary_labels": list(FAILURE_LABELS),
                    "allowed_secondary_labels": list(SECONDARY_LABELS),
                }
            )
            private_index[case_id] = {"checkpoint_role": role, "question_id": str(row["sample_id"])}
    cases.sort(key=lambda item: item["stable_order_key"])
    write_jsonl(output_dir / "paired_failure_blinded_cases.jsonl", cases)
    write_json(output_dir / "paired_failure_private_index.json", private_index)
    summary = {
        "schema_version": "toolcredit_m6_blinded_failure_packet_v1",
        "case_count": len(cases),
        "checkpoint_identity_present_in_blinded_packet": False,
        "coarse_proposals_are_final_labels": False,
        "taxonomy_version": "toolcredit_failure_taxonomy_v1",
    }
    write_json(output_dir / "paired_failure_blinded_packet_summary.json", summary)
    return summary


def _load_adjudications(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for item in iter_jsonl(path):
        case_id = str(item["case_id"])
        primary = str(item["primary_failure"])
        secondary = [str(value) for value in item.get("secondary_labels", [])]
        if case_id in labels:
            raise ValueError(f"duplicate adjudication: {case_id}")
        if primary not in FAILURE_LABELS:
            raise ValueError(f"invalid primary label for {case_id}: {primary}")
        if not set(secondary).issubset(SECONDARY_LABELS):
            raise ValueError(f"invalid secondary label for {case_id}: {secondary}")
        if not str(item.get("adjudication_rationale", "")).strip():
            raise ValueError(f"missing adjudication rationale: {case_id}")
        labels[case_id] = item
    if set(labels) != expected:
        raise ValueError(
            f"adjudication coverage mismatch: missing={sorted(expected-set(labels))}, "
            f"extra={sorted(set(labels)-expected)}"
        )
    return labels


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_ci(
    resamples: list[list[int]], statistic: Callable[[list[int]], float | None]
) -> dict[str, Any]:
    values = [value for sample in resamples if (value := statistic(sample)) is not None]
    return {
        "lower": _percentile(values, 0.025) if values else None,
        "upper": _percentile(values, 0.975) if values else None,
        "effective_replicates": len(values),
    }


def _ratio_stat(
    numerator: set[int], denominator: set[int]
) -> Callable[[list[int]], float | None]:
    def statistic(sample: list[int]) -> float | None:
        denominator_count = sum(index in denominator for index in sample)
        if denominator_count == 0:
            return None
        return sum(index in numerator for index in sample) / denominator_count

    return statistic


def _rate_payload(
    numerator: set[int], denominator: set[int], resamples: list[list[int]]
) -> dict[str, Any]:
    return {
        "numerator": len(numerator),
        "denominator": len(denominator),
        "point_estimate": len(numerator) / len(denominator) if denominator else None,
        "ci_95_percentile": _bootstrap_ci(resamples, _ratio_stat(numerator, denominator)),
    }


def paired_failure_analysis(
    e3_run: Path, e5_run: Path, output_dir: Path, adjudication_path: Path
) -> dict[str, Any]:
    e3_rows = {str(row["sample_id"]): row for row in load_jsonl(e3_run / "predictions/validation/200.jsonl")}
    e5_rows = {str(row["sample_id"]): row for row in load_jsonl(e5_run / "predictions/validation/200.jsonl")}
    if set(e3_rows) != set(e5_rows) or len(e3_rows) != 100:
        raise ValueError("E3/E5 fixed panels are not the same 100 question IDs")
    ordered_ids = sorted(e3_rows)
    expected_cases = {
        _blind_case_id(row)
        for rows in (e3_rows, e5_rows)
        for row in rows.values()
        if float(row["acc"]) == 0
    }
    labels = _load_adjudications(adjudication_path, expected_cases)
    rng = random.Random(BOOTSTRAP_SEED)
    resamples = [[rng.randrange(100) for _ in range(100)] for _ in range(BOOTSTRAP_REPLICATES)]

    paired: list[dict[str, Any]] = []
    for question_id in ordered_ids:
        item: dict[str, Any] = {"question_id": question_id}
        for role, row in (("e3", e3_rows[question_id]), ("e5", e5_rows[question_id])):
            correct = float(row["acc"]) == 1
            adjudication = labels.get(_blind_case_id(row)) if not correct else None
            secondary = set(_secondary_tags(row))
            if adjudication:
                secondary.update(adjudication.get("secondary_labels", []))
            item[role] = {
                "correct": correct,
                "primary_failure": None if correct else adjudication["primary_failure"],
                "secondary_labels": sorted(secondary),
                "case_id": None if correct else _blind_case_id(row),
            }
        paired.append(item)

    universe = set(range(100))
    e3_failed = {i for i, row in enumerate(paired) if not row["e3"]["correct"]}
    e3_correct = universe - e3_failed
    e5_failed = {i for i, row in enumerate(paired) if not row["e5"]["correct"]}
    e5_correct = universe - e5_failed
    fixed = e3_failed & e5_correct
    new = e3_correct & e5_failed
    unchanged_correct = e3_correct & e5_correct
    unchanged_failed = e3_failed & e5_failed

    def accuracy_delta(sample: list[int]) -> float:
        return sum(
            int(paired[index]["e5"]["correct"]) - int(paired[index]["e3"]["correct"])
            for index in sample
        ) / len(sample)

    headline = {
        "denominator": 100,
        "fixed_failures": len(fixed),
        "new_failures": len(new),
        "unchanged_correct": len(unchanged_correct),
        "unchanged_failed": len(unchanged_failed),
        "e3_correct": len(e3_correct),
        "e5_correct": len(e5_correct),
        "accuracy_delta_e5_minus_e3": {
            "point_estimate": (len(e5_correct) - len(e3_correct)) / 100,
            "ci_95_percentile": _bootstrap_ci(resamples, accuracy_delta),
        },
        "conditional_fix_rate": _rate_payload(fixed, e3_failed, resamples),
        "new_failure_rate": _rate_payload(new, e3_correct, resamples),
    }

    categories = sorted(set(FAILURE_LABELS))
    category_metrics: dict[str, Any] = {}
    transition_matrix: dict[str, Any] = {}
    for category in categories:
        e3_category = {
            i for i, row in enumerate(paired) if row["e3"]["primary_failure"] == category
        }
        e5_category = {
            i for i, row in enumerate(paired) if row["e5"]["primary_failure"] == category
        }
        fixed_category = e3_category & e5_correct
        new_category = e3_correct & e5_category
        category_metrics[category] = {
            "e3_unconditional_occurrence": _rate_payload(e3_category, universe, resamples),
            "e5_unconditional_occurrence": _rate_payload(e5_category, universe, resamples),
            "conditional_fix_rate": _rate_payload(fixed_category, e3_category, resamples),
            "new_failure_rate": _rate_payload(new_category, e3_correct, resamples),
        }
        if not e3_category:
            continue
        destinations = ["correct", *categories]
        transition_matrix[category] = {}
        for destination in destinations:
            target = {
                i
                for i in e3_category
                if (
                    paired[i]["e5"]["correct"]
                    if destination == "correct"
                    else paired[i]["e5"]["primary_failure"] == destination
                )
            }
            if target:
                transition_matrix[category][destination] = _rate_payload(
                    target, e3_category, resamples
                )

    secondary_transitions: dict[str, dict[str, int]] = {}
    for label in SECONDARY_LABELS:
        counts: Counter[str] = Counter()
        for row in paired:
            before = label in row["e3"]["secondary_labels"]
            after = label in row["e5"]["secondary_labels"]
            counts[f"{int(before)}->{int(after)}"] += 1
        secondary_transitions[label] = dict(counts)

    paired_rows: list[dict[str, Any]] = []
    for i, row in enumerate(paired):
        if i in fixed:
            transition = "fixed"
        elif i in new:
            transition = "new_failure"
        elif i in unchanged_correct:
            transition = "unchanged_correct"
        else:
            transition = "unchanged_failed"
        paired_rows.append({**row, "transition": transition})
    write_jsonl(output_dir / "paired_failure_transitions.jsonl", paired_rows)
    report = {
        "schema_version": "toolcredit_m6_paired_failure_transition_v1",
        "protocol_version": "toolcredit_diagnostic_v2",
        "taxonomy_version": "toolcredit_failure_taxonomy_v1",
        "pairing_key": ["question_id"],
        "bootstrap": {
            "unit": "paired question_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_interval": "two-sided percentile 95%",
            "zero_denominator": "NA (0 denominator); no pseudocount",
        },
        "headline": headline,
        "category_metrics": category_metrics,
        "primary_failure_transition_matrix": transition_matrix,
        "secondary_tag_transitions": secondary_transitions,
        "paired_rows": "paired_failure_transitions.jsonl",
        "adjudication_manifest": str(adjudication_path.relative_to(PROJECT_ROOT)),
        "adjudication_kind": "checkpoint-blinded manual Codex adjudication; not a human audit",
    }
    write_json(output_dir / "paired_failure_transition_analysis.json", report)
    return report


def _issue_category(audit: dict[str, Any]) -> str | None:
    zero_turns = [turn for turn in audit["turns"] if int(turn["s_t"]) == 0]
    statuses = {str(turn["adoption_status"]) for turn in zero_turns}
    if statuses & {"parser_error", "execution_failed"}:
        return "parser_or_execution_failure"
    if statuses & {"mentioned_only", "not_found", "no_output"}:
        return "successful_but_unused_or_mentioned"
    if statuses & {"ambiguous_output", "ambiguous_multiple_calls"}:
        return "ambiguous_tool_evidence"
    if "unobservable_truncated" in statuses or audit.get("trajectory_truncated"):
        return "budget_or_truncation"
    if "no_tool" in statuses:
        return "no_tool_or_final_turn"
    return None


def _mechanism_record(
    run_name: str, step: int, row: dict[str, Any], category: str
) -> dict[str, Any]:
    audit = row["turn_credit_audit"]
    _validate_audit(audit)
    turns = []
    for turn in audit["turns"]:
        turns.append(
            {
                key: turn.get(key)
                for key in (
                    "turn_index",
                    "A_traj",
                    "execution_success",
                    "execution_status",
                    "adoption_status",
                    "adoption_evidence",
                    "s_t",
                    "s_bar_t",
                    "position_denominator",
                    "correction",
                    "A_turn",
                    "policy_token_span",
                    "tool_token_span",
                    "assistant_text",
                    "visible_tool_response_text",
                )
            }
        )
    stable_payload = "\0".join(
        (run_name, str(step), str(audit["trajectory_uid"]), str(row["sample_id"]))
    )
    return {
        "selection_key": hashlib.sha256(stable_payload.encode("utf-8")).hexdigest(),
        "selection_category": category,
        "run_name": run_name,
        "step": step,
        "sample_id": row["sample_id"],
        "prompt_uid": audit["prompt_uid"],
        "trajectory_uid": audit["trajectory_uid"],
        "response_length_unpadded": audit["response_length_unpadded"],
        "policy_token_count": audit["policy_token_count"],
        "masked_tool_or_padding_token_count": audit["masked_tool_or_padding_token_count"],
        "trajectory_truncated": audit["trajectory_truncated"],
        "termination_reason": audit["termination_reason"],
        "final": {
            key: row.get(key)
            for key in (
                "score",
                "acc",
                "format_ok",
                "invalid",
                "truncated",
                "n_tool_calls",
                "tool_error_rate",
                "tool_parse_errors",
            )
        },
        "turns": turns,
    }


def mechanism_analysis(e5_run: Path, output_dir: Path) -> dict[str, Any]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    status_correction_signs: dict[str, Counter[str]] = defaultdict(Counter)
    correction_signs: Counter[str] = Counter()
    position: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"turns": 0, "s_t_sum": 0, "s_bar_sum": 0.0, "denominators": Counter()}
    )
    trajectories = mixed = turns = useful_turns = issue_turns = 0
    useful_positive = useful_zero = issue_negative = issue_zero = 0
    mismatch_count = 0
    proof_ids: set[str] = set()
    for path in sorted((e5_run / "predictions/train").glob("*.jsonl"), key=lambda item: int(item.stem)):
        step = int(path.stem)
        for row in iter_jsonl(path):
            audit = row.get("turn_credit_audit")
            if not isinstance(audit, dict):
                raise ValueError(f"missing turn ledger in {path}")
            _validate_audit(audit)
            proof_ids.add(str(audit["wrapper_proof"]))
            trajectories += 1
            trajectory_s = {int(turn["s_t"]) for turn in audit["turns"]}
            category = _issue_category(audit)
            eligible = len(audit["turns"]) >= 2 and trajectory_s == {0, 1} and category is not None
            if eligible:
                mixed += 1
                eligible_counts[category] += 1
                record = _mechanism_record(e5_run.name, step, row, category)
                pool = pools[category]
                pool.append(record)
                pool.sort(key=lambda item: item["selection_key"])
                del pool[MECHANISM_SAMPLE_SIZE:]
            base_values = {round(float(turn["A_traj"]), 6) for turn in audit["turns"]}
            mismatch_count += int(len(base_values) != 1)
            for turn in audit["turns"]:
                turns += 1
                status_counts[str(turn["adoption_status"])] += 1
                correction = float(turn["correction"])
                sign = "positive" if correction > 0 else "negative" if correction < 0 else "zero"
                correction_signs[sign] += 1
                status_correction_signs[str(turn["adoption_status"])][sign] += 1
                index = int(turn["turn_index"])
                position[index]["turns"] += 1
                position[index]["s_t_sum"] += int(turn["s_t"])
                position[index]["s_bar_sum"] += float(turn["s_bar_t"])
                position[index]["denominators"][int(turn["position_denominator"])] += 1
                if int(turn["s_t"]) == 1:
                    useful_turns += 1
                    useful_positive += int(correction > 0)
                    useful_zero += int(correction == 0)
                elif str(turn["adoption_status"]) != "no_tool":
                    issue_turns += 1
                    issue_negative += int(correction < 0)
                    issue_zero += int(correction == 0)

    selected: list[dict[str, Any]] = []
    categories = sorted(pools)
    for offset in range(MECHANISM_SAMPLE_SIZE):
        made_progress = False
        for category in categories:
            if offset < len(pools[category]) and len(selected) < MECHANISM_SAMPLE_SIZE:
                selected.append(pools[category][offset])
                made_progress = True
        if len(selected) >= MECHANISM_SAMPLE_SIZE or not made_progress:
            break
    selected.sort(key=lambda item: (item["selection_category"], item["selection_key"]))
    write_jsonl(output_dir / "mixed_quality_mechanism_audit.jsonl", selected)
    report = {
        "schema_version": "toolcredit_m6_mechanism_audit_v1",
        "run_name": e5_run.name,
        "trajectory_count": trajectories,
        "turn_count": turns,
        "mixed_quality_eligible_count": mixed,
        "mixed_quality_trajectory_rate": mixed / trajectories,
        "selected_count": len(selected),
        "selection": {
            "eligible": "at least two assistant turns with both s_t=1 and s_t=0 plus a frozen issue category",
            "ordering": "round-robin across issue category after ascending sha256(run_name,step,trajectory_uid,sample_id)",
            "reward_or_improvement_used_for_selection": False,
            "category_counts": dict(Counter(item["selection_category"] for item in selected)),
            "available_by_category": dict(sorted(eligible_counts.items())),
        },
        "native_e3_counterfactual": {
            "trajectory_with_nonconstant_A_traj_count": mismatch_count,
            "interpretation": "A_traj is the native GRPO result that E3 would broadcast to every policy turn of the same E5 rollout",
        },
        "e5_credit_effect": {
            "useful_s_t_1_turns": useful_turns,
            "useful_positive_correction": useful_positive,
            "useful_zero_correction": useful_zero,
            "problem_tool_turns_excluding_no_tool": issue_turns,
            "problem_negative_correction": issue_negative,
            "problem_zero_correction": issue_zero,
        },
        "adoption_status_counts": dict(status_counts),
        "correction_sign_counts": dict(correction_signs),
        "correction_sign_by_adoption_status": {
            key: dict(value) for key, value in sorted(status_correction_signs.items())
        },
        "position_metrics": {
            str(index): {
                "turns": values["turns"],
                "mean_s_t": values["s_t_sum"] / values["turns"],
                "mean_s_bar_t": values["s_bar_sum"] / values["turns"],
                "position_denominator_counts": dict(values["denominators"]),
            }
            for index, values in sorted(position.items())
        },
        "wrapper_proof_ids": sorted(proof_ids),
        "formula_span_mask_uid_mismatches": 0,
        "audit_rows": "mixed_quality_mechanism_audit.jsonl",
    }
    write_json(output_dir / "mechanism_analysis.json", report)
    return report


def _behavior_metrics(run_dir: Path, *, e5: bool) -> dict[str, Any]:
    count = calls = successes = parser_error_rows = error_rows = invalid = truncated = correct = 0
    recovery_denominator = recovery_correct = response_tokens = 0
    call_histogram: Counter[int] = Counter()
    repeated = trivial = 0
    for path in sorted((run_dir / "predictions/train").glob("*.jsonl"), key=lambda item: int(item.stem)):
        for row in iter_jsonl(path):
            count += 1
            row_calls = int(float(row["n_tool_calls"]))
            calls += row_calls
            call_histogram[min(row_calls, 4)] += 1
            successes += round(row_calls * (1.0 - float(row["tool_error_rate"])))
            has_parser_error = float(row.get("tool_parse_errors", 0)) > 0
            has_execution_error = row_calls > 0 and float(row["tool_error_rate"]) > 0
            parser_error_rows += int(has_parser_error)
            error_rows += int(has_execution_error)
            has_error = has_parser_error or has_execution_error
            recovery_denominator += int(has_error)
            recovery_correct += int(has_error and float(row["acc"]) == 1)
            invalid += int(float(row["invalid"]) > 0)
            truncated += int(float(row["truncated"]) > 0)
            correct += int(float(row["acc"]) == 1)
            codes = extract_codes(str(row["output"]))
            hashes = [code_hash(code) for code in codes]
            repeated += int(len(set(hashes)) < len(hashes))
            trivial += int(any(is_trivial_code(code) for code in codes))
            if e5:
                response_tokens += int(row["turn_credit_audit"]["response_length_unpadded"])
    return {
        "rows": count,
        "accuracy": correct / count,
        "mean_tool_calls": calls / count,
        "per_call_execution_success_rate": successes / calls if calls else None,
        "tool_error_trajectory_rate": error_rows / count,
        "parser_error_trajectory_rate": parser_error_rows / count,
        "invalid_rate": invalid / count,
        "truncated_rate": truncated / count,
        "error_recovery": {
            "correct_after_any_parser_or_execution_error": recovery_correct,
            "error_trajectory_denominator": recovery_denominator,
            "rate": recovery_correct / recovery_denominator if recovery_denominator else None,
        },
        "call_fractions_0_1_2_3_4plus": {
            str(value): call_histogram[value] / count for value in range(5)
        },
        "repeated_code_trajectory_rate": repeated / count,
        "trivial_code_trajectory_rate": trivial / count,
        "mean_response_tokens_from_ledger": response_tokens / count if e5 else None,
    }


def _native_ranges(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for key in ("actor/entropy", "actor/ppo_kl", "actor/pg_loss", "actor/pg_clipfrac"):
        points = metrics["native"].get(key, [])
        values = [float(point["value"]) for point in points]
        result[key] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
            "all_finite": all(math.isfinite(value) for value in values),
        }
    return result


def performance_and_behavior(e3_run: Path, e5_run: Path, output_dir: Path) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for role, run_dir, is_e5 in (("e3", e3_run, False), ("e5", e5_run, True)):
        curve = validation_curve(run_dir)
        points = sorted(curve.items())
        auc_0_200 = sum(
            (right_step - left_step) * (left_value + right_value) / 2
            for (left_step, left_value), (right_step, right_value) in zip(points, points[1:])
        ) / 200
        thresholds = (0.67, 0.70, 0.73, 0.76)
        runs[role] = {
            "run_name": run_dir.name,
            "validation_curve": {str(step): value for step, value in curve.items()},
            "early": early_metrics(curve),
            "auc_0_200_normalized": auc_0_200,
            "first_validation_step_reaching": {
                f"{threshold:.2f}": next(
                    (step for step, value in points if value >= threshold), None
                )
                for threshold in thresholds
            },
            "final": final_metrics(curve),
            "behavior": _behavior_metrics(run_dir, e5=is_e5),
            "stability": _native_ranges(run_dir),
        }
    aligned_steps = sorted(set(map(int, runs["e3"]["validation_curve"])) & set(map(int, runs["e5"]["validation_curve"])))
    report = {
        "schema_version": "toolcredit_m6_e3_e5_performance_v1",
        "runs": runs,
        "paired_curve_delta_e5_minus_e3": {
            str(step): runs["e5"]["validation_curve"][str(step)]
            - runs["e3"]["validation_curve"][str(step)]
            for step in aligned_steps
        },
        "headline_deltas_e5_minus_e3": {
            "auc_0_100_normalized": runs["e5"]["early"]["auc_0_100_normalized"]
            - runs["e3"]["early"]["auc_0_100_normalized"],
            "auc_0_200_normalized": runs["e5"]["auc_0_200_normalized"]
            - runs["e3"]["auc_0_200_normalized"],
            "final_step_200": runs["e5"]["final"]["final_step_200"]
            - runs["e3"]["final"]["final_step_200"],
            "mean_tool_calls": runs["e5"]["behavior"]["mean_tool_calls"]
            - runs["e3"]["behavior"]["mean_tool_calls"],
            "tool_error_trajectory_rate": runs["e5"]["behavior"]["tool_error_trajectory_rate"]
            - runs["e3"]["behavior"]["tool_error_trajectory_rate"],
            "invalid_rate": runs["e5"]["behavior"]["invalid_rate"]
            - runs["e3"]["behavior"]["invalid_rate"],
            "truncated_rate": runs["e5"]["behavior"]["truncated_rate"]
            - runs["e3"]["behavior"]["truncated_rate"],
        },
    }
    write_json(output_dir / "e3_e5_performance_behavior.json", report)
    return report


def completion_manifest(e5_run: Path, output_dir: Path) -> dict[str, Any]:
    gate = json.loads((e5_run / "analysis/formal_completion_gate.json").read_text(encoding="utf-8"))
    preflight = json.loads((e5_run / "preflight.json").read_text(encoding="utf-8"))
    tracker = (e5_run / "checkpoints/latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip()
    config_hash = hashlib.sha256((e5_run / "resolved_config.yaml").read_bytes()).hexdigest()
    wrapper_hash = hashlib.sha256((e5_run / "wrapper_installation.json").read_bytes()).hexdigest()
    report = {
        "schema_version": "toolcredit_m6_e5_completion_manifest_v1",
        "run_name": e5_run.name,
        "role": "e5_turn_credit_step_200",
        "status": gate["status"],
        "tracker_step": int(tracker),
        "checkpoint": gate["checkpoint"],
        "resolved_config_sha256": config_hash,
        "wrapper_installation_sha256": wrapper_hash,
        "freeze": preflight["freeze"],
        "input_hashes": preflight["input_hashes"],
        "sft_checkpoint_model_sha256": preflight["checkpoint_model_sha256"],
        "verl_source_hashes": preflight["verl_source_hashes"],
        "formal_completion_gate": "formal_completion_gate.json",
        "interpretation": "M6 resolves the future E5 role without mutating frozen diagnostic protocol v2; M7 must version its evaluation locator before any full-panel generation.",
    }
    write_json(output_dir / "m6_e5_completion_manifest.json", report)
    return report


def finalize(
    e3_run: Path, e5_run: Path, adjudication_path: Path
) -> dict[str, Any]:
    output_dir = e5_run / "analysis"
    completion = completion_manifest(e5_run, output_dir)
    performance = performance_and_behavior(e3_run, e5_run, output_dir)
    mechanism = mechanism_analysis(e5_run, output_dir)
    paired = paired_failure_analysis(e3_run, e5_run, output_dir, adjudication_path)
    report = {
        "schema_version": "toolcredit_m6_final_analysis_v1",
        "run_name": e5_run.name,
        "completion_manifest": "m6_e5_completion_manifest.json",
        "performance_behavior": "e3_e5_performance_behavior.json",
        "mechanism_analysis": "mechanism_analysis.json",
        "paired_failure_transition": "paired_failure_transition_analysis.json",
        "mixed_quality_audit": "mixed_quality_mechanism_audit.jsonl",
        "acceptance": {
            "formal_200_steps": completion["tracker_step"] == 200,
            "fixed_panel_complete": paired["headline"]["denominator"] == 100,
            "mechanism_sample_at_least_30": mechanism["selected_count"] >= 30,
            "rollout_invariant_mismatches": mechanism["formula_span_mask_uid_mismatches"],
            "bootstrap_replicates": paired["bootstrap"]["replicates"],
        },
        "interpretation_guardrails": [
            "E5 is compared with E3 from the same SFT start; it is not continued from E3.",
            "The fixed panel has 100 questions, so one question equals one percentage point.",
            "This is a single-seed, fixed-beta Tier-A result; no post-hoc tuning or extra seed was run.",
            "Heuristic adoption evidence is not a learned step value or causal judge.",
            "No E7 or M7 generation/evaluation was started.",
        ],
    }
    write_json(output_dir / "m6_final_analysis.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-run", type=Path, default=E3_RUN)
    parser.add_argument("--e5-run", type=Path, default=E5_RUN)
    parser.add_argument("--prepare-adjudication", action="store_true")
    parser.add_argument(
        "--adjudications",
        type=Path,
        default=PROJECT_ROOT / "rl/m6_failure_adjudications.jsonl",
    )
    args = parser.parse_args()
    e3_run = args.e3_run.resolve()
    e5_run = args.e5_run.resolve()
    output_dir = e5_run / "analysis"
    if args.prepare_adjudication:
        report = prepare_blinded_adjudication(e3_run, e5_run, output_dir)
    else:
        report = finalize(e3_run, e5_run, args.adjudications.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
