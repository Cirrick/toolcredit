"""M7 pass@k and behavior metrics with explicit numerators and denominators."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c,k)/C(n,k)."""
    if not 0 <= c <= n:
        raise ValueError(f"correct count must satisfy 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "zero_denominator": denominator == 0,
    }


def pass_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["checkpoint_role"], row["evaluation_setting"], row["source"])].append(row)
        grouped[(row["checkpoint_role"], row["evaluation_setting"], "FULL760")].append(row)
    result: dict[str, Any] = {}
    for (role, setting, source), values in sorted(grouped.items()):
        by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            by_question[str(row["question_id"])].append(row)
        key = f"{role}:{setting}:{source}"
        if setting == "greedy":
            if any(len(samples) != 1 for samples in by_question.values()):
                raise ValueError(f"greedy group does not have exactly one sample per question: {key}")
            correct = sum(bool(samples[0]["correct"]) for samples in by_question.values())
            result[key] = {"pass@1": _rate(correct, len(by_question))}
        elif setting == "sampled":
            if any(len(samples) != 4 for samples in by_question.values()):
                raise ValueError(f"sampled group does not have exactly four samples per question: {key}")
            estimates = {k: [] for k in (1, 2, 4)}
            for samples in by_question.values():
                correct = sum(bool(sample["correct"]) for sample in samples)
                for k in estimates:
                    estimates[k].append(pass_at_k(4, correct, k))
            result[key] = {
                f"pass@{k}": {
                    "sum_question_estimators": sum(values_k),
                    "denominator_questions": len(values_k),
                    "value": sum(values_k) / len(values_k) if values_k else None,
                }
                for k, values_k in estimates.items()
            }
        else:
            raise ValueError(f"unknown evaluation setting: {setting}")
    return result


def tool_behavior_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    total = len(values)
    called = sum(float(row["reward"]["n_tool_calls"]) > 0 for row in values)
    errors = sum(float(row["reward"]["tool_error_rate"]) > 0 for row in values)
    parser_errors = sum(float(row["reward"]["tool_parse_errors"]) > 0 for row in values)
    invalid = sum(bool(row["invalid"]) for row in values)
    truncated = sum(bool(row["truncated"]) for row in values)
    parser_recovered = sum(
        bool(row["correct"]) and float(row["reward"]["tool_parse_errors"]) > 0 for row in values
    )
    execution_recovered = sum(
        bool(row["correct"]) and float(row["reward"]["tool_error_rate"]) > 0 for row in values
    )
    return {
        "trajectory_count": total,
        "tool_call_rate": _rate(called, total),
        "tool_error_trajectory_rate": _rate(errors, total),
        "parser_error_trajectory_rate": _rate(parser_errors, total),
        "invalid_rate": _rate(invalid, total),
        "truncation_rate": _rate(truncated, total),
        "parser_error_recovery": _rate(parser_recovered, parser_errors),
        "execution_error_recovery": _rate(execution_recovered, errors),
    }
