"""Paired M7 transitions with greedy-primary and sampled-exploratory guardrails."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

import numpy as np

SAMPLED_GUARDRAIL = (
    "Matched sample indices share the frozen request seed but are not counterfactual trajectory identities; "
    "sampled transitions are secondary/exploratory and do not support strong causal claims."
)


def _pair_key(row: dict[str, Any], setting: str) -> tuple[Any, ...]:
    if row.get("evaluation_setting") != setting:
        raise ValueError("row setting differs from transition setting")
    if setting == "greedy":
        return (row["question_id"],)
    if setting == "sampled":
        return (row["question_id"], int(row["sample_index"]))
    raise ValueError(f"unknown transition setting: {setting}")


def _state(row: dict[str, Any]) -> str:
    if row["correct"]:
        if row.get("primary_failure") is not None:
            raise ValueError("correct transition row has a primary failure")
        return "correct"
    primary = row.get("primary_failure")
    if not isinstance(primary, str) or not primary:
        raise ValueError("wrong transition row lacks a primary failure")
    return primary


def paired_transition(
    before_rows: Iterable[dict[str, Any]],
    after_rows: Iterable[dict[str, Any]],
    setting: str,
    before_role: str,
    after_role: str,
) -> dict[str, Any]:
    before_values = list(before_rows)
    after_values = list(after_rows)
    before = {_pair_key(row, setting): row for row in before_values}
    after = {_pair_key(row, setting): row for row in after_values}
    if len(before) != len(before_values):
        raise ValueError("duplicate key in before rows")
    if len(after) != len(after_values):
        raise ValueError("duplicate key in after rows")
    if set(before) != set(after):
        raise ValueError("paired transition exact keys do not match")
    cells = Counter()
    matrix: Counter[tuple[str, str]] = Counter()
    secondary: Counter[tuple[str, str]] = Counter()
    exact_rows: list[dict[str, Any]] = []
    for key in sorted(before):
        left, right = before[key], after[key]
        if left["source"] != right["source"] or left.get("sample_seed") != right.get("sample_seed"):
            raise ValueError(f"paired source/seed mismatch at {key}")
        left_correct, right_correct = bool(left["correct"]), bool(right["correct"])
        cell = {
            (False, True): "fixed_failures",
            (True, False): "new_failures",
            (True, True): "unchanged_correct",
            (False, False): "unchanged_failed",
        }[(left_correct, right_correct)]
        cells[cell] += 1
        matrix[(_state(left), _state(right))] += 1
        left_secondary = set(left.get("secondary_labels", [])) or {"none"}
        right_secondary = set(right.get("secondary_labels", [])) or {"none"}
        for source_tag in left_secondary:
            for target_tag in right_secondary:
                secondary[(source_tag, target_tag)] += 1
        exact_rows.append(
            {
                "key": list(key),
                "source": left["source"],
                "sample_seed": left["sample_seed"],
                "cell": cell,
                "before_state": _state(left),
                "after_state": _state(right),
                "before_evidence": left.get("raw_evidence_pointer"),
                "after_evidence": right.get("raw_evidence_pointer"),
            }
        )
    denominator = len(before)
    if sum(cells.values()) != denominator or sum(matrix.values()) != denominator:
        raise AssertionError("transition four-cell/matrix conservation failed")
    before_correct = cells["new_failures"] + cells["unchanged_correct"]
    before_failed = cells["fixed_failures"] + cells["unchanged_failed"]
    evidence_priority = "primary_diagnostic" if setting == "greedy" else "secondary_exploratory"
    return {
        "before_role": before_role,
        "after_role": after_role,
        "setting": setting,
        "diagnostic_evidence_priority": evidence_priority,
        "causal_claim_allowed": False,
        "interpretation_guardrail": None if setting == "greedy" else SAMPLED_GUARDRAIL,
        "denominator": denominator,
        **{name: cells[name] for name in ("fixed_failures", "new_failures", "unchanged_correct", "unchanged_failed")},
        "accuracy_delta": (cells["fixed_failures"] - cells["new_failures"]) / denominator if denominator else None,
        "conditional_fix_rate": cells["fixed_failures"] / before_failed if before_failed else None,
        "new_failure_rate": cells["new_failures"] / before_correct if before_correct else None,
        "primary_failure_transition_matrix": {
            source: dict(sorted(targets.items()))
            for source, targets in _nested_counter(matrix).items()
        },
        "secondary_tag_transition": {
            source: dict(sorted(targets.items()))
            for source, targets in _nested_counter(secondary).items()
        },
        "exact_rows": exact_rows,
    }


def _nested_counter(counter: Counter[tuple[str, str]]) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for (source, target), count in counter.items():
        result[source][target] = count
    return dict(result)


def _percentile(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def source_stratified_question_bootstrap(
    paired_rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Resample questions within source and retain all matched sample indices per draw."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    clusters: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in paired_rows:
        clusters[str(row["source"])][str(row["key"][0])].append(row)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled: list[dict[str, Any]] = []
        for source in sorted(clusters):
            question_ids = sorted(clusters[source])
            selected = rng.choice(question_ids, size=len(question_ids), replace=True)
            for question_id in selected:
                sampled.extend(clusters[source][str(question_id)])
        value = statistic(sampled)
        if value is not None:
            values.append(float(value))
    point = statistic(paired_rows)
    if point is None or not values:
        return {
            "point_estimate": None,
            "lower": None,
            "upper": None,
            "effective_replicates": 0,
            "zero_denominator": True,
        }
    lower, upper = _percentile(values)
    return {
        "point_estimate": float(point),
        "lower": lower,
        "upper": upper,
        "effective_replicates": len(values),
        "replicates": replicates,
        "seed": seed,
        "confidence_interval": "two-sided percentile 95%",
        "zero_denominator": False,
    }


def accuracy_delta_statistic(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    fixed = sum(row["cell"] == "fixed_failures" for row in rows)
    new = sum(row["cell"] == "new_failures" for row in rows)
    return (fixed - new) / len(rows)


def conditional_fix_statistic(rows: list[dict[str, Any]]) -> float | None:
    failed = [row for row in rows if row["cell"] in {"fixed_failures", "unchanged_failed"}]
    return sum(row["cell"] == "fixed_failures" for row in failed) / len(failed) if failed else None


def new_failure_statistic(rows: list[dict[str, Any]]) -> float | None:
    correct = [row for row in rows if row["cell"] in {"new_failures", "unchanged_correct"}]
    return sum(row["cell"] == "new_failures" for row in correct) / len(correct) if correct else None
