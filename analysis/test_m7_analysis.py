"""Taxonomy blindness, transition conservation, priority, and clustered bootstrap fixtures."""

from __future__ import annotations

import pytest

from analysis.badcase_taxonomy import build_blind_packet, propose_primary
from analysis.paired_transitions import (
    SAMPLED_GUARDRAIL,
    accuracy_delta_statistic,
    paired_transition,
    source_stratified_question_bootstrap,
)


def _row(role: str, setting: str, qid: str, index: int, correct: bool, primary: str | None, source: str = "MATH500") -> dict:
    return {
        "checkpoint_role": role,
        "evaluation_setting": setting,
        "question_id": qid,
        "sample_index": index,
        "sample_seed": 100 + index,
        "source": source,
        "correct": correct,
        "primary_failure": primary,
        "secondary_labels": [],
        "raw_evidence_pointer": {"id": f"{role}-{qid}-{index}"},
        "prompt": f"question {qid}",
        "messages": [],
        "response": "wrong",
        "verifier_outcome": {"correct": correct},
        "adjudication_rationale": "fixture",
    }


def test_greedy_transition_is_primary_and_conserves_four_cells() -> None:
    before = [
        _row("e3", "greedy", "q1", 0, False, "tool_json_or_parser_failure"),
        _row("e3", "greedy", "q2", 0, True, None),
        _row("e3", "greedy", "q3", 0, True, None),
        _row("e3", "greedy", "q4", 0, False, "tool_overuse"),
    ]
    after = [
        _row("e5", "greedy", "q1", 0, True, None),
        _row("e5", "greedy", "q2", 0, False, "boxed_or_verifier_format_error"),
        _row("e5", "greedy", "q3", 0, True, None),
        _row("e5", "greedy", "q4", 0, False, "tool_correct_semantic_or_constraint_miss"),
    ]
    result = paired_transition(before, after, "greedy", "e3", "e5")
    assert result["diagnostic_evidence_priority"] == "primary_diagnostic"
    assert result["causal_claim_allowed"] is False
    assert result["denominator"] == 4
    assert [result[key] for key in ("fixed_failures", "new_failures", "unchanged_correct", "unchanged_failed")] == [1, 1, 1, 1]
    assert sum(sum(row.values()) for row in result["primary_failure_transition_matrix"].values()) == 4


def test_sampled_transition_is_exploratory_and_clustered() -> None:
    before = [_row("e3", "sampled", qid, i, i % 2 == 0, None if i % 2 == 0 else "tool_overuse") for qid in ("q1", "q2") for i in range(4)]
    after = [_row("e5", "sampled", qid, i, True, None) for qid in ("q1", "q2") for i in range(4)]
    result = paired_transition(before, after, "sampled", "e3", "e5")
    assert result["diagnostic_evidence_priority"] == "secondary_exploratory"
    assert result["interpretation_guardrail"] == SAMPLED_GUARDRAIL
    report = source_stratified_question_bootstrap(result["exact_rows"], accuracy_delta_statistic, replicates=50, seed=42)
    assert report["effective_replicates"] == 50
    assert report["point_estimate"] == 0.5


def test_zero_denominator_bootstrap_is_na() -> None:
    report = source_stratified_question_bootstrap([], accuracy_delta_statistic, replicates=5)
    assert report["point_estimate"] is None
    assert report["effective_replicates"] == 0
    assert report["zero_denominator"] is True


def test_taxonomy_invariants_and_blind_packet() -> None:
    row = _row("raw_base_model", "greedy", "q1", 0, False, None)
    row.update(
        {
            "reward": {"n_tool_calls": 0.0, "tool_parse_errors": 0.0, "tool_error_rate": 0.0},
            "format_ok": False,
            "truncated": False,
            "budget_exhausted": False,
            "tool_metadata": {},
        }
    )
    proposal = propose_primary(row, gold="4")
    assert proposal["primary_failure"] == "no_tool_wrong_tool_likely_helpful"
    row.update(proposal)
    packet, private = build_blind_packet([row], {"q1": "4"})
    assert len(packet) == len(private) == 1
    assert not ({"checkpoint_role", "stage", "transition_direction", "sample_index"} & packet[0].keys())
    assert private[packet[0]["audit_id"]]["checkpoint_role"] == "raw_base_model"
