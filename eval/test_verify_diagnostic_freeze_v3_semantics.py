"""Regression gate for the prepared M7 v3 freeze candidate."""

from __future__ import annotations

from eval.verify_diagnostic_freeze_v3_semantics import (
    EXPECTED_CHANGED_TOP_LEVEL,
    FROZEN_SEMANTIC_FIELDS,
    verify_semantic_diff,
)


def test_v3_has_only_approved_runtime_binding_delta() -> None:
    report = verify_semantic_diff()
    assert set(report["changed_top_level_fields"]) == EXPECTED_CHANGED_TOP_LEVEL
    assert set(report["unchanged_frozen_semantic_fields"]) == FROZEN_SEMANTIC_FIELDS
    assert report["parent_v2_closure_file_count"] == 37
    assert report["panel_question_count"] == 760
    assert report["expected_key_count"] == 15200
    assert report["diagnostic_evidence"]["greedy_paired_transitions"] == "primary_diagnostic"
    assert report["diagnostic_evidence"]["sampled_matched_index_transitions"] == "secondary_exploratory"
    assert report["gpu_generation_performed"] is False
