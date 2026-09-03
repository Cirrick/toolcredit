"""CPU gates for the E3 longer-budget diagnostic."""

from __future__ import annotations

import json
from collections import Counter

import pytest

import eval.budget_diagnostic as diagnostic
from eval.budget_diagnostic import (
    diagnostic_class,
    residual_subtype,
    selected_original_rows,
    selection_manifest,
    validate_config,
)


def _row(reason: str, codes: list[str]) -> dict:
    turns = []
    for code in codes:
        turns.append(
            {
                "parsed_calls": [
                    {
                        "name": "code_interpreter",
                        "arguments": json.dumps({"code": code}),
                    }
                ]
            }
        )
    return {
        "question_id": "fixture",
        "termination_reason": reason,
        "tool_metadata": {
            "turn_credit_ledger": {
                "termination_reason": reason,
                "assistant_turns": turns,
            }
        },
    }


def test_budget_config_changes_only_pinned_budget_fields() -> None:
    config = validate_config()
    assert config["selection"]["training_use_forbidden"] is True
    assert config["generation"]["max_response_tokens_total"] == 6144
    assert config["generation"]["max_user_tool_turns"] == 6
    assert config["generation"]["max_assistant_turns"] == 7

    drifted = json.loads(json.dumps(config))
    drifted["generation"]["prompt_suffix"] += " drift"
    with pytest.raises(ValueError, match="non-budget generation semantics drifted"):
        validate_config(drifted)


def test_original_selection_is_exact_and_training_forbidden() -> None:
    manifest = selection_manifest(selected_original_rows())
    assert len(manifest) == 70
    assert len({row["question_id"] for row in manifest}) == 70
    assert all(row["training_use_forbidden"] is True for row in manifest)
    assert Counter(row["stopping_mechanism"] for row in manifest) == {
        "response_token_limit": 63,
        "assistant_or_tool_turn_limit": 7,
    }
    assert Counter(row["exclusive_class"] for row in manifest) == {
        "response_token_limit": 58,
        "assistant_or_tool_turn_limit": 2,
        "repeated_normalized_code": 10,
    }
    assert sum(row["repeated_normalized_code"] for row in manifest) == 10


def test_repeated_code_is_orthogonal_and_has_exclusive_precedence() -> None:
    repeated = diagnostic_class(_row("assistant_turn_budget", ["print(1)", "print(1)"]))
    assert repeated["stopping_mechanism"] == "assistant_or_tool_turn_limit"
    assert repeated["repeated_normalized_code"] is True
    assert repeated["exclusive_class"] == "repeated_normalized_code"
    assert repeated["mechanism_tags"] == [
        "assistant_or_tool_turn_limit",
        "repeated_normalized_code",
    ]

    token_only = diagnostic_class(_row("response_length", []))
    assert token_only["exclusive_class"] == "response_token_limit"


def test_residual_wrong_subtypes_do_not_hide_persistent_exhaustion() -> None:
    assert residual_subtype(_row("final_answer", [])) == "completed_wrong_semantic"
    assert residual_subtype(_row("response_length", [])) == "persistent_response_token_limit"
    assert residual_subtype(_row("assistant_turn_budget", [])) == "persistent_assistant_or_tool_turn_limit"


def test_correct_answer_before_later_budget_stop_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostic,
        "compute_score",
        lambda **_: {
            "acc": 1.0,
            "format_ok": 1.0,
            "invalid": 0.0,
            "truncated": 0.0,
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "verify_answer",
        lambda *_args, **_kwargs: {"correct": True},
    )
    row = _row("response_length", [])
    row.update({"response": r"\boxed{1}"})
    row["tool_metadata"]["turn_credit_ledger"]["trajectory_truncated"] = True

    scored = diagnostic._score_long_row(row, "1")

    assert scored["correct"] is True
    assert scored["truncated"] is True
    assert scored["correct_despite_truncation"] is True
