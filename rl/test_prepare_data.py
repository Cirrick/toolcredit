from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.prepare_data import PROMPT_SUFFIX, select_validation, to_verl_row, validate_split


def _row(id_: str, level: int = 3) -> dict:
    return {
        "id": id_,
        "question": f"question {id_}",
        "answer": "1",
        "level": level,
        "subject": "Algebra",
        "source": "MATH",
        "split": "train",
    }


def test_to_verl_row_uses_official_chat_and_reward_schema() -> None:
    converted = to_verl_row(_row("x"), "toolcredit_math")
    assert converted["prompt"] == [{"role": "user", "content": "question x" + PROMPT_SUFFIX}]
    assert converted["reward_model"] == {"style": "rule", "ground_truth": "1"}
    assert converted["extra_info"]["id"] == "x"


def test_validation_selection_is_deterministic_and_stratified() -> None:
    rows = [_row(f"l{level}_{i}", level) for level in range(1, 6) for i in range(4)]
    first = select_validation(rows, size=10, seed=7)
    second = select_validation(rows, size=10, seed=7)
    assert first == second
    assert {level: sum(row["level"] == level for row in first) for level in range(1, 6)} == {
        level: 2 for level in range(1, 6)
    }


def test_validate_split_rejects_heldout_leak() -> None:
    subset = [_row(str(i)) for i in range(5203)] + [_row("heldout")]
    heldout = [_row("heldout")]
    with pytest.raises(ValueError, match="leaked"):
        validate_split(subset, heldout, subset[:5202] + heldout)
