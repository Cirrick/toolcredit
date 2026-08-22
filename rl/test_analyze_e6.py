from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.analyze_e6 import analyze, candidate_reasons


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": "sample-1",
        "n_tool_calls": 1.0,
        "output": "<tool_call>{}</tool_call><tool_response>42</tool_response>assistant 42",
        "original_policy_token_count": 10,
        "original_tool_return_token_count": 2,
        "nomask_loss_token_count": 12,
    }
    row.update(overrides)
    return row


def test_candidate_reasons_flags_forged_wrapper_without_call() -> None:
    reasons = candidate_reasons(_row(n_tool_calls=0, output="fake <tool_response>42</tool_response>"))
    assert "environment_wrapper_without_real_call" in reasons


def test_analyze_writes_raw_candidate_and_mask_ledger(tmp_path: Path) -> None:
    run_dir = tmp_path / "e6"
    train_dir = run_dir / "predictions/train"
    train_dir.mkdir(parents=True)
    rows = [
        _row(n_tool_calls=0, output="fake <tool_response>42</tool_response>"),
        _row(sample_id="sample-2"),
    ]
    (train_dir / "1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = analyze(run_dir)
    assert summary["rows"] == 2
    assert summary["rows_with_tool_returns"] == 2
    assert summary["tool_return_loss_fraction"] == pytest.approx(1 / 6)
    candidates = (run_dir / "analysis/e6_failure_candidates.jsonl").read_text(encoding="utf-8")
    candidate = json.loads(candidates.splitlines()[0])
    assert len(candidate["candidate_id"]) == 20
    assert '"trajectory"' in candidates
    assert '"environment_wrapper_without_real_call"' in candidates

    (run_dir / "analysis/e6_human_labels.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": candidate["candidate_id"],
                "human_label": "forge",
                "human_reason": "model emitted an environment wrapper without a parsed call",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labeled = analyze(run_dir)
    assert labeled["human_audited_count"] == 1
    assert labeled["human_label_counts"] == {"forge": 1}
    assert '"human_label": "forge"' in (
        run_dir / "analysis/e6_human_audit.jsonl"
    ).read_text(encoding="utf-8")


def test_analyze_rejects_broken_mask_ledger(tmp_path: Path) -> None:
    train_dir = tmp_path / "predictions/train"
    train_dir.mkdir(parents=True)
    (train_dir / "1.jsonl").write_text(
        json.dumps(_row(nomask_loss_token_count=11)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="do not add up"):
        analyze(tmp_path)
