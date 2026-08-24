from __future__ import annotations

import json
import random
from pathlib import Path

from rl.finalize_m6 import (
    BOOTSTRAP_REPLICATES,
    _blind_case_id,
    _issue_category,
    _rate_payload,
    prepare_blinded_adjudication,
)


def _validation_row(sample_id: str, *, correct: bool, output: str) -> dict:
    return {
        "sample_id": sample_id,
        "input": "system\n...\nuser\nquestion\nassistant\n",
        "output": output,
        "gts": "1",
        "acc": float(correct),
        "format_ok": float(correct),
        "invalid": float(not correct),
        "n_tool_calls": 0.0,
        "tool_error_rate": 0.0,
        "tool_parse_errors": 0.0,
        "truncated": 0.0,
        "verify_method": "string" if correct else "none",
    }


def test_blind_case_id_is_role_independent() -> None:
    row = _validation_row("q1", correct=False, output="wrong")
    assert _blind_case_id(row) == _blind_case_id(dict(row))


def test_prepared_failure_packet_contains_no_checkpoint_identity(tmp_path: Path) -> None:
    e3 = tmp_path / "e3"
    e5 = tmp_path / "e5"
    for run_dir, suffix in ((e3, "a"), (e5, "b")):
        validation = run_dir / "predictions/validation"
        validation.mkdir(parents=True)
        rows = [
            _validation_row("q1", correct=False, output=f"wrong-{suffix}"),
            _validation_row("q2", correct=True, output="right"),
        ]
        (validation / "200.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    summary = prepare_blinded_adjudication(e3, e5, tmp_path / "analysis")
    packet = (tmp_path / "analysis/paired_failure_blinded_cases.jsonl").read_text(
        encoding="utf-8"
    )
    assert summary["case_count"] == 2
    assert "checkpoint_role" not in packet
    assert '"e3"' not in packet and '"e5"' not in packet


def test_rate_payload_reports_exact_counts_and_all_bootstrap_replicates() -> None:
    rng = random.Random(42)
    resamples = [[rng.randrange(4) for _ in range(4)] for _ in range(BOOTSTRAP_REPLICATES)]
    payload = _rate_payload({0}, {0, 1}, resamples)
    assert payload["numerator"] == 1
    assert payload["denominator"] == 2
    assert payload["point_estimate"] == 0.5
    assert 0 < payload["ci_95_percentile"]["effective_replicates"] <= BOOTSTRAP_REPLICATES


def test_issue_category_uses_frozen_priority() -> None:
    audit = {
        "trajectory_truncated": True,
        "turns": [
            {"s_t": 1, "adoption_status": "adopted"},
            {"s_t": 0, "adoption_status": "execution_failed"},
            {"s_t": 0, "adoption_status": "unobservable_truncated"},
        ],
    }
    assert _issue_category(audit) == "parser_or_execution_failure"
