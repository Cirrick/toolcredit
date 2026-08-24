from __future__ import annotations

import json
from pathlib import Path

from rl.analyze_e5 import analyze


def _turn(index: int, adopted: bool) -> dict:
    s_t = int(adopted)
    return {
        "turn_index": index,
        "A_traj": 0.25,
        "execution_success": s_t,
        "execution_status": "ok" if adopted else "no_tool",
        "adopted": s_t,
        "adoption_status": "adopted" if adopted else "no_tool",
        "adoption_evidence": {"kind": "boxed"} if adopted else None,
        "s_t": s_t,
        "s_bar_t": 0.5 if adopted else 0.0,
        "position_denominator": 2,
        "correction": 0.25 if adopted else 0.0,
        "A_turn": 0.5 if adopted else 0.25,
        "policy_token_span": [index * 3, index * 3 + 2],
        "tool_token_span": [2, 3] if index == 0 else None,
        "assistant_text": "therefore 42" if adopted else "final",
        "visible_tool_response_text": "42" if adopted else None,
        "final_score": 1.1,
    }


def test_smoke_audit_requires_ten_decisive_cases(tmp_path: Path) -> None:
    run_dir = tmp_path / "e5_turn_credit_smoke_test"
    predictions = run_dir / "predictions/train"
    predictions.mkdir(parents=True)
    proof_id = "e5-wrapper-v1:test"
    (run_dir / "wrapper_installation.json").write_text(
        json.dumps({"installed": True, "restored": True, "proof_id": proof_id}), encoding="utf-8"
    )
    rows = []
    for index in range(9):
        audit = {
            "schema_version": "toolcredit_turn_ledger_v1",
            "adoption_version": "toolcredit_adoption_v1",
            "wrapper_proof": proof_id,
            "prompt_uid": f"prompt-{index}",
            "trajectory_uid": f"trajectory-{index}",
            "response_length_unpadded": 5,
            "response_mask_sha256_uint8": "0" * 64,
            "policy_token_count": 4,
            "masked_tool_or_padding_token_count": 1,
            "trajectory_truncated": False,
            "termination_reason": "final_answer",
            "turns": [_turn(0, True), {**_turn(1, False), "policy_token_span": [3, 5]}],
        }
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "output": "tool output then therefore 42",
                "turn_credit_audit": audit,
                "turn_credit_wrapper_proof": proof_id,
            }
        )
    (predictions / "1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = analyze(run_dir)
    assert summary["selected_auto_adopted_cases"] == 9
    assert summary["automatic_80pct_gate_applicable"] is False
    assert summary["heuristic_precision_gate_passed"] is False
    assert summary["explicit_review_required"] is True

    candidates = [
        json.loads(line)
        for line in (run_dir / "analysis/e5_adoption_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    preliminary_path = run_dir / "analysis/e5_adoption_agent_preliminary_labels.jsonl"
    preliminary_path.write_text(
        "".join(
            json.dumps(
                {
                    "candidate_id": item["candidate_id"],
                    "agent_label": "true_adoption",
                    "agent_reason": "Codex-only preliminary judgment",
                }
            )
            + "\n"
            for item in candidates
        ),
        encoding="utf-8",
    )
    with_agent_labels = analyze(run_dir)
    assert with_agent_labels["agent_preliminary_labeled_cases"] == 9
    assert with_agent_labels["agent_preliminary_precision"] == 1.0
    assert with_agent_labels["agent_preliminary_counts_toward_human_gate"] is False
    assert with_agent_labels["human_labeled_cases"] == 0
    assert with_agent_labels["decisively_adjudicated_cases"] == 0
    assert with_agent_labels["heuristic_precision_gate_passed"] is False
