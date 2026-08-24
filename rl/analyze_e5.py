"""Validate E5 turn-credit rollout evidence and stage the frozen heuristic audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

AUTO_ADOPTED_AUDIT_LIMIT = 50
HUMAN_LABELS = ("true_adoption", "false_adoption", "indecisive")


def _iter_rows(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        for line in path.open(encoding="utf-8"):
            if line.strip():
                yield path, json.loads(line)


def _candidate_id(run_name: str, step: int, audit: dict[str, Any], turn: dict[str, Any]) -> str:
    payload = "\0".join(
        (
            run_name,
            str(step),
            str(audit["prompt_uid"]),
            str(audit["trajectory_uid"]),
            str(turn["turn_index"]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _validate_audit(audit: dict[str, Any]) -> dict[str, int]:
    required = {
        "schema_version",
        "adoption_version",
        "wrapper_proof",
        "prompt_uid",
        "trajectory_uid",
        "response_length_unpadded",
        "response_mask_sha256_uint8",
        "policy_token_count",
        "masked_tool_or_padding_token_count",
        "turns",
    }
    missing = required - audit.keys()
    if missing:
        raise ValueError(f"turn-credit audit missing fields: {sorted(missing)}")
    if audit["schema_version"] != "toolcredit_turn_ledger_v1":
        raise ValueError("unexpected ledger schema")
    if audit["adoption_version"] != "toolcredit_adoption_v1":
        raise ValueError("unexpected adoption version")
    if not str(audit["wrapper_proof"]).startswith("e5-wrapper-v1:"):
        raise ValueError("Ray-driver wrapper proof is absent")

    response_length = int(audit["response_length_unpadded"])
    policy_count = int(audit["policy_token_count"])
    if policy_count <= 0 or response_length <= 0 or policy_count > response_length:
        raise ValueError("invalid response/policy token counts")
    previous_end = 0
    policy_span_count = 0
    tool_span_count = 0
    base_advantage: float | None = None
    for expected_index, turn in enumerate(audit["turns"]):
        if int(turn["turn_index"]) != expected_index:
            raise ValueError("noncontiguous turn index")
        start, end = map(int, turn["policy_token_span"])
        if not 0 <= start < end <= response_length or start < previous_end:
            raise ValueError("invalid or overlapping policy token span")
        policy_span_count += end - start
        previous_end = end
        tool_span = turn.get("tool_token_span")
        if tool_span is not None:
            tool_start, tool_end = map(int, tool_span)
            if not end <= tool_start < tool_end <= response_length:
                raise ValueError("invalid tool token span")
            tool_span_count += tool_end - tool_start
            previous_end = tool_end
        a_traj = float(turn["A_traj"])
        if base_advantage is None:
            base_advantage = a_traj
        elif not math.isclose(a_traj, base_advantage, abs_tol=1e-6):
            raise ValueError("native trajectory advantage differs across turns")
        expected_correction = 0.5 * (int(turn["s_t"]) - float(turn["s_bar_t"]))
        if not math.isclose(float(turn["correction"]), expected_correction, abs_tol=1e-6):
            raise ValueError("turn correction does not match frozen beta/formula")
        if not math.isclose(float(turn["A_turn"]), a_traj + expected_correction, abs_tol=1e-6):
            raise ValueError("A_turn does not equal A_traj + correction")
        if int(turn["position_denominator"]) <= 0:
            raise ValueError("invalid same-position group denominator")
        if int(turn["s_t"]) == 1 and (
            int(turn["execution_success"]) != 1 or int(turn["adopted"]) != 1
        ):
            raise ValueError("s_t=1 without execution success and adoption")
    if policy_span_count != policy_count:
        raise ValueError("policy spans do not conserve policy token count")
    if tool_span_count > int(audit["masked_tool_or_padding_token_count"]):
        raise ValueError("tool spans exceed masked token count")
    return {"policy_span_tokens": policy_span_count, "tool_span_tokens": tool_span_count}


def _load_labels(
    path: Path, *, label_key: str, reason_key: str
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    labels: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        label = str(item[label_key])
        candidate_id = str(item["candidate_id"])
        if label not in HUMAN_LABELS:
            raise ValueError(f"unknown human label: {label}")
        if candidate_id in labels:
            raise ValueError(f"duplicate human label: {candidate_id}")
        if not str(item.get(reason_key, "")).strip():
            raise ValueError(f"missing adjudication reason: {candidate_id}")
        labels[candidate_id] = item
    return labels


def analyze(run_dir: Path) -> dict[str, Any]:
    prediction_paths = sorted((run_dir / "predictions/train").glob("*.jsonl"), key=lambda p: int(p.stem))
    if not prediction_paths:
        raise FileNotFoundError("no E5 train rollout files found")
    proof = json.loads((run_dir / "wrapper_installation.json").read_text(encoding="utf-8"))
    if not proof.get("installed") or not proof.get("restored"):
        raise ValueError("Ray-driver wrapper installation/restoration proof is incomplete")

    candidates: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    proof_ids: set[str] = set()
    trajectory_count = 0
    turn_count = 0
    for path, row in _iter_rows(prediction_paths):
        audit = row.get("turn_credit_audit")
        if not isinstance(audit, dict):
            raise ValueError(f"missing structured turn_credit_audit in {path}")
        if row.get("turn_credit_wrapper_proof") != audit["wrapper_proof"]:
            raise ValueError("rollout and audit wrapper proofs differ")
        _validate_audit(audit)
        proof_ids.add(str(audit["wrapper_proof"]))
        trajectory_count += 1
        for turn in audit["turns"]:
            turn_count += 1
            status_counts[str(turn["adoption_status"])] += 1
            if int(turn["s_t"]) != 1:
                continue
            candidate_id = _candidate_id(run_dir.name, int(path.stem), audit, turn)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "stable_order_key": hashlib.sha256(candidate_id.encode("ascii")).hexdigest(),
                    "split": "train",
                    "step": int(path.stem),
                    "sample_id": row.get("sample_id"),
                    "prompt_uid": audit["prompt_uid"],
                    "trajectory_uid": audit["trajectory_uid"],
                    "turn": turn,
                    "trajectory_output": row.get("output"),
                    "allowed_human_labels": list(HUMAN_LABELS),
                }
            )
    if proof_ids != {proof["proof_id"]}:
        raise ValueError(f"wrapper proof drift across rollout data: {proof_ids}")
    candidates.sort(key=lambda item: item["stable_order_key"])
    candidates = candidates[:AUTO_ADOPTED_AUDIT_LIMIT]

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    human_labels_path = analysis_dir / "e5_adoption_human_review_labels.jsonl"
    human_labels = _load_labels(
        human_labels_path, label_key="human_label", reason_key="human_reason"
    )
    agent_labels_path = analysis_dir / "e5_adoption_agent_preliminary_labels.jsonl"
    agent_labels = _load_labels(
        agent_labels_path, label_key="agent_label", reason_key="agent_reason"
    )
    selected_ids = {item["candidate_id"] for item in candidates}
    if set(human_labels) - selected_ids:
        raise ValueError("human labels reference candidates outside the frozen selection")
    if set(agent_labels) - selected_ids:
        raise ValueError("agent labels reference candidates outside the frozen selection")
    candidate_path = analysis_dir / "e5_adoption_candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in candidates), encoding="utf-8"
    )

    decisive = [
        human_labels[item["candidate_id"]]
        for item in candidates
        if item["candidate_id"] in human_labels
        and human_labels[item["candidate_id"]]["human_label"] != "indecisive"
    ]
    true_count = sum(item["human_label"] == "true_adoption" for item in decisive)
    precision = true_count / len(decisive) if decisive else None
    automatic_gate_applicable = len(decisive) >= 10
    gate_passed = automatic_gate_applicable and precision is not None and precision >= 0.80
    explicit_review_required = not automatic_gate_applicable
    if len(candidates) < 10:
        explicit_review_required = True
        gate_passed = False
    agent_decisive = [
        agent_labels[item["candidate_id"]]
        for item in candidates
        if item["candidate_id"] in agent_labels
        and agent_labels[item["candidate_id"]]["agent_label"] != "indecisive"
    ]
    agent_true_count = sum(item["agent_label"] == "true_adoption" for item in agent_decisive)
    agent_precision = agent_true_count / len(agent_decisive) if agent_decisive else None
    summary = {
        "run_name": run_dir.name,
        "trajectory_count": trajectory_count,
        "turn_count": turn_count,
        "adoption_status_counts": dict(status_counts),
        "wrapper_proof_id": proof["proof_id"],
        "auto_adopted_available": sum(status_counts.values()) and status_counts.get("adopted", 0),
        "selected_auto_adopted_cases": len(candidates),
        "human_labeled_cases": len(human_labels),
        "decisively_adjudicated_cases": len(decisive),
        "true_adoption_cases": true_count,
        "heuristic_precision": precision,
        "automatic_80pct_gate_applicable": automatic_gate_applicable,
        "heuristic_precision_gate_passed": gate_passed,
        "explicit_review_required": explicit_review_required,
        "minimum_decisive_denominator": 10,
        "candidate_manifest": str(candidate_path.relative_to(run_dir)),
        "human_label_manifest": str(human_labels_path.relative_to(run_dir)),
        "agent_preliminary_labeled_cases": len(agent_labels),
        "agent_preliminary_decisive_cases": len(agent_decisive),
        "agent_preliminary_true_cases": agent_true_count,
        "agent_preliminary_precision": agent_precision,
        "agent_preliminary_counts_toward_human_gate": False,
        "agent_preliminary_manifest": str(agent_labels_path.relative_to(run_dir)),
    }
    (analysis_dir / "e5_smoke_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
