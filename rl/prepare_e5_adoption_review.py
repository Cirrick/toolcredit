"""Create a compact blinded human-review packet for E5 adoption precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rl.analyze_e5 import _candidate_id, _iter_rows
from rl.custom.turn_advantage import normalize_visible_text

EXCERPT_RADIUS = 600
CONTEXT_LIMIT = 900
HUMAN_REVIEW_CASE_COUNT = 15
FORBIDDEN_KEYS = {
    "adopted",
    "adoption_status",
    "adoption_evidence",
    "s_t",
    "s_bar_t",
    "correction",
    "A_traj",
    "A_turn",
    "reward",
    "score",
    "acc",
    "correctness",
    "agent_label",
    "agent_reason",
    "human_label",
    "human_reason",
}


def _compact_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "[earlier context omitted]\n" + text[-limit:]


def _relevant_excerpt(text: str, match: str) -> str:
    position = text.find(match)
    source = text
    if position < 0:
        source = normalize_visible_text(text)
        position = source.find(normalize_visible_text(match))
    if position < 0:
        raise ValueError("cannot locate frozen adoption evidence in the later assistant span")
    start = max(0, position - EXCERPT_RADIUS)
    end = min(len(source), position + len(match) + EXCERPT_RADIUS)
    prefix = "[earlier span omitted]\n" if start else ""
    suffix = "\n[later span omitted]" if end < len(source) else ""
    return prefix + source[start:end] + suffix


def _assert_blinded(item: dict[str, Any]) -> None:
    overlap = FORBIDDEN_KEYS & item.keys()
    if overlap:
        raise ValueError(f"blinded review item leaked forbidden fields: {sorted(overlap)}")
    if set(item) != {
        "review_case_id",
        "minimal_context_before_tool",
        "visible_post_truncation_tool_response",
        "relevant_later_assistant_span",
    }:
        raise ValueError(f"unexpected blinded review fields: {sorted(item)}")


def prepare(run_dir: Path) -> dict[str, Any]:
    analysis_dir = run_dir / "analysis"
    candidate_path = analysis_dir / "e5_adoption_candidates.jsonl"
    candidates = [
        json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines() if line
    ]
    all_candidate_ids = [str(item["candidate_id"]) for item in candidates]
    if len(all_candidate_ids) != 50 or len(set(all_candidate_ids)) != 50:
        raise ValueError("expected the preregistered 50 unique stable-hash candidates")
    candidate_order = all_candidate_ids[:HUMAN_REVIEW_CASE_COUNT]
    preliminary_path = analysis_dir / "e5_adoption_agent_preliminary_labels.jsonl"
    preliminary_ids = [
        str(json.loads(line)["candidate_id"])
        for line in preliminary_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if preliminary_ids != candidate_order:
        raise ValueError("human review packet must use the exact same stable-hash cases Codex saw")

    source: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    prediction_paths = sorted((run_dir / "predictions/train").glob("*.jsonl"), key=lambda p: int(p.stem))
    for path, row in _iter_rows(prediction_paths):
        audit = row["turn_credit_audit"]
        for turn in audit["turns"]:
            if int(turn["s_t"]) != 1:
                continue
            candidate_id = _candidate_id(run_dir.name, int(path.stem), audit, turn)
            if candidate_id in candidate_order:
                source[candidate_id] = (audit, turn)
    if set(source) != set(candidate_order):
        raise ValueError("could not recover every stable-hash review case from structured rollout data")

    packet: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []
    key_rows: list[dict[str, str]] = []
    for index, candidate_id in enumerate(candidate_order, start=1):
        audit, turn = source[candidate_id]
        evidence = turn.get("adoption_evidence")
        if not isinstance(evidence, dict) or "later_turn_offset" not in evidence:
            raise ValueError(f"review case lacks a structured later-turn locator: {candidate_id}")
        target_index = int(turn["turn_index"]) + int(evidence["later_turn_offset"])
        later_turns = audit["turns"]
        if not 0 <= target_index < len(later_turns):
            raise ValueError(f"later assistant locator is out of range: {candidate_id}")
        match = str(evidence.get("match", evidence.get("candidate", "")))
        if not match:
            raise ValueError(f"review case lacks locatable evidence text: {candidate_id}")
        review_case_id = f"TC-ADOPT-{index:03d}"
        item = {
            "review_case_id": review_case_id,
            "minimal_context_before_tool": _compact_tail(str(turn.get("assistant_text", "")), CONTEXT_LIMIT),
            "visible_post_truncation_tool_response": turn.get("visible_tool_response_text"),
            "relevant_later_assistant_span": _relevant_excerpt(
                str(later_turns[target_index].get("assistant_text", "")), match
            ),
        }
        _assert_blinded(item)
        packet.append(item)
        template.append(
            {
                "review_case_id": review_case_id,
                "human_label": None,
                "human_reason": "",
            }
        )
        key_rows.append({"review_case_id": review_case_id, "candidate_id": candidate_id})

    packet_path = analysis_dir / "e5_adoption_human_review_blinded.jsonl"
    template_path = analysis_dir / "e5_adoption_human_review_template.jsonl"
    key_path = analysis_dir / "e5_adoption_human_review_key.jsonl"
    packet_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in packet),
        encoding="utf-8",
    )
    template_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in template),
        encoding="utf-8",
    )
    key_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in key_rows), encoding="utf-8"
    )
    return {
        "review_case_count": len(packet),
        "blinded_packet": str(packet_path.relative_to(run_dir)),
        "blank_label_template": str(template_path.relative_to(run_dir)),
        "join_key": str(key_path.relative_to(run_dir)),
        "forbidden_review_fields": sorted(FORBIDDEN_KEYS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
