"""Finalize user-confirmed E5 adoption labels without using agent judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rl.analyze_e5 import HUMAN_LABELS, analyze

EXPECTED_REVIEW_CASES = 15
WILSON_Z_95 = 1.959963984540054


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the two-sided Wilson score interval with 95% coverage."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError(f"invalid Wilson counts: successes={successes}, total={total}")
    proportion = successes / total
    z_squared = WILSON_Z_95**2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def _unique_index(rows: list[dict[str, Any]], key: str, source: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            raise ValueError(f"{source} contains an empty {key}")
        if value in index:
            raise ValueError(f"{source} contains duplicate {key}: {value}")
        index[value] = row
    return index


def join_rows(
    templates: list[dict[str, Any]],
    keys: list[dict[str, Any]],
    blinded: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Validate exact 15/15 alignment and join review IDs to candidate IDs."""
    counts = {
        "template": len(templates),
        "join_key": len(keys),
        "blinded_packet": len(blinded),
    }
    if set(counts.values()) != {EXPECTED_REVIEW_CASES}:
        raise ValueError(f"human review artifacts must each have exactly 15 cases: {counts}")
    if len(candidates) < EXPECTED_REVIEW_CASES:
        raise ValueError("candidate manifest has fewer than 15 frozen stable-hash cases")

    template_by_id = _unique_index(templates, "review_case_id", "template")
    key_by_id = _unique_index(keys, "review_case_id", "join key")
    blinded_by_id = _unique_index(blinded, "review_case_id", "blinded packet")
    case_ids = set(template_by_id)
    if case_ids != set(key_by_id) or case_ids != set(blinded_by_id):
        raise ValueError("missing or extra review_case_id across template/key/blinded artifacts")

    candidate_by_id = _unique_index(candidates, "candidate_id", "candidate manifest")
    keyed_candidate_ids = [str(row.get("candidate_id", "")) for row in keys]
    if len(set(keyed_candidate_ids)) != EXPECTED_REVIEW_CASES or "" in keyed_candidate_ids:
        raise ValueError("join key contains missing or duplicate candidate_id")
    frozen_first_15 = [str(row["candidate_id"]) for row in candidates[:EXPECTED_REVIEW_CASES]]
    if keyed_candidate_ids != frozen_first_15:
        raise ValueError("join key does not match the frozen first 15 stable-hash candidates")
    if any(candidate_id not in candidate_by_id for candidate_id in keyed_candidate_ids):
        raise ValueError("join key references a missing candidate")

    joined: list[dict[str, str]] = []
    for template in templates:
        if set(template) != {"review_case_id", "human_label", "human_reason"}:
            raise ValueError(f"unexpected template fields for {template.get('review_case_id')}")
        review_case_id = str(template["review_case_id"])
        label = template["human_label"]
        reason = str(template["human_reason"]).strip()
        if label not in HUMAN_LABELS:
            raise ValueError(f"missing or invalid human label for {review_case_id}: {label!r}")
        if not reason:
            raise ValueError(f"missing human reason for {review_case_id}")
        candidate_id = str(key_by_id[review_case_id]["candidate_id"])
        joined.append(
            {
                "candidate_id": candidate_id,
                "human_label": str(label),
                "human_reason": reason,
                "review_case_id": review_case_id,
                "adjudicator": "user",
                "adjudication_provenance": "user_confirmed_2026-08-22",
            }
        )
    if len({row["candidate_id"] for row in joined}) != EXPECTED_REVIEW_CASES:
        raise ValueError("joined manifest does not contain 15 unique candidate IDs")
    return joined


def finalize(run_dir: Path) -> dict[str, Any]:
    analysis_dir = run_dir / "analysis"
    template_path = analysis_dir / "e5_adoption_human_review_template.jsonl"
    key_path = analysis_dir / "e5_adoption_human_review_key.jsonl"
    blinded_path = analysis_dir / "e5_adoption_human_review_blinded.jsonl"
    candidate_path = analysis_dir / "e5_adoption_candidates.jsonl"
    joined_path = analysis_dir / "e5_adoption_human_review_labels.jsonl"

    joined = join_rows(
        _read_jsonl(template_path),
        _read_jsonl(key_path),
        _read_jsonl(blinded_path),
        _read_jsonl(candidate_path),
    )
    joined_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in joined),
        encoding="utf-8",
    )
    formal = analyze(run_dir)

    true_positive = sum(row["human_label"] == "true_adoption" for row in joined)
    false_positive = sum(row["human_label"] == "false_adoption" for row in joined)
    uncertain = sum(row["human_label"] == "indecisive" for row in joined)
    decisive = true_positive + false_positive
    precision = true_positive / decisive if decisive else None
    if decisive == 0:
        raise ValueError("formal human gate has no decisive labels")
    lower, upper = wilson_interval(true_positive, decisive)
    expected_gate = decisive >= 10 and precision is not None and precision >= 0.80
    expected = {
        "human_labeled_cases": EXPECTED_REVIEW_CASES,
        "decisively_adjudicated_cases": decisive,
        "true_adoption_cases": true_positive,
        "heuristic_precision": precision,
        "automatic_80pct_gate_applicable": decisive >= 10,
        "heuristic_precision_gate_passed": expected_gate,
        "explicit_review_required": decisive < 10,
    }
    for key, value in expected.items():
        actual = formal[key]
        if isinstance(value, float):
            if not math.isclose(float(actual), value, abs_tol=1e-12):
                raise ValueError(f"formal analyzer disagrees on {key}: {actual} != {value}")
        elif actual != value:
            raise ValueError(f"formal analyzer disagrees on {key}: {actual} != {value}")

    report = {
        "schema_version": "toolcredit_human_adoption_gate_v1",
        "run_name": run_dir.name,
        "label_source": "user-confirmed human review template only",
        "agent_preliminary_labels_used": False,
        "exact_case_alignment": {
            "expected": EXPECTED_REVIEW_CASES,
            "joined": len(joined),
            "missing_review_case_ids": [],
            "duplicate_review_case_ids": [],
            "missing_candidate_ids": [],
            "duplicate_candidate_ids": [],
        },
        "counts": {
            "decisive_denominator": decisive,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "uncertain": uncertain,
        },
        "precision": precision,
        "wilson_95_ci": {"lower": lower, "upper": upper},
        "minimum_decisive_denominator": 10,
        "minimum_precision": 0.80,
        "automatic_gate_applicable": decisive >= 10,
        "gate_passed": expected_gate,
        "template_sha256": _sha256(template_path),
        "join_key_sha256": _sha256(key_path),
        "joined_manifest": str(joined_path.relative_to(run_dir)),
        "joined_manifest_sha256": _sha256(joined_path),
        "formal_analyzer_artifact": "analysis/e5_smoke_audit.json",
    }
    report_path = analysis_dir / "e5_human_adoption_gate.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
