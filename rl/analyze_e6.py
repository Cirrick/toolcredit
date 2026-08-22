"""Build E6 mask audits and no-mask failure candidate evidence from raw JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

LABELS = ("imitate", "forge", "repeat", "legitimate_quote", "none")
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def candidate_reasons(row: dict[str, Any]) -> list[str]:
    """Return conservative automatic reasons; human labels remain separate."""
    output = str(row.get("output", ""))
    n_calls = int(float(row.get("n_tool_calls", 0)))
    responses = [match.strip() for match in TOOL_RESPONSE_RE.findall(output) if match.strip()]
    reasons: list[str] = []
    if n_calls == 0 and ("<tool_response>" in output or "</tool_response>" in output):
        reasons.append("environment_wrapper_without_real_call")
    if n_calls > 0 and len(responses) != n_calls:
        reasons.append("tool_response_count_mismatch")
    if any(responses.count(response) > 1 for response in set(responses)):
        reasons.append("repeated_identical_tool_response")

    assistant_tail = output.rsplit("</tool_response>", maxsplit=1)[-1]
    for response in responses:
        normalized = " ".join(response.split())
        if len(normalized) >= 24 and " ".join(assistant_tail.split()).count(normalized) > 0:
            reasons.append("long_tool_output_replayed_by_assistant")
            break
    if re.search(r"(?:error:|traceback)[\s\S]{0,300}(?:error:|traceback)", assistant_tail, re.I):
        reasons.append("repeated_error_style_text")
    if re.search(r"\d{40,}", assistant_tail):
        reasons.append("long_unnatural_numeric_string")
    return sorted(set(reasons))


def _rows(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield path, json.loads(line)


def _candidate_id(split: str, step: int, row: dict[str, Any]) -> str:
    payload = "\0".join((split, str(step), str(row.get("sample_id", "unknown")), str(row.get("output", ""))))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _load_human_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    labels: dict[str, dict[str, str]] = {}
    for line in path.open(encoding="utf-8"):
        item = json.loads(line)
        candidate_id = str(item["candidate_id"])
        label = str(item["human_label"])
        reason = str(item["human_reason"])
        if label not in LABELS:
            raise ValueError(f"invalid E6 human label: {label!r}")
        if candidate_id in labels:
            raise ValueError(f"duplicate E6 human label: {candidate_id}")
        labels[candidate_id] = {"human_label": label, "human_reason": reason}
    return labels


def analyze(run_dir: Path) -> dict[str, Any]:
    train_paths = sorted(
        (run_dir / "predictions/train").glob("*.jsonl"), key=lambda path: int(path.stem)
    )
    validation_paths = sorted(
        (run_dir / "predictions/validation").glob("*.jsonl"), key=lambda path: int(path.stem)
    )
    if not train_paths:
        raise FileNotFoundError(f"no training predictions found under {run_dir}")

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    human_labels = _load_human_labels(analysis_dir / "e6_human_labels.jsonl")
    candidates: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    mask_totals = Counter[str]()
    row_count = 0
    rows_with_tool_returns = 0

    for split, paths in (("train", train_paths), ("validation", validation_paths)):
        for path, row in _rows(paths):
            row_count += 1
            if "nomask_loss_token_count" not in row:
                raise ValueError(f"missing E6 mask audit field in {path}")
            policy = int(float(row["original_policy_token_count"]))
            tool = int(float(row["original_tool_return_token_count"]))
            loss = int(float(row["nomask_loss_token_count"]))
            if policy + tool != loss:
                raise ValueError(f"mask token counts do not add up in {path}")
            mask_totals.update(policy_tokens=policy, tool_return_tokens=tool, loss_tokens=loss)
            rows_with_tool_returns += int(tool > 0)

            reasons = candidate_reasons(row)
            if reasons:
                candidate_id = _candidate_id(split, int(path.stem), row)
                human = human_labels.get(candidate_id, {})
                reason_counts.update(reasons)
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "split": split,
                        "step": int(path.stem),
                        "sample_id": str(row.get("sample_id", "unknown")),
                        "auto_reasons": reasons,
                        "human_label": human.get("human_label"),
                        "human_reason": human.get("human_reason"),
                        "allowed_human_labels": list(LABELS),
                        "trajectory": row,
                    }
                )

    candidate_path = analysis_dir / "e6_failure_candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates), encoding="utf-8"
    )
    candidate_ids = {row["candidate_id"] for row in candidates}
    missing_labels = sorted(set(human_labels) - candidate_ids)
    if missing_labels:
        raise ValueError(f"human labels reference missing candidates: {missing_labels}")
    audited = [row for row in candidates if row["human_label"] is not None]
    human_audit_path = analysis_dir / "e6_human_audit.jsonl"
    human_audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audited), encoding="utf-8"
    )
    loss_tokens = mask_totals["loss_tokens"]
    summary = {
        "run_name": run_dir.name,
        "rows": row_count,
        "rows_with_tool_returns": rows_with_tool_returns,
        "mask_totals": dict(mask_totals),
        "tool_return_loss_fraction": (
            mask_totals["tool_return_tokens"] / loss_tokens if loss_tokens else 0.0
        ),
        "candidate_count": len(candidates),
        "candidate_reason_counts": dict(reason_counts),
        "human_audited_count": len(audited),
        "human_label_counts": dict(Counter(row["human_label"] for row in audited)),
        "candidate_manifest": str(candidate_path.relative_to(run_dir)),
        "human_audit_manifest": str(human_audit_path.relative_to(run_dir)),
    }
    (analysis_dir / "mask_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
