"""Descriptive accuracy conditional on tool use, per checkpoint role (qa_log Q14).

Reads the canonical M7 (raw/SFT/E3/E5-v1) and M8 (E5-v2) scored trajectories and
reports, for each role x setting x source, the count and accuracy of trajectories
that made zero tool calls, at least one call, at least one call with no execution
error, and at least one call with an execution error.

This is correlational evidence only: the policy chooses which questions get a tool
call, so the strata are not comparable populations. It does not replace the
CoT-only RL control proposed in plans/M9.md. Read-only; no canonical file is touched.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROLE_DIRS: dict[str, Path] = {
    "raw_base_model": ROOT / "eval/runs/m7_unified_eval_20260824_201032/scored/raw_base_model",
    "sft_start": ROOT / "eval/runs/m7_unified_eval_20260824_201032/scored/sft_start",
    "e3_grpo_baseline_step_200": ROOT / "eval/runs/m7_unified_eval_20260824_201032/scored/e3_grpo_baseline_step_200",
    "e5_turn_credit_step_200": ROOT / "eval/runs/m7_unified_eval_20260824_201032/scored/e5_turn_credit_step_200",
    "e5v2_conserved_credit_step_200": ROOT / "eval/runs/m8_e5v2_eval_20260905_134737/scored/e5v2_conserved_credit_step_200",
}
SETTINGS = ("greedy", "sampled")
STRATA = ("no_tool", "any_tool", "tool_all_success", "tool_any_error")


def _stratum(row: dict[str, Any]) -> tuple[str, ...]:
    meta = row["tool_metadata"]
    calls = int(meta["tool_call_counts"])
    if calls == 0:
        return ("no_tool",)
    errors = int(meta["tool_error_count"])
    return ("any_tool", "tool_any_error" if errors > 0 else "tool_all_success")


def _cell(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    n = len(rows)
    correct = sum(1 for r in rows if bool(r["correct"]))
    return {"n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else None}


def summarize(role_dirs: dict[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role, base in role_dirs.items():
        out[role] = {}
        for setting in SETTINGS:
            buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
            for path in sorted((base / setting).glob("*.jsonl")):
                for line in path.open():
                    row = json.loads(line)
                    keys = _stratum(row)
                    for group in (row["source"], "FULL760", row.get("stratum", "unknown")):
                        for key in keys:
                            buckets[group][key].append(row)
            out[role][setting] = {
                group: {key: _cell(buckets[group].get(key, [])) for key in STRATA}
                for group in sorted(buckets)
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/tool_use_conditional_accuracy.json")
    args = parser.parse_args()
    result = {
        "note": "descriptive, selection-biased; tool-use is policy-chosen, strata are not comparable populations",
        "roles": summarize(ROLE_DIRS),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for role, per_setting in result["roles"].items():
        cells = per_setting["greedy"]["FULL760"]
        print(role, {k: (v["n"], v["accuracy"]) for k, v in cells.items()})


if __name__ == "__main__":
    main()
