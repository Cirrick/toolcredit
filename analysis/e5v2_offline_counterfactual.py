"""Offline counterfactual for M8 / E5-v2 on the frozen E5 v1 train ledger (plan §4, step 0).

Reads E5 v1 rollout ledgers read-only, re-derives ``s_t`` under the v2 conditions
(plan §3.2), applies the conserved formula (plan §3.1) and reports, per step window:
informative-group share, multi-call share, treatment exposure, conservation residuals
and the zero-variance identity count.  No GPU, no writes outside ``--output``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rl.analyze_e4 import extract_codes
from rl.custom.turn_advantage_v2 import (
    CONSERVATION_TOLERANCE,
    FROZEN_BETA,
    conserved_turn_corrections,
    v2_turn_conditions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT_ROOT / "rl/runs/e5_turn_credit_20260823_082012"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis/e5v2_offline_counterfactual.json"
DEFAULT_WINDOWS = ("1-25", "176-200")
INELIGIBLE_STATUSES = {"no_tool", "ambiguous_multiple_calls", "parser_error"}
SCHEMA_VERSION = "toolcredit_e5v2_offline_counterfactual_v1"


def parse_window(text: str) -> tuple[int, int]:
    start, end = (int(part) for part in text.split("-", maxsplit=1))
    if not 1 <= start <= end:
        raise ValueError(f"invalid step window: {text!r}")
    return start, end


def iter_rows(run_dir: Path, steps: Iterable[int]) -> Iterable[dict[str, Any]]:
    for step in steps:
        path = run_dir / "predictions/train" / f"{step}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing train ledger for step {step}: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def turn_eligibility(turn: dict[str, Any]) -> bool:
    """v1 audit rows do not carry parsed_call_count; adoption_status encodes the same partition."""
    return str(turn.get("adoption_status")) not in INELIGIBLE_STATUSES


def trajectory_counterfactual(
    audit: dict[str, Any], *, informative_group: bool, beta: float = FROZEN_BETA
) -> dict[str, Any]:
    """Recompute v2 ``s_t`` and conserved corrections for one v1 audit record."""
    turns = audit["turns"]
    codes_by_turn = [extract_codes(str(turn.get("assistant_text", ""))) for turn in turns]
    texts = [str(turn.get("assistant_text", "")) for turn in turns]
    s_v1 = [int(turn["s_t"]) for turn in turns]
    eligible = [turn_eligibility(turn) for turn in turns]
    conditions = v2_turn_conditions(codes_by_turn, texts, s_v1, eligible)
    token_counts = [
        int(turn["policy_token_span"][1]) - int(turn["policy_token_span"][0]) for turn in turns
    ]
    s_v2 = [item["s_t"] for item in conditions]
    corrections, s_tilde, gated = conserved_turn_corrections(
        s_v2, token_counts, eligible, beta=beta, informative_group=informative_group
    )
    ungated, _, _ = conserved_turn_corrections(
        s_v2, token_counts, eligible, beta=beta, informative_group=True
    )
    residual = sum(c * n for c, n in zip(corrections, token_counts, strict=True))
    v1_shift = sum(float(turn["correction"]) * n for turn, n in zip(turns, token_counts, strict=True))
    return {
        "eligible_turn_count": sum(eligible),
        "informative_group": informative_group,
        "gated": gated,
        "s_tilde": s_tilde,
        "treated": any(value != 0.0 for value in corrections),
        "treated_if_ungated": any(value != 0.0 for value in ungated),
        "conservation_residual": residual,
        "v1_token_weighted_shift": v1_shift,
        "s_t_v1": s_v1,
        "s_t_v2": s_v2,
        "repeat_of_turn": [item["repeat_of_turn"] for item in conditions],
        "after_boxed": [item["after_boxed"] for item in conditions],
        "corrections": corrections,
    }


def _group_informative(scores: list[float]) -> bool:
    return max(scores) > min(scores)


def analyze_window(run_dir: Path, start: int, end: int) -> dict[str, Any]:
    by_group: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_rows(run_dir, range(start, end + 1)):
        audit = row["turn_credit_audit"]
        by_group[(int(row["step"]), str(audit["prompt_uid"]))].append(row)

    trajectories = groups = informative_groups = multi_call = 0
    treated = treated_ungated = zero_variance_rows = zero_variance_nonzero = 0
    s1_v1 = s1_v2 = eligible_turns = repeats = after_boxed = 0
    max_residual = 0.0
    v1_shift_total = 0.0
    for (_, _), rows in by_group.items():
        groups += 1
        scores = [float(row["score"]) for row in rows]
        informative = _group_informative(scores)
        informative_groups += int(informative)
        for row in rows:
            trajectories += 1
            result = trajectory_counterfactual(row["turn_credit_audit"], informative_group=informative)
            multi_call += int(result["eligible_turn_count"] >= 2)
            treated += int(result["treated"])
            treated_ungated += int(result["treated_if_ungated"])
            max_residual = max(max_residual, abs(result["conservation_residual"]))
            v1_shift_total += result["v1_token_weighted_shift"]
            if not informative:
                zero_variance_rows += 1
                zero_variance_nonzero += int(result["treated"])
            eligible_turns += result["eligible_turn_count"]
            s1_v1 += sum(result["s_t_v1"])
            s1_v2 += sum(result["s_t_v2"])
            repeats += sum(value is not None for value in result["repeat_of_turn"])
            after_boxed += sum(result["after_boxed"])
    if trajectories == 0:
        raise ValueError(f"no trajectories in steps {start}-{end}")
    if max_residual > CONSERVATION_TOLERANCE:
        raise ValueError(f"conservation violated in window {start}-{end}: {max_residual}")
    if zero_variance_nonzero:
        raise ValueError("zero-variance group received a nonzero v2 correction")
    return {
        "steps": [start, end],
        "trajectories": trajectories,
        "groups": groups,
        "informative_group_fraction": informative_groups / groups,
        "multi_call_trajectory_fraction": multi_call / trajectories,
        "treated_fraction_ungated": treated_ungated / trajectories,
        "treated_fraction_v2": treated / trajectories,
        "max_abs_conservation_residual": max_residual,
        "zero_variance_trajectories": zero_variance_rows,
        "zero_variance_trajectories_with_nonzero_correction": zero_variance_nonzero,
        "eligible_turns": eligible_turns,
        "s_t_v1_positive_turns": s1_v1,
        "s_t_v2_positive_turns": s1_v2,
        "turns_flagged_repeat": repeats,
        "turns_flagged_after_boxed": after_boxed,
        "v1_mean_token_weighted_shift_per_trajectory": v1_shift_total / trajectories,
    }


def analyze(run_dir: Path, windows: Iterable[str], *, exposure_floor: float = 0.08) -> dict[str, Any]:
    results = {window: analyze_window(run_dir, *parse_window(window)) for window in windows}
    first = results[next(iter(results))]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_run": run_dir.name,
        "beta": FROZEN_BETA,
        "conservation_tolerance": CONSERVATION_TOLERANCE,
        "windows": results,
        "preregistered_exposure_floor": exposure_floor,
        "early_window_exposure_above_floor": first["treated_fraction_v2"] >= exposure_floor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--windows", nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(args.run_dir.resolve(), args.windows)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for window, result in report["windows"].items():
        print(
            f"{window}: informative={result['informative_group_fraction']:.3f} "
            f"multi_call={result['multi_call_trajectory_fraction']:.3f} "
            f"ungated={result['treated_fraction_ungated']:.3f} "
            f"v2={result['treated_fraction_v2']:.3f} "
            f"max_residual={result['max_abs_conservation_residual']:.2e}"
        )
    if not math.isfinite(report["windows"][args.windows[0]]["treated_fraction_v2"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
