"""Exploratory step-125 sensitivity analysis for E3-NoTool (user-authorized 2026-09-07).

Not part of protocol v5: the role ``e3notool_grpo_step_125_sensitivity`` is never canonical,
never enters the v5 key universe, pairings or freeze closure, and its output is written to
its own run directory with an explicit ``exploratory`` status.  It answers one question
only: is the primary E3 -> E3-NoTool greedy FULL760 delta sensitive to evaluating the
fixed-100 peak checkpoint (step 125) instead of the preregistered step-200 endpoint?

Everything that generates or scores a trajectory is the v5 code path (same prompt, same
sampling, same verifier); only the checkpoint identity differs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze_v5 import E3_ROLE, MANIFEST_PATH as MANIFEST_V5, verify as verify_v5, verify_m7_canonical_ledger
from eval.checkpoints import (
    COMMON_TOKENIZER,
    MATERIALIZATION_LEDGER_ROOT,
    MATERIALIZED_ROOT,
    ROOT,
    SFT,
    _common_tokenizer_identity,
    _export_files,
    _load_yaml,
    _merger_source,
    _nested,
    _require_files,
    _tensor_inventory,
    sha256,
    utc_now,
)
from eval.checkpoints_v5 import GENERATION_MODE_NOTOOL, NOTOOL_RUN, REQUIRED_CHECKPOINT_FILES, ROLE_V5, SINGLE_TURN_AGENT_LOOP
from eval.generate import SGLangHTTPServer, SOURCE_COUNTS, load_panel_questions, validate_frozen_runtime_config
from eval.m8_e5v2_eval import _read_rows, _write_json, load_m7_labeled_rows, transitions_and_bootstrap
from eval.m9_notool_eval import (
    PROTOCOL_VERSION_V5,
    SENSITIVITY_ROLES,
    _require_approval,
    generate_shard_rows,
    notool_behavior,
    paired_delta_by_stratum,
    score_trajectory_v5,
)
from eval.metrics import pass_metrics

from analysis.badcase_taxonomy import enrich_with_coarse_proposal

SENSITIVITY_ROLE = SENSITIVITY_ROLES[0]
SENSITIVITY_STEP = 125
SOURCE_125 = NOTOOL_RUN / f"checkpoints/global_step_{SENSITIVITY_STEP}"
MATERIALIZED_125 = MATERIALIZED_ROOT / "m9_sensitivity_e3notool_grpo_step_125_hf"
MANIFEST_125 = MATERIALIZATION_LEDGER_ROOT / "sensitivity_e3notool_grpo_step_125.json"
SETTING = "greedy"
SOURCES = tuple(SOURCE_COUNTS)


def validate_step125_source() -> dict[str, Any]:
    """The resume-anchor checkpoint of the completed canonical run; explicit path, hash recorded."""
    path = SOURCE_125
    if not path.is_dir():
        raise FileNotFoundError(f"step-{SENSITIVITY_STEP} checkpoint does not exist: {path}")
    _require_files(path, REQUIRED_CHECKPOINT_FILES, SENSITIVITY_ROLE)
    status = json.loads((NOTOOL_RUN / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError("canonical E3-NoTool run is not completed")
    config = _load_yaml(NOTOOL_RUN / "resolved_config.yaml")
    if Path(str(_nested(config, "actor_rollout_ref.model.path"))).resolve() != SFT.resolve():
        raise ValueError("canonical run did not start from the frozen SFT checkpoint")
    if _nested(config, "actor_rollout_ref.rollout.agent.default_agent_loop") != SINGLE_TURN_AGENT_LOOP:
        raise ValueError("canonical run did not use the native single-turn loop")
    recovery = json.loads((NOTOOL_RUN / "recovery/resume_from_125_20260907_140752/recovery.json").read_text(encoding="utf-8"))
    if int(recovery.get("checkpoint_step", -1)) != SENSITIVITY_STEP:
        raise ValueError("recovery ledger does not anchor at step 125")
    return {
        "role": SENSITIVITY_ROLE,
        "state": "validated_exploratory",
        "exploratory": True,
        "generation_mode": GENERATION_MODE_NOTOOL,
        "weight_role": SENSITIVITY_ROLE,
        "canonical_counterpart": ROLE_V5,
        "source_locator": str(path.resolve()),
        "run_name": NOTOOL_RUN.name,
        "step": SENSITIVITY_STEP,
        "reason": "fixed-100 greedy peak (0.76 at step 125 vs 0.70 at step 200); user-authorized sensitivity check, not a checkpoint selection",
        "source_actor_sha256": sha256(path / "actor/model_world_size_1_rank_0.pt"),
        "resolved_config_sha256": sha256(NOTOOL_RUN / "resolved_config.yaml"),
        "recovery_ledger_sha256": sha256(NOTOOL_RUN / "recovery/resume_from_125_20260907_140752/recovery.json"),
    }


def materialize_step125(minimum_free_bytes: int = 40 * 1024**3) -> dict[str, Any]:
    source = validate_step125_source()
    if MATERIALIZED_125.exists() or MANIFEST_125.exists():
        raise FileExistsError("refusing to overwrite the existing step-125 materialization")
    if shutil.disk_usage(ROOT).free < minimum_free_bytes:
        raise RuntimeError("step-125 materialization disk gate failed")
    MATERIALIZED_125.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", str((SOURCE_125 / "actor").resolve()), "--target_dir", str(MATERIALIZED_125.resolve())]
    started_at = utc_now()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    completed_at = utc_now()
    if process.returncode != 0 or not MATERIALIZED_125.is_dir():
        raise RuntimeError(f"official veRL FSDP merge failed for step 125: {process.stderr[-4000:]}")
    (MATERIALIZED_125 / "materialization.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    import verl

    manifest = {
        "schema_version": "toolcredit_m7_materialization_manifest_v1",
        "role": SENSITIVITY_ROLE,
        "exploratory": True,
        "source": source,
        "target_locator": str(MATERIALIZED_125.resolve()),
        "started_at": started_at,
        "completed_at": completed_at,
        "command": command,
        "return_code": process.returncode,
        "official_fsdp_merge_validation": "passed_exit_zero",
        "verl_version": getattr(verl, "__version__", "0.8.0"),
        "merger_source_path": str(_merger_source()),
        "merger_source_sha256": sha256(_merger_source()),
        "output_files": _export_files(MATERIALIZED_125),
        "tensor_inventory": _tensor_inventory(MATERIALIZED_125),
        "common_tokenizer": _common_tokenizer_identity(),
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (MATERIALIZED_125 / "materialization_manifest.json").write_text(payload, encoding="utf-8")
    MANIFEST_125.write_text(payload, encoding="utf-8")
    return manifest


def resolved_step125_role() -> dict[str, Any]:
    role = validate_step125_source()
    internal = MATERIALIZED_125 / "materialization_manifest.json"
    if not MANIFEST_125.is_file() or MANIFEST_125.read_bytes() != internal.read_bytes():
        raise ValueError("step-125 internal/external materialization ledgers differ")
    manifest = json.loads(MANIFEST_125.read_text(encoding="utf-8"))
    if manifest["source"]["source_actor_sha256"] != role["source_actor_sha256"]:
        raise ValueError("step-125 materialization source linkage drifted")
    if {item["path"]: item for item in _export_files(MATERIALIZED_125)} != {item["path"]: item for item in manifest["output_files"]}:
        raise ValueError("step-125 materialized output file identity drifted")
    role["materialization"] = {
        "export_locator": str(MATERIALIZED_125.resolve()),
        "materialization_manifest": str(MANIFEST_125.relative_to(ROOT)),
        "materialization_manifest_sha256": sha256(MANIFEST_125),
        "tensor_count": manifest["tensor_inventory"]["tensor_count"],
        "total_parameters": manifest["tensor_inventory"]["total_parameters"],
    }
    role["inference_locator"] = str(MATERIALIZED_125.resolve())
    role["evaluator_tokenizer"] = _common_tokenizer_identity()
    return role


async def generate_greedy(endpoint: str, run_dir: Path) -> dict[str, Any]:
    _require_approval()
    config = validate_frozen_runtime_config()
    verify_v5()
    role = resolved_step125_role()
    server = SGLangHTTPServer(endpoint)
    await server.validate_model_path(Path(role["inference_locator"]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(COMMON_TOKENIZER)
    manifest_sha256 = sha256(MANIFEST_V5)
    panel = load_panel_questions()
    shards: dict[str, Any] = {}
    for source in SOURCES:
        questions = [row for row in panel if row["source"] == source]
        output = run_dir / "generation" / SENSITIVITY_ROLE / SETTING / f"{source}.jsonl"
        shards[source] = await generate_shard_rows(tokenizer, endpoint, SENSITIVITY_ROLE, SETTING, questions, role, manifest_sha256, config, output)
    return {"status": "completed", "role": SENSITIVITY_ROLE, "exploratory": True, "shards": shards}


def score_all(run_dir: Path) -> dict[str, Any]:
    questions = {row["question_id"]: row for row in load_panel_questions()}
    summary: dict[str, Any] = {}
    for source in SOURCES:
        raw_path = run_dir / "generation" / SENSITIVITY_ROLE / SETTING / f"{source}.jsonl"
        out_path = run_dir / "scored" / SENSITIVITY_ROLE / SETTING / f"{source}.jsonl"
        if out_path.exists():
            summary[source] = {"status": "existing", "output_sha256": sha256(out_path)}
            continue
        rows = [score_trajectory_v5(row, questions[row["question_id"]]) for row in _read_rows(raw_path)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary[source] = {"rows": len(rows), "output_sha256": sha256(out_path)}
    return summary


def downstream(run_dir: Path, canonical_run_dir: Path) -> dict[str, Any]:
    verify_v5()
    reuse = verify_m7_canonical_ledger()
    canonical_status = json.loads((canonical_run_dir / "status.json").read_text(encoding="utf-8"))
    if canonical_status.get("status") != "notool_evaluation_completed":
        raise ValueError("canonical M9 evaluation must be completed before the sensitivity comparison")
    questions = {row["question_id"]: row for row in load_panel_questions()}
    rows125 = [row for source in SOURCES for row in _read_rows(run_dir / "scored" / SENSITIVITY_ROLE / SETTING / f"{source}.jsonl")]
    if len(rows125) != 760 or len({row["question_id"] for row in rows125}) != 760:
        raise ValueError("sensitivity role must have exactly the 760 greedy panel rows")
    labeled125 = [enrich_with_coarse_proposal(row, str(questions[row["question_id"]]["answer"])) for row in rows125]
    rows200 = [row for source in SOURCES for row in _read_rows(canonical_run_dir / "scored" / ROLE_V5 / SETTING / f"{source}.jsonl")]
    labeled200 = [enrich_with_coarse_proposal(row, str(questions[row["question_id"]]["answer"])) for row in rows200]
    e3 = [row for row in load_m7_labeled_rows((E3_ROLE,))[E3_ROLE] if row["evaluation_setting"] == SETTING]
    transitions_dir = run_dir / "transitions"
    e3_to_125 = transitions_and_bootstrap(e3, labeled125, E3_ROLE, SENSITIVITY_ROLE, transitions_dir, "e3_to_step125")
    s200_to_125 = transitions_and_bootstrap(labeled200, labeled125, ROLE_V5, SENSITIVITY_ROLE, transitions_dir, "step200_to_step125")
    canonical_headline = json.loads((canonical_run_dir / "metrics/m9_headline.json").read_text(encoding="utf-8"))
    passes = pass_metrics(rows125)
    headline = {
        "protocol_version": PROTOCOL_VERSION_V5,
        "status": "exploratory_sensitivity",
        "interpretation_boundary": (
            "user-authorized sensitivity analysis on the fixed-100 peak checkpoint; the preregistered primary "
            "remains the step-200 comparison in the canonical run; this result may not replace it"
        ),
        "canonical_run": canonical_run_dir.name,
        "reused_m7_canonical": reuse,
        "greedy_full760_pass_at_1": {
            E3_ROLE: canonical_headline["greedy_full760_pass_at_1"][E3_ROLE],
            ROLE_V5: canonical_headline["greedy_full760_pass_at_1"][ROLE_V5],
            SENSITIVITY_ROLE: passes[f"{SENSITIVITY_ROLE}:greedy:FULL760"]["pass@1"],
        },
        "greedy_full760_pass_at_1_by_source": {key: value["pass@1"] for key, value in passes.items()},
        "paired_greedy_full760": {
            "e3_to_step125": {key: e3_to_125["summary"]["settings"][SETTING]["FULL760"][key] for key in ("denominator", "fixed_failures", "new_failures", "unchanged_correct", "unchanged_failed", "accuracy_delta")},
            "step200_to_step125": {key: s200_to_125["summary"]["settings"][SETTING]["FULL760"][key] for key in ("denominator", "fixed_failures", "new_failures", "unchanged_correct", "unchanged_failed", "accuracy_delta")},
            "e3_to_step200_canonical": canonical_headline["paired_greedy_full760"]["tool_net_contribution_at_rl_endpoint"],
        },
        "paired_greedy_full760_bootstrap": {
            "e3_to_step125": e3_to_125["bootstrap"]["settings"][SETTING]["FULL760"],
            "step200_to_step125": s200_to_125["bootstrap"]["settings"][SETTING]["FULL760"],
            "e3_to_step200_canonical": canonical_headline["paired_greedy_full760_bootstrap"]["tool_net_contribution_at_rl_endpoint"],
        },
        "e3_to_step125_by_stratum": paired_delta_by_stratum(e3, labeled125),
        "notool_behavior_greedy_full760": notool_behavior(rows125),
        "scientific_interpretation_performed": False,
    }
    _write_json(run_dir / "metrics" / "sensitivity_headline.json", headline)
    return headline


def write_closure(run_dir: Path) -> dict[str, Any]:
    entries = [f"{sha256(path)}  {path.relative_to(run_dir)}" for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name not in {"hashes.sha256", "status.json"} and "server_logs" not in path.parts and "stage_logs" not in path.parts]
    (run_dir / "hashes.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    status = {
        "run_name": run_dir.name,
        "protocol_version": PROTOCOL_VERSION_V5,
        "freeze_manifest_sha256": sha256(MANIFEST_V5),
        "status": "exploratory_sensitivity_completed",
        "exploratory": True,
        "completed_at": utc_now(),
        "hashes_sha256": sha256(run_dir / "hashes.sha256"),
        "artifact_count": len(entries),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize")
    sub.add_parser("resolve")
    gen = sub.add_parser("generate")
    gen.add_argument("--endpoint", default="http://127.0.0.1:30000")
    gen.add_argument("--run-dir", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--run-dir", type=Path, required=True)
    down = sub.add_parser("downstream")
    down.add_argument("--run-dir", type=Path, required=True)
    down.add_argument("--canonical-run-dir", type=Path, required=True)
    closure = sub.add_parser("closure")
    closure.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        result: Any = materialize_step125()
    elif args.command == "resolve":
        result = resolved_step125_role()
    elif args.command == "generate":
        result = asyncio.run(generate_greedy(args.endpoint, args.run_dir.resolve()))
    elif args.command == "score":
        result = score_all(args.run_dir.resolve())
    elif args.command == "downstream":
        result = downstream(args.run_dir.resolve(), args.canonical_run_dir.resolve())
    else:
        result = write_closure(args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
