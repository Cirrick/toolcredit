"""M9 / protocol v5: the NoTool role universe and the E3-NoTool step-200 checkpoint role.

The frozen v3 module ``eval/checkpoints.py`` and the v4 module ``eval/checkpoints_v4.py`` are
hashed by the M7/M8 freeze closures and are not modified.  v5 adds four ``notool`` roles
(plans/M9.md §3):

* ``raw_base_model_notool`` / ``sft_start_notool`` / ``e3_grpo_baseline_step_200_notool`` reuse
  the frozen v3 weights (raw Qwen3-1.7B, the SFT checkpoint, the M7 E3 materialization) and
  differ from their TIR counterparts only in the generation mode;
* ``e3notool_grpo_step_200`` is the only newly trained checkpoint; its source validation,
  materialization and runtime binding mirror the M8 E5-v2 role.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.checkpoints import (
    COMMON_TOKENIZER,
    MATERIALIZATION_LEDGER_ROOT,
    MATERIALIZED_PATHS,
    MATERIALIZED_ROOT,
    RAW,
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
    validate_materialization,
    validate_source_role,
)

ROLE_V5 = "e3notool_grpo_step_200"
GENERATION_MODE_NOTOOL = "notool"
GENERATION_MODE_TIR = "tir"
NOTOOL_RUN = ROOT / "rl/runs/e3notool_grpo_20260906_220907"
NOTOOL_SOURCE = NOTOOL_RUN / "checkpoints/global_step_200"
NOTOOL_MATERIALIZED = MATERIALIZED_ROOT / "m9_e3notool_grpo_step_200_hf"
NOTOOL_MANIFEST = MATERIALIZATION_LEDGER_ROOT / f"{ROLE_V5}.json"
NOTOOL_COMPLETION_GATE = NOTOOL_RUN / "analysis/formal_completion_gate.json"
NOTOOL_RECOVERY_LEDGER = NOTOOL_RUN / "recovery/resume_from_125_20260907_140752/recovery.json"
REQUIRED_CHECKPOINT_FILES = (
    "actor/model_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "actor/fsdp_config.json",
    "data.pt",
)
# Plan §2: the complete set of resolved-config paths E3 -> E3-NoTool may differ on.  Kept as a
# literal here so the evaluator does not import the launcher; the test asserts equality.
E3_CONTROL_DIFF_PATHS = (
    "actor_rollout_ref.rollout.agent.agent_loop_config_path",
    "actor_rollout_ref.rollout.agent.default_agent_loop",
    "actor_rollout_ref.rollout.multi_turn.enable",
    "actor_rollout_ref.rollout.trace.project_name",
    "trainer.project_name",
)
SINGLE_TURN_AGENT_LOOP = "single_turn_agent"

# notool role -> the frozen weight role whose inference locator it reuses.
NOTOOL_WEIGHT_ROLES = {
    "raw_base_model_notool": "raw_base_model",
    "sft_start_notool": "sft_start",
    "e3_grpo_baseline_step_200_notool": "e3_grpo_baseline_step_200",
    ROLE_V5: ROLE_V5,
}
NOTOOL_ROLE_NAMES = tuple(NOTOOL_WEIGHT_ROLES)
REUSED_WEIGHT_LOCATORS = {
    "raw_base_model": RAW,
    "sft_start": SFT,
    "e3_grpo_baseline_step_200": MATERIALIZED_PATHS["e3_grpo_baseline_step_200"],
}


def validate_e3notool_source_role(locator: Path | None = None) -> dict[str, Any]:
    """Explicit, completed, gate-validated E3-NoTool step-200 source; null/glob/swap fail closed."""
    path = NOTOOL_SOURCE if locator is None else Path(locator)
    if path.resolve() != NOTOOL_SOURCE.resolve() or path.name != "global_step_200":
        raise ValueError(f"{ROLE_V5} locator is not its explicit canonical step-200 path: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{ROLE_V5} locator does not exist: {path}")
    _require_files(path, REQUIRED_CHECKPOINT_FILES, ROLE_V5)
    sizes = {relative: (path / relative).stat().st_size for relative in REQUIRED_CHECKPOINT_FILES}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError(f"{ROLE_V5} checkpoint has an empty required file")
    run = path.parents[1]
    if run.resolve() != NOTOOL_RUN.resolve():
        raise ValueError(f"{ROLE_V5} checkpoint does not belong to the canonical run: {run}")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    tracker = (path.parent / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip()
    if status.get("state") != "completed" or tracker != "200":
        raise ValueError(f"{ROLE_V5} status/tracker is not completed step 200")
    config_path = run / "resolved_config.yaml"
    config = _load_yaml(config_path)
    if Path(str(_nested(config, "actor_rollout_ref.model.path"))).resolve() != SFT.resolve():
        raise ValueError(f"{ROLE_V5} did not start from the frozen SFT checkpoint")
    if int(_nested(config, "trainer.total_training_steps")) != 200:
        raise ValueError(f"{ROLE_V5} is not a 200-step run")
    if _nested(config, "actor_rollout_ref.rollout.multi_turn.enable") is not False:
        raise ValueError(f"{ROLE_V5} resolved config did not disable the multi-turn tool loop")
    if _nested(config, "actor_rollout_ref.rollout.agent.default_agent_loop") != SINGLE_TURN_AGENT_LOOP:
        raise ValueError(f"{ROLE_V5} did not train with veRL's native {SINGLE_TURN_AGENT_LOOP}")
    if _nested(config, "actor_rollout_ref.rollout.agent.agent_loop_config_path") is not None:
        raise ValueError(f"{ROLE_V5} registered a custom agent loop")
    gate_path = run / "analysis/formal_completion_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"{ROLE_V5} formal completion gate has not been run: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "completed" or gate.get("run_name") != run.name:
        raise ValueError(f"{ROLE_V5} formal completion gate did not pass for this run")
    rollouts = gate.get("rollouts", {})
    if int(rollouts.get("trajectory_count", 0)) != 200 * 512:
        raise ValueError(f"{ROLE_V5} formal completion gate does not cover 200 x 512 trajectories")
    if rollouts.get("tool_call_counts_all_zero") is not True or rollouts.get("prompts_free_of_tool_schema") is not True:
        raise ValueError(f"{ROLE_V5} completion gate does not prove tool-free training")
    if gate.get("tool_tag_gate", {}).get("pause_required") is not False:
        raise ValueError(f"{ROLE_V5} tool-tag gate was triggered")
    observed_diff = tuple(gate.get("resolved_config", {}).get("e3_control_diff_paths", []))
    if observed_diff != tuple(sorted(E3_CONTROL_DIFF_PATHS)):
        raise ValueError(f"{ROLE_V5} E3-control diff is not the preregistered set: {observed_diff}")
    if int(gate.get("checkpoint", {}).get("step", 0)) != 200:
        raise ValueError(f"{ROLE_V5} completion gate does not bind the step-200 checkpoint")
    recovery_events = list(gate.get("recovery_events", []))
    summary: dict[str, Any] = {
        "role": ROLE_V5,
        "state": "validated",
        "generation_mode": GENERATION_MODE_NOTOOL,
        "weight_role": ROLE_V5,
        "source_locator": str(path.resolve()),
        "run_name": run.name,
        "required_file_bytes": sizes,
        "source_actor_sha256": sha256(path / "actor/model_world_size_1_rank_0.pt"),
        "resolved_config_sha256": sha256(config_path),
        "status_sha256": sha256(run / "status.json"),
        "tracker_sha256": sha256(path.parent / "latest_checkpointed_iteration.txt"),
        "formal_completion_gate_sha256": sha256(gate_path),
        "e3_control": {
            "diff_paths": list(sorted(E3_CONTROL_DIFF_PATHS)),
            "agent_loop": SINGLE_TURN_AGENT_LOOP,
            "multi_turn_enabled": False,
            "training_tool_calls": 0,
        },
        "recovery_events": recovery_events,
    }
    if recovery_events:
        if not NOTOOL_RECOVERY_LEDGER.is_file():
            raise FileNotFoundError(f"{ROLE_V5} recovery ledger is missing: {NOTOOL_RECOVERY_LEDGER}")
        summary["recovery_ledger_sha256"] = sha256(NOTOOL_RECOVERY_LEDGER)
    return summary


def materialize_e3notool(minimum_free_bytes: int = 40 * 1024**3) -> dict[str, Any]:
    """Run the pinned official merger once; identical command shape to the frozen M7/M8 roles."""
    source_summary = validate_e3notool_source_role()
    if NOTOOL_MATERIALIZED.exists() or NOTOOL_MANIFEST.exists():
        raise FileExistsError(f"refusing to overwrite existing materialization for {ROLE_V5}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(f"M9 materialization disk gate failed: {free_bytes} < {minimum_free_bytes}")
    NOTOOL_MATERIALIZED.parent.mkdir(parents=True, exist_ok=True)
    NOTOOL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str((NOTOOL_SOURCE / "actor").resolve()),
        "--target_dir",
        str(NOTOOL_MATERIALIZED.resolve()),
    ]
    started_at = utc_now()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    completed_at = utc_now()
    if process.returncode != 0:
        raise RuntimeError(
            f"official veRL FSDP merge failed for {ROLE_V5} with code {process.returncode}: {process.stderr[-4000:]}"
        )
    if not NOTOOL_MATERIALIZED.is_dir():
        raise RuntimeError("official merger returned success without creating target directory")
    (NOTOOL_MATERIALIZED / "materialization.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    import verl

    manifest = {
        "schema_version": "toolcredit_m7_materialization_manifest_v1",
        "role": ROLE_V5,
        "source": source_summary,
        "target_locator": str(NOTOOL_MATERIALIZED.resolve()),
        "started_at": started_at,
        "completed_at": completed_at,
        "command": command,
        "return_code": process.returncode,
        "official_fsdp_merge_validation": "passed_exit_zero",
        "verl_version": getattr(verl, "__version__", "0.8.0"),
        "merger_source_path": str(_merger_source()),
        "merger_source_sha256": sha256(_merger_source()),
        "output_files": _export_files(NOTOOL_MATERIALIZED),
        "tensor_inventory": _tensor_inventory(NOTOOL_MATERIALIZED),
        "common_tokenizer": _common_tokenizer_identity(),
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (NOTOOL_MATERIALIZED / "materialization_manifest.json").write_text(payload, encoding="utf-8")
    NOTOOL_MANIFEST.write_text(payload, encoding="utf-8")
    return manifest


def validate_e3notool_materialization() -> dict[str, Any]:
    source = validate_e3notool_source_role()
    internal = NOTOOL_MATERIALIZED / "materialization_manifest.json"
    _require_files(NOTOOL_MATERIALIZED, ("materialization_manifest.json", "config.json"), ROLE_V5)
    if not NOTOOL_MANIFEST.is_file() or NOTOOL_MANIFEST.read_bytes() != internal.read_bytes():
        raise ValueError(f"{ROLE_V5} internal/external materialization ledgers differ")
    manifest = json.loads(NOTOOL_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("role") != ROLE_V5 or Path(manifest.get("target_locator", "")).resolve() != NOTOOL_MATERIALIZED.resolve():
        raise ValueError(f"{ROLE_V5} materialization manifest role/target mismatch")
    if manifest.get("source", {}).get("source_actor_sha256") != source["source_actor_sha256"]:
        raise ValueError(f"{ROLE_V5} materialization source linkage drifted")
    if manifest.get("common_tokenizer") != _common_tokenizer_identity():
        raise ValueError(f"{ROLE_V5} materialization does not bind the common SFT tokenizer")
    expected_files = {item["path"]: item for item in manifest.get("output_files", [])}
    actual_files = {item["path"]: item for item in _export_files(NOTOOL_MATERIALIZED)}
    if actual_files != expected_files:
        raise ValueError(f"{ROLE_V5} materialized output file identity drifted")
    inventory = _tensor_inventory(NOTOOL_MATERIALIZED)
    if inventory != manifest.get("tensor_inventory"):
        raise ValueError(f"{ROLE_V5} materialized tensor inventory drifted")
    return {
        "role": ROLE_V5,
        "state": "validated",
        "source_actor_sha256": source["source_actor_sha256"],
        "export_locator": str(NOTOOL_MATERIALIZED.resolve()),
        "materialization_manifest": str(NOTOOL_MANIFEST.relative_to(ROOT)),
        "materialization_manifest_sha256": sha256(NOTOOL_MANIFEST),
        "output_file_count": len(actual_files),
        "tensor_count": inventory["tensor_count"],
        "total_parameters": inventory["total_parameters"],
    }


def resolved_e3notool_role() -> dict[str, Any]:
    """Runtime binding entry with the same shape as the frozen v3/v4 checkpoint roles."""
    role = validate_e3notool_source_role()
    role["materialization"] = validate_e3notool_materialization()
    role["inference_locator"] = str(NOTOOL_MATERIALIZED.resolve())
    role["evaluator_tokenizer"] = _common_tokenizer_identity()
    return role


def resolved_reused_weight_role(notool_role: str) -> dict[str, Any]:
    """A notool role over frozen v3 weights: same identity dict as v3 plus the mode binding."""
    weight_role = NOTOOL_WEIGHT_ROLES[notool_role]
    if weight_role not in REUSED_WEIGHT_LOCATORS:
        raise ValueError(f"{notool_role} does not reuse a frozen weight role")
    identity = validate_source_role(weight_role)
    if weight_role in MATERIALIZED_PATHS:
        identity["materialization"] = validate_materialization(weight_role)
    identity["inference_locator"] = str(REUSED_WEIGHT_LOCATORS[weight_role].resolve())
    identity["evaluator_tokenizer"] = _common_tokenizer_identity()
    return {
        "role": notool_role,
        "state": "validated",
        "generation_mode": GENERATION_MODE_NOTOOL,
        "weight_role": weight_role,
        "weight_identity": identity,
        "inference_locator": identity["inference_locator"],
        "evaluator_tokenizer": identity["evaluator_tokenizer"],
    }


def resolved_notool_roles() -> dict[str, dict[str, Any]]:
    """All four notool roles in plan §3 order; every locator explicit and hash validated."""
    roles: dict[str, dict[str, Any]] = {}
    for notool_role, weight_role in NOTOOL_WEIGHT_ROLES.items():
        roles[notool_role] = resolved_e3notool_role() if weight_role == ROLE_V5 else resolved_reused_weight_role(notool_role)
    return roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-source", "materialize", "validate-materialization", "resolve-roles"))
    args = parser.parse_args()
    if args.command == "validate-source":
        result: Any = validate_e3notool_source_role()
    elif args.command == "materialize":
        result = materialize_e3notool()
    elif args.command == "validate-materialization":
        result = validate_e3notool_materialization()
    else:
        result = resolved_notool_roles()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
