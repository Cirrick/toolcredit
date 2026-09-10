"""M10 / protocol v6: the two E7 dynamic-filtering roles and their materialization.

The frozen v3 module ``eval/checkpoints.py`` and the v4/v5 modules are hashed by the M7/M8/M9
freeze closures and are not modified.  v6 adds exactly two roles (plans/M10.md §5), both TIR:

* ``e7_dynamic_filter_step_200`` — the end of the E7 run, compared with E3 at equal *effective
  step*;
* ``e7_equal_compute`` — the extra checkpoint E7 saves the first time its cumulative rollout
  trajectories reach the E3 total of 102,400 (plan §3.4), compared with E3 step-200 at equal
  *rollout compute*.  Its locator is ``checkpoints/equal_compute_step_<N>/``, where ``N`` is
  discovered from disk and cross-checked against the run ledger and the completion gate — the
  plan deliberately does not fix ``N`` in advance.

Both roles come from one run and one training recipe, so every check is parameterised by role
rather than duplicated.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.checkpoints import (
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

ROLE_STEP_200 = "e7_dynamic_filter_step_200"
ROLE_EQUAL_COMPUTE = "e7_equal_compute"
E7_ROLE_NAMES = (ROLE_STEP_200, ROLE_EQUAL_COMPUTE)
GENERATION_MODE_TIR = "tir"

E7_RUN = ROOT / "rl/runs/e7_dynamic_filtering_20260909_200152"
E7_CHECKPOINTS = E7_RUN / "checkpoints"
E7_COMPLETION_GATE = E7_RUN / "analysis/formal_completion_gate.json"
E7_LEDGER = E7_RUN / "e7_ledger.jsonl"
EQUAL_COMPUTE_DIR_RE = re.compile(r"^equal_compute_step_(\d+)$")

MATERIALIZED_PATHS_V6 = {
    ROLE_STEP_200: MATERIALIZED_ROOT / "m10_e7_dynamic_filter_step_200_hf",
    ROLE_EQUAL_COMPUTE: MATERIALIZED_ROOT / "m10_e7_equal_compute_hf",
}
MANIFEST_PATHS_V6 = {role: MATERIALIZATION_LEDGER_ROOT / f"{role}.json" for role in E7_ROLE_NAMES}

REQUIRED_CHECKPOINT_FILES = (
    "actor/model_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "actor/fsdp_config.json",
    "data.pt",
    "e7_ledger.json",
)
# Plan §3.2: the complete set of resolved-config paths E3 -> E7 may differ on.  Kept as a literal
# so the evaluator does not import the launcher; the test asserts equality with
# ``rl.launch.e7_dynamic_filtering.E3_DIFF_PATHS``.
E3_CONTROL_DIFF_PATHS = (
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.actor.fsdp_config.param_offload",
    "actor_rollout_ref.ref.fsdp_config.param_offload",
    "actor_rollout_ref.rollout.trace.project_name",
    "algorithm.filter_groups",
    "trainer.project_name",
    "trainer.toolcredit_trainer_class",
)
TRAINER_CLASS_FQN = "rl.custom.dynamic_filter_trainer.DynamicFilterRayPPOTrainer"
FILTER_GROUPS = {"enable": True, "metric": "score", "max_num_gen_batches": 4}
E3_TOTAL_TRAJECTORIES = 200 * 64 * 8
FORMAL_STEPS = 200
TRAJECTORIES_PER_UPDATE = 64 * 8


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def equal_compute_step() -> int:
    """The single ``equal_compute_step_<N>`` directory on disk, cross-checked with the run ledger."""
    if not E7_CHECKPOINTS.is_dir():
        raise FileNotFoundError(f"E7 checkpoint root does not exist: {E7_CHECKPOINTS}")
    found: list[tuple[int, Path]] = []
    for path in sorted(E7_CHECKPOINTS.iterdir()):
        match = EQUAL_COMPUTE_DIR_RE.match(path.name)
        if path.is_dir() and match:
            found.append((int(match.group(1)), path))
    if len(found) != 1:
        raise ValueError(f"expected exactly one equal-compute checkpoint, found {[p.name for _, p in found]}")
    step, _ = found[0]
    rows = _ledger_rows(E7_LEDGER)
    flagged = [int(row["effective_step"]) for row in rows if row.get("equal_compute_checkpoint")]
    if flagged != [step]:
        raise ValueError(f"run ledger flags {flagged} for the equal-compute step, disk says {step}")
    row = next(row for row in rows if int(row["effective_step"]) == step)
    if int(row["cumulative_trajectories"]) < E3_TOTAL_TRAJECTORIES:
        raise ValueError(f"equal-compute step {step} is below the E3 total of {E3_TOTAL_TRAJECTORIES}")
    previous = [r for r in rows if int(r["effective_step"]) == step - 1]
    if previous and int(previous[0]["cumulative_trajectories"]) >= E3_TOTAL_TRAJECTORIES:
        raise ValueError(f"equal-compute step {step} is not the first step reaching the E3 total")
    return step


def source_locator(role: str) -> Path:
    if role == ROLE_STEP_200:
        return E7_CHECKPOINTS / f"global_step_{FORMAL_STEPS}"
    if role == ROLE_EQUAL_COMPUTE:
        return E7_CHECKPOINTS / f"equal_compute_step_{equal_compute_step()}"
    raise ValueError(f"unknown v6 role: {role!r}")


def _validate_training_config(role: str, run: Path) -> Path:
    """The E7 treatment and every E3 control, read from the run's own resolved config."""
    config_path = run / "resolved_config.yaml"
    config = _load_yaml(config_path)
    if Path(str(_nested(config, "actor_rollout_ref.model.path"))).resolve() != SFT.resolve():
        raise ValueError(f"{role} did not start from the frozen SFT checkpoint")
    if int(_nested(config, "trainer.total_training_steps")) != FORMAL_STEPS:
        raise ValueError(f"{role} is not a {FORMAL_STEPS}-effective-step run")
    if _nested(config, "trainer.toolcredit_trainer_class") != TRAINER_CLASS_FQN:
        raise ValueError(f"{role} was not trained by the dynamic-filter trainer")
    observed = {key: _nested(config, f"algorithm.filter_groups.{key}") for key in FILTER_GROUPS}
    if observed != FILTER_GROUPS:
        raise ValueError(f"{role} filter_groups drifted: {observed}")
    if _nested(config, "algorithm.adv_estimator") != "grpo":
        raise ValueError(f"{role} is not a GRPO run")
    if _nested(config, "algorithm.norm_adv_by_std_in_grpo") is not True:
        raise ValueError(f"{role} dropped GRPO std normalisation")
    for path in (
        "actor_rollout_ref.actor.fsdp_config.param_offload",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
        "actor_rollout_ref.ref.fsdp_config.param_offload",
    ):
        if _nested(config, path) is not False:
            raise ValueError(f"{role} resolved config has {path} enabled; E7 runs with offload off")
    if _nested(config, "actor_rollout_ref.rollout.multi_turn.enable") is not True:
        raise ValueError(f"{role} is a TIR arm and must keep the multi-turn tool loop")
    if int(_nested(config, "actor_rollout_ref.rollout.n")) != 8:
        raise ValueError(f"{role} rollout n is not 8")
    if int(_nested(config, "data.train_batch_size")) != 64:
        raise ValueError(f"{role} prompt batch is not 64")
    return config_path


def _validate_completion_gate(role: str, run: Path, step: int) -> dict[str, Any]:
    """``rl/validate_e7_run.py formal`` must have passed for this run and cover this checkpoint."""
    if not E7_COMPLETION_GATE.is_file():
        raise FileNotFoundError(f"{role} formal completion gate has not been run: {E7_COMPLETION_GATE}")
    gate = json.loads(E7_COMPLETION_GATE.read_text(encoding="utf-8"))
    if gate.get("gate") != "formal_completion" or gate.get("run_name") != run.name:
        raise ValueError(f"{role} formal completion gate does not belong to this run")
    if gate.get("status") != "completed" or int(gate.get("stop_step", 0)) != FORMAL_STEPS:
        raise ValueError(f"{role} formal completion gate did not cover {FORMAL_STEPS} effective steps")
    ledger = gate.get("ledger", {})
    if int(ledger.get("steps", 0)) != FORMAL_STEPS:
        raise ValueError(f"{role} completion gate ledger does not span {FORMAL_STEPS} steps")
    if int(gate.get("candidates", {}).get("steps_checked", 0)) != FORMAL_STEPS:
        raise ValueError(f"{role} completion gate did not verify every step's candidates")
    if int(ledger.get("max_full_batch_streak", 0)) >= 10:
        raise ValueError(f"{role} run hit the plan §4 four-batch streak gate")
    equal = gate.get("equal_compute", {})
    if equal.get("present") is not True:
        raise ValueError(f"{role} completion gate has no equal-compute checkpoint")
    if int(equal.get("step", 0)) != int(ledger.get("equal_compute_step", -1)):
        raise ValueError(f"{role} completion gate equal-compute step disagrees with its ledger")
    if role == ROLE_EQUAL_COMPUTE and int(equal.get("step", 0)) != step:
        raise ValueError(f"{role} locator step {step} != gate equal-compute step {equal.get('step')}")
    if role == ROLE_STEP_200 and int(gate.get("final_checkpoint", {}).get("step", 0)) != FORMAL_STEPS:
        raise ValueError(f"{role} completion gate does not bind the step-{FORMAL_STEPS} checkpoint")
    return gate


def validate_e7_source_role(role: str, locator: Path | None = None) -> dict[str, Any]:
    """Explicit, completed, gate-validated E7 source checkpoint; null/glob/swap fail closed."""
    if role not in E7_ROLE_NAMES:
        raise ValueError(f"unknown v6 role: {role!r}")
    expected = source_locator(role)
    path = expected if locator is None else Path(locator)
    if path.resolve() != expected.resolve():
        raise ValueError(f"{role} locator is not its explicit canonical path: {path} != {expected}")
    if not path.is_dir():
        raise FileNotFoundError(f"{role} locator does not exist: {path}")
    _require_files(path, REQUIRED_CHECKPOINT_FILES, role)
    sizes = {relative: (path / relative).stat().st_size for relative in REQUIRED_CHECKPOINT_FILES}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError(f"{role} checkpoint has an empty required file")
    run = path.parents[1]
    if run.resolve() != E7_RUN.resolve():
        raise ValueError(f"{role} checkpoint does not belong to the canonical run: {run}")

    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"{role} run status is {status.get('state')!r}, not completed")
    tracker_path = path.parent / "latest_checkpointed_iteration.txt"
    tracker = tracker_path.read_text(encoding="utf-8").strip()
    if tracker != str(FORMAL_STEPS):
        raise ValueError(f"{role} tracker is at step {tracker}, not {FORMAL_STEPS}")

    checkpoint_ledger = json.loads((path / "e7_ledger.json").read_text(encoding="utf-8"))
    step = int(checkpoint_ledger["effective_step"])
    if role == ROLE_STEP_200 and step != FORMAL_STEPS:
        raise ValueError(f"{role} checkpoint ledger is at step {step}, not {FORMAL_STEPS}")
    if role == ROLE_EQUAL_COMPUTE:
        if step != equal_compute_step() or int(checkpoint_ledger.get("equal_compute_step", -1)) != step:
            raise ValueError(f"{role} checkpoint ledger does not identify the equal-compute step")
        if checkpoint_ledger.get("equal_compute_saved") is not True:
            raise ValueError(f"{role} checkpoint ledger is not marked as the equal-compute save")
    cumulative = int(checkpoint_ledger["cumulative_trajectories"])
    if role == ROLE_EQUAL_COMPUTE and not (
        E3_TOTAL_TRAJECTORIES <= cumulative < E3_TOTAL_TRAJECTORIES + TRAJECTORIES_PER_UPDATE * 4
    ):
        raise ValueError(f"{role} cumulative trajectories {cumulative} is not one update past {E3_TOTAL_TRAJECTORIES}")
    if role == ROLE_STEP_200 and cumulative < E3_TOTAL_TRAJECTORIES:
        raise ValueError(f"{role} spent fewer trajectories than E3: {cumulative}")

    config_path = _validate_training_config(role, run)
    gate = _validate_completion_gate(role, run, step)
    recovery_archives = sorted(p.name for p in (run / "recovery").glob("resume_from_*") if p.is_dir())
    summary: dict[str, Any] = {
        "role": role,
        "state": "validated",
        "generation_mode": GENERATION_MODE_TIR,
        "weight_role": role,
        "source_locator": str(path.resolve()),
        "run_name": run.name,
        "required_file_bytes": sizes,
        "source_actor_sha256": sha256(path / "actor/model_world_size_1_rank_0.pt"),
        "resolved_config_sha256": sha256(config_path),
        "status_sha256": sha256(run / "status.json"),
        "tracker_sha256": sha256(tracker_path),
        "formal_completion_gate_sha256": sha256(E7_COMPLETION_GATE),
        "checkpoint_ledger_sha256": sha256(path / "e7_ledger.json"),
        "run_ledger_sha256": sha256(E7_LEDGER),
        "dynamic_filtering": {
            "diff_paths": list(E3_CONTROL_DIFF_PATHS),
            "filter_groups": dict(FILTER_GROUPS),
            "trainer_class": TRAINER_CLASS_FQN,
            "effective_step": step,
            "cumulative_trajectories": cumulative,
            "equal_compute_step": int(gate["ledger"]["equal_compute_step"]),
            "e3_total_trajectories": E3_TOTAL_TRAJECTORIES,
            "mean_generation_batches": gate["ledger"]["mean_generation_batches"],
            "zero_std_frac_mean": gate["ledger"]["zero_std_frac_mean"],
        },
        "recovery_events": recovery_archives,
    }
    if recovery_archives:
        summary["recovery_ledger_sha256"] = {
            name: sha256(run / "recovery" / name / "recovery.json") for name in recovery_archives
        }
    return summary


def materialize_e7(role: str, minimum_free_bytes: int = 40 * 1024**3) -> dict[str, Any]:
    """Run the pinned official merger once for one role; same command shape as M7/M8/M9."""
    source_summary = validate_e7_source_role(role)
    target = MATERIALIZED_PATHS_V6[role]
    manifest_path = MANIFEST_PATHS_V6[role]
    if target.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing materialization for {role}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(f"M10 materialization disk gate failed: {free_bytes} < {minimum_free_bytes}")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str((Path(source_summary["source_locator"]) / "actor").resolve()),
        "--target_dir",
        str(target.resolve()),
    ]
    started_at = utc_now()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    completed_at = utc_now()
    if process.returncode != 0:
        raise RuntimeError(
            f"official veRL FSDP merge failed for {role} with code {process.returncode}: {process.stderr[-4000:]}"
        )
    if not target.is_dir():
        raise RuntimeError("official merger returned success without creating target directory")
    (target / "materialization.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    import verl

    manifest = {
        "schema_version": "toolcredit_m7_materialization_manifest_v1",
        "role": role,
        "source": source_summary,
        "target_locator": str(target.resolve()),
        "started_at": started_at,
        "completed_at": completed_at,
        "command": command,
        "return_code": process.returncode,
        "official_fsdp_merge_validation": "passed_exit_zero",
        "verl_version": getattr(verl, "__version__", "0.8.0"),
        "merger_source_path": str(_merger_source()),
        "merger_source_sha256": sha256(_merger_source()),
        "output_files": _export_files(target),
        "tensor_inventory": _tensor_inventory(target),
        "common_tokenizer": _common_tokenizer_identity(),
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (target / "materialization_manifest.json").write_text(payload, encoding="utf-8")
    manifest_path.write_text(payload, encoding="utf-8")
    return manifest


def validate_e7_materialization(role: str) -> dict[str, Any]:
    source = validate_e7_source_role(role)
    target = MATERIALIZED_PATHS_V6[role]
    manifest_path = MANIFEST_PATHS_V6[role]
    internal = target / "materialization_manifest.json"
    _require_files(target, ("materialization_manifest.json", "config.json"), role)
    if not manifest_path.is_file() or manifest_path.read_bytes() != internal.read_bytes():
        raise ValueError(f"{role} internal/external materialization ledgers differ")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("role") != role or Path(manifest.get("target_locator", "")).resolve() != target.resolve():
        raise ValueError(f"{role} materialization manifest role/target mismatch")
    if manifest.get("source", {}).get("source_actor_sha256") != source["source_actor_sha256"]:
        raise ValueError(f"{role} materialization source linkage drifted")
    if manifest.get("common_tokenizer") != _common_tokenizer_identity():
        raise ValueError(f"{role} materialization does not bind the common SFT tokenizer")
    expected_files = {item["path"]: item for item in manifest.get("output_files", [])}
    actual_files = {item["path"]: item for item in _export_files(target)}
    if actual_files != expected_files:
        raise ValueError(f"{role} materialized output file identity drifted")
    inventory = _tensor_inventory(target)
    if inventory != manifest.get("tensor_inventory"):
        raise ValueError(f"{role} materialized tensor inventory drifted")
    return {
        "role": role,
        "state": "validated",
        "source_actor_sha256": source["source_actor_sha256"],
        "export_locator": str(target.resolve()),
        "materialization_manifest": str(manifest_path.relative_to(ROOT)),
        "materialization_manifest_sha256": sha256(manifest_path),
        "output_file_count": len(actual_files),
        "tensor_count": inventory["tensor_count"],
        "total_parameters": inventory["total_parameters"],
    }


def resolved_e7_role(role: str) -> dict[str, Any]:
    """Runtime binding entry with the same shape as the frozen v3/v4/v5 checkpoint roles."""
    entry = validate_e7_source_role(role)
    entry["materialization"] = validate_e7_materialization(role)
    entry["inference_locator"] = str(MATERIALIZED_PATHS_V6[role].resolve())
    entry["evaluator_tokenizer"] = _common_tokenizer_identity()
    return entry


def resolved_e7_roles() -> dict[str, dict[str, Any]]:
    """Both E7 roles in plan §5 order; every locator explicit and hash validated."""
    return {role: resolved_e7_role(role) for role in E7_ROLE_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("equal-compute-step", "validate-source", "materialize", "validate-materialization", "resolve-roles"),
    )
    parser.add_argument("--role", choices=E7_ROLE_NAMES, default=None)
    args = parser.parse_args()
    if args.command == "equal-compute-step":
        result: Any = {"equal_compute_step": equal_compute_step(), "locator": str(source_locator(ROLE_EQUAL_COMPUTE))}
    elif args.command == "resolve-roles":
        result = resolved_e7_roles()
    else:
        if args.role is None:
            raise SystemExit(f"--role is required for {args.command}")
        if args.command == "validate-source":
            result = validate_e7_source_role(args.role)
        elif args.command == "materialize":
            result = materialize_e7(args.role)
        else:
            result = validate_e7_materialization(args.role)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
