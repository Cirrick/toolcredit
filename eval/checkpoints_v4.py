"""M8 / protocol v4: the single new checkpoint role ``e5v2_conserved_credit_step_200``.

The frozen v3 module ``eval/checkpoints.py`` is hashed by the M7 freeze closure and is not
modified.  This module reuses its helpers for the official veRL FSDP merge, tensor
inventory, export inventory and common-tokenizer identity, and adds only the E5-v2
role's fail-closed source validation, materialization and runtime binding.
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

ROLE_V4 = "e5v2_conserved_credit_step_200"
E5V2_RUN = ROOT / "rl/runs/e5v2_conserved_credit_20260905_003041"
E5V2_SOURCE = E5V2_RUN / "checkpoints/global_step_200"
E5V2_MATERIALIZED = MATERIALIZED_ROOT / "m8_e5v2_conserved_credit_step_200_hf"
E5V2_MANIFEST = MATERIALIZATION_LEDGER_ROOT / f"{ROLE_V4}.json"
E5V2_COMPLETION_GATE = E5V2_RUN / "analysis/formal_completion_gate.json"
REQUIRED_CHECKPOINT_FILES = (
    "actor/model_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "actor/fsdp_config.json",
    "data.pt",
)


def validate_e5v2_source_role(locator: Path | None = None) -> dict[str, Any]:
    """Explicit, completed, gate-validated E5-v2 step-200 source; null/glob/swap fail closed."""
    path = E5V2_SOURCE if locator is None else Path(locator)
    if path.resolve() != E5V2_SOURCE.resolve() or path.name != "global_step_200":
        raise ValueError(f"{ROLE_V4} locator is not its explicit canonical step-200 path: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{ROLE_V4} locator does not exist: {path}")
    _require_files(path, REQUIRED_CHECKPOINT_FILES, ROLE_V4)
    sizes = {relative: (path / relative).stat().st_size for relative in REQUIRED_CHECKPOINT_FILES}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError(f"{ROLE_V4} checkpoint has an empty required file")
    run = path.parents[1]
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    tracker = (path.parent / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip()
    if status.get("state") != "completed" or tracker != "200":
        raise ValueError(f"{ROLE_V4} status/tracker is not completed step 200")
    config_path = run / "resolved_config.yaml"
    config = _load_yaml(config_path)
    if Path(str(_nested(config, "actor_rollout_ref.model.path"))).resolve() != SFT.resolve():
        raise ValueError(f"{ROLE_V4} did not start from the frozen SFT checkpoint")
    if int(_nested(config, "trainer.total_training_steps")) != 200:
        raise ValueError(f"{ROLE_V4} is not a 200-step run")
    if _nested(config, "turn_credit.variant") != "conserved_v2":
        raise ValueError(f"{ROLE_V4} resolved config is not the conserved_v2 variant")
    if float(_nested(config, "turn_credit.beta")) != 0.5:
        raise ValueError(f"{ROLE_V4} beta drifted from the frozen 0.5")
    if _nested(config, "turn_credit.adoption_version") != "toolcredit_adoption_v2":
        raise ValueError(f"{ROLE_V4} adoption version drifted")
    if _nested(config, "actor_rollout_ref.rollout.agent.default_agent_loop") != "toolcredit_turn_credit_v2_agent":
        raise ValueError(f"{ROLE_V4} did not use the v2 agent loop")
    wrapper_path = run / "wrapper_installation.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    if wrapper.get("boundary_version") != "e5v2_runtime_wrapper_v1" or wrapper.get("variant") != "conserved_v2":
        raise ValueError(f"{ROLE_V4} wrapper proof is not the E5-v2 boundary")
    if wrapper.get("installed") is not True or wrapper.get("restored") is not True:
        raise ValueError(f"{ROLE_V4} wrapper proof is not installed/restored")
    gate_path = run / "analysis/formal_completion_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"{ROLE_V4} formal completion gate has not been run: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "completed" or gate.get("run_name") != run.name:
        raise ValueError(f"{ROLE_V4} formal completion gate did not pass for this run")
    if int(gate.get("rollouts", {}).get("trajectory_count", 0)) != 200 * 512:
        raise ValueError(f"{ROLE_V4} formal completion gate does not cover 200 x 512 trajectories")
    if gate.get("wrapper_proof_id") != wrapper.get("proof_id"):
        raise ValueError(f"{ROLE_V4} completion gate proof differs from the wrapper installation")
    return {
        "role": ROLE_V4,
        "state": "validated",
        "source_locator": str(path.resolve()),
        "run_name": run.name,
        "required_file_bytes": sizes,
        "source_actor_sha256": sha256(path / "actor/model_world_size_1_rank_0.pt"),
        "resolved_config_sha256": sha256(config_path),
        "status_sha256": sha256(run / "status.json"),
        "tracker_sha256": sha256(path.parent / "latest_checkpointed_iteration.txt"),
        "wrapper_installation_sha256": sha256(wrapper_path),
        "formal_completion_gate_sha256": sha256(gate_path),
        "turn_credit": {
            "variant": "conserved_v2",
            "beta": 0.5,
            "adoption_version": "toolcredit_adoption_v2",
            "ledger_schema_version": str(_nested(config, "turn_credit.ledger_schema_version")),
        },
    }


def materialize_e5v2(minimum_free_bytes: int = 40 * 1024**3) -> dict[str, Any]:
    """Run the pinned official merger once; identical command shape to the frozen M7 roles."""
    source_summary = validate_e5v2_source_role()
    if E5V2_MATERIALIZED.exists() or E5V2_MANIFEST.exists():
        raise FileExistsError(f"refusing to overwrite existing materialization for {ROLE_V4}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(f"M8 materialization disk gate failed: {free_bytes} < {minimum_free_bytes}")
    E5V2_MATERIALIZED.parent.mkdir(parents=True, exist_ok=True)
    E5V2_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str((E5V2_SOURCE / "actor").resolve()),
        "--target_dir",
        str(E5V2_MATERIALIZED.resolve()),
    ]
    started_at = utc_now()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    completed_at = utc_now()
    if process.returncode != 0:
        raise RuntimeError(
            f"official veRL FSDP merge failed for {ROLE_V4} with code {process.returncode}: {process.stderr[-4000:]}"
        )
    if not E5V2_MATERIALIZED.is_dir():
        raise RuntimeError("official merger returned success without creating target directory")
    (E5V2_MATERIALIZED / "materialization.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    import verl

    manifest = {
        "schema_version": "toolcredit_m7_materialization_manifest_v1",
        "role": ROLE_V4,
        "source": source_summary,
        "target_locator": str(E5V2_MATERIALIZED.resolve()),
        "started_at": started_at,
        "completed_at": completed_at,
        "command": command,
        "return_code": process.returncode,
        "official_fsdp_merge_validation": "passed_exit_zero",
        "verl_version": getattr(verl, "__version__", "0.8.0"),
        "merger_source_path": str(_merger_source()),
        "merger_source_sha256": sha256(_merger_source()),
        "output_files": _export_files(E5V2_MATERIALIZED),
        "tensor_inventory": _tensor_inventory(E5V2_MATERIALIZED),
        "common_tokenizer": _common_tokenizer_identity(),
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (E5V2_MATERIALIZED / "materialization_manifest.json").write_text(payload, encoding="utf-8")
    E5V2_MANIFEST.write_text(payload, encoding="utf-8")
    return manifest


def validate_e5v2_materialization() -> dict[str, Any]:
    source = validate_e5v2_source_role()
    internal = E5V2_MATERIALIZED / "materialization_manifest.json"
    _require_files(E5V2_MATERIALIZED, ("materialization_manifest.json", "config.json"), ROLE_V4)
    if not E5V2_MANIFEST.is_file() or E5V2_MANIFEST.read_bytes() != internal.read_bytes():
        raise ValueError(f"{ROLE_V4} internal/external materialization ledgers differ")
    manifest = json.loads(E5V2_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("role") != ROLE_V4 or Path(manifest.get("target_locator", "")).resolve() != E5V2_MATERIALIZED.resolve():
        raise ValueError(f"{ROLE_V4} materialization manifest role/target mismatch")
    if manifest.get("source", {}).get("source_actor_sha256") != source["source_actor_sha256"]:
        raise ValueError(f"{ROLE_V4} materialization source linkage drifted")
    if manifest.get("common_tokenizer") != _common_tokenizer_identity():
        raise ValueError(f"{ROLE_V4} materialization does not bind the common SFT tokenizer")
    expected_files = {item["path"]: item for item in manifest.get("output_files", [])}
    actual_files = {item["path"]: item for item in _export_files(E5V2_MATERIALIZED)}
    if actual_files != expected_files:
        raise ValueError(f"{ROLE_V4} materialized output file identity drifted")
    inventory = _tensor_inventory(E5V2_MATERIALIZED)
    if inventory != manifest.get("tensor_inventory"):
        raise ValueError(f"{ROLE_V4} materialized tensor inventory drifted")
    return {
        "role": ROLE_V4,
        "state": "validated",
        "source_actor_sha256": source["source_actor_sha256"],
        "export_locator": str(E5V2_MATERIALIZED.resolve()),
        "materialization_manifest": str(E5V2_MANIFEST.relative_to(ROOT)),
        "materialization_manifest_sha256": sha256(E5V2_MANIFEST),
        "output_file_count": len(actual_files),
        "tensor_count": inventory["tensor_count"],
        "total_parameters": inventory["total_parameters"],
    }


def resolved_e5v2_role() -> dict[str, Any]:
    """Runtime binding entry with the same shape as the frozen v3 checkpoint roles."""
    role = validate_e5v2_source_role()
    role["materialization"] = validate_e5v2_materialization()
    role["inference_locator"] = str(E5V2_MATERIALIZED.resolve())
    role["evaluator_tokenizer"] = _common_tokenizer_identity()
    return role


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-source", "materialize", "validate-materialization"))
    args = parser.parse_args()
    if args.command == "validate-source":
        result = validate_e5v2_source_role()
    elif args.command == "materialize":
        result = materialize_e5v2()
    else:
        result = validate_e5v2_materialization()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
