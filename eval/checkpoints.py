"""Fail-closed M7 checkpoint roles and official veRL FSDP materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/home/jovyan/Qwen/Qwen3-1.7B")
SFT = ROOT / "sft/checkpoints/qwen3-1.7b-sft"
E3_RUN = ROOT / "rl/runs/e3_grpo_baseline_20260819_224555"
E5_RUN = ROOT / "rl/runs/e5_turn_credit_20260823_082012"
E3_SOURCE = E3_RUN / "checkpoints/global_step_200"
E5_SOURCE = E5_RUN / "checkpoints/global_step_200"
M6_COMPLETION = E5_RUN / "analysis/m6_e5_completion_manifest.json"
V2_MANIFEST = ROOT / "eval/diagnostic_freeze_v2.sha256"
COMMON_TOKENIZER = SFT
MATERIALIZED_ROOT = ROOT / "eval/checkpoints"
MATERIALIZATION_LEDGER_ROOT = ROOT / "eval/materialization"

ROLE_ORDER = (
    "raw_base_model",
    "sft_start",
    "e3_grpo_baseline_step_200",
    "e5_turn_credit_step_200",
)
ROLE_SOURCE_PATHS = {
    "raw_base_model": RAW,
    "sft_start": SFT,
    "e3_grpo_baseline_step_200": E3_SOURCE,
    "e5_turn_credit_step_200": E5_SOURCE,
}
MATERIALIZED_PATHS = {
    "e3_grpo_baseline_step_200": MATERIALIZED_ROOT / "m7_e3_grpo_baseline_step_200_hf",
    "e5_turn_credit_step_200": MATERIALIZED_ROOT / "m7_e5_turn_credit_step_200_hf",
}
MATERIALIZATION_MANIFESTS = {
    role: MATERIALIZATION_LEDGER_ROOT / f"{role}.json" for role in MATERIALIZED_PATHS
}
SOURCE_ACTOR_SHA256 = {
    "e3_grpo_baseline_step_200": "8b7cd75fda9754194686e6a617debee81c66920aa8bae34dffd6b39017b32ff4",
    "e5_turn_credit_step_200": "0841645cc7ad39e25e2f142de64c29b0c791248457e53163b93fdefa98cfa471",
}
SOURCE_REQUIRED_BYTES = {
    "e3_grpo_baseline_step_200": {
        "actor/model_world_size_1_rank_0.pt": 8127118819,
        "actor/optim_world_size_1_rank_0.pt": 13764626512,
        "actor/extra_state_world_size_1_rank_0.pt": 15205,
        "actor/fsdp_config.json": 46,
        "data.pt": 6868,
    },
    "e5_turn_credit_step_200": {
        "actor/model_world_size_1_rank_0.pt": 8127118819,
        "actor/optim_world_size_1_rank_0.pt": 13764626512,
        "actor/extra_state_world_size_1_rank_0.pt": 15141,
        "actor/fsdp_config.json": 46,
        "data.pt": 6868,
    },
}
FROZEN_HASHES = {
    "raw_config": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
    "raw_index": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
    "sft_model": "cb5f43dafad692b2a3dc37ad6bc22f86486ada7682739036a22b6d6117bcb235",
    "sft_chat_template": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "v2_manifest": "6fa185e18c17761cf9c238cb2a9754e245cda8179136e59cb69c729c9b07f3f0",
    "e3_resolved_config": "254a3188aa85d062b1fc690190729f5f96ad35d3494e7f218649c9708a0e9c16",
    "e5_resolved_config": "3bf617027df30705026648a176dc31e4be7e18f7151b6a004e0917d6ab92f5e2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_files(base: Path, relative_files: Iterable[str], role: str) -> None:
    missing = [relative for relative in relative_files if not (base / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"{role} is incomplete at {base}: missing {missing}")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping at {path}")
    return value


def _nested(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"missing required config key {dotted}")
        value = value[part]
    return value


def _file_inventory(base: Path, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    paths = sorted(base.rglob("*")) if names is None else [base / name for name in names]
    return [
        {"path": str(path.relative_to(base)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
        if path.is_file() and path.name != "materialization_manifest.json"
    ]


def _validate_raw(path: Path) -> dict[str, Any]:
    if path.resolve() != RAW.resolve():
        raise ValueError(f"raw role locator drifted: {path}")
    _require_files(path, ("config.json", "model.safetensors.index.json", "tokenizer.json"), "raw")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("raw role is not Qwen3")
    if sha256(path / "config.json") != FROZEN_HASHES["raw_config"]:
        raise ValueError("raw config hash drifted from frozen v2")
    if sha256(path / "model.safetensors.index.json") != FROZEN_HASHES["raw_index"]:
        raise ValueError("raw model index hash drifted from frozen v2")
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError("raw model index has no weight shards")
    _require_files(path, shard_names, "raw")
    return {"source_files": _file_inventory(path, ["config.json", "model.safetensors.index.json", *shard_names])}


def _validate_sft(path: Path) -> dict[str, Any]:
    if path.resolve() != SFT.resolve():
        raise ValueError(f"SFT role locator drifted: {path}")
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    _require_files(path, required, "SFT")
    if sha256(path / "model.safetensors") != FROZEN_HASHES["sft_model"]:
        raise ValueError("SFT model hash drifted from frozen v2")
    if sha256(path / "chat_template.jinja") != FROZEN_HASHES["sft_chat_template"]:
        raise ValueError("SFT chat template hash drifted from frozen v2")
    return {"source_files": _file_inventory(path, required)}


def _validate_training_role(role: str, path: Path) -> dict[str, Any]:
    expected = ROLE_SOURCE_PATHS[role]
    if path.resolve() != expected.resolve() or path.name != "global_step_200":
        raise ValueError(f"{role} locator is not its explicit canonical step-200 path: {path}")
    required = SOURCE_REQUIRED_BYTES[role]
    _require_files(path, required, role)
    for relative, expected_bytes in required.items():
        actual = (path / relative).stat().st_size
        if actual != expected_bytes:
            raise ValueError(f"{role} size drift for {relative}: {actual} != {expected_bytes}")
    run = path.parents[1]
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    tracker = (path.parent / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip()
    if status.get("state") != "completed" or tracker != "200":
        raise ValueError(f"{role} status/tracker is not completed step 200")
    config_path = run / "resolved_config.yaml"
    config = _load_yaml(config_path)
    model_path = Path(str(_nested(config, "actor_rollout_ref.model.path"))).resolve()
    if model_path != SFT.resolve() or int(_nested(config, "trainer.total_training_steps")) != 200:
        raise ValueError(f"{role} is not a 200-step run from the frozen SFT start")
    expected_config_hash = FROZEN_HASHES["e3_resolved_config" if role.startswith("e3_") else "e5_resolved_config"]
    if sha256(config_path) != expected_config_hash:
        raise ValueError(f"{role} resolved config identity drifted")
    actor = path / "actor/model_world_size_1_rank_0.pt"
    if sha256(actor) != SOURCE_ACTOR_SHA256[role]:
        raise ValueError(f"{role} source actor hash drifted")
    result: dict[str, Any] = {
        "run_name": run.name,
        "source_actor_sha256": SOURCE_ACTOR_SHA256[role],
        "resolved_config_sha256": expected_config_hash,
        "status_sha256": sha256(run / "status.json"),
        "tracker_sha256": sha256(path.parent / "latest_checkpointed_iteration.txt"),
    }
    if role.startswith("e5_"):
        completion = json.loads(M6_COMPLETION.read_text(encoding="utf-8"))
        wrapper = json.loads((run / "wrapper_installation.json").read_text(encoding="utf-8"))
        if completion.get("role") != role or completion.get("status") != "completed" or completion.get("tracker_step") != 200:
            raise ValueError("E5 M6 completion manifest does not bind the canonical role")
        if completion.get("resolved_config_sha256") != expected_config_hash:
            raise ValueError("E5 completion manifest/config mismatch")
        if completion.get("freeze", {}).get("manifest_sha256") != FROZEN_HASHES["v2_manifest"]:
            raise ValueError("E5 was not trained against the frozen v2 manifest")
        if sha256(V2_MANIFEST) != FROZEN_HASHES["v2_manifest"]:
            raise ValueError("frozen v2 manifest drifted")
        if float(_nested(config, "turn_credit.beta")) != 0.5 or _nested(config, "turn_credit.heuristic_version") != "toolcredit_adoption_v1":
            raise ValueError("E5 beta/adoption heuristic drifted")
        if wrapper.get("installed") is not True or wrapper.get("restored") is not True:
            raise ValueError("E5 wrapper proof is not installed/restored")
        if completion.get("wrapper_installation_sha256") != sha256(run / "wrapper_installation.json"):
            raise ValueError("E5 wrapper proof hash differs from completion manifest")
        result.update(
            {
                "m6_completion_manifest_sha256": sha256(M6_COMPLETION),
                "wrapper_installation_sha256": sha256(run / "wrapper_installation.json"),
                "frozen_v2_manifest_sha256": sha256(V2_MANIFEST),
            }
        )
    return result


def validate_source_role(role: str, locator: Path | None = None) -> dict[str, Any]:
    """Validate one explicit source role; null and role-swapped locators fail closed."""
    if role not in ROLE_ORDER:
        raise ValueError(f"unknown M7 checkpoint role: {role}")
    path = ROLE_SOURCE_PATHS[role] if locator is None else Path(locator)
    if not path.is_dir():
        raise FileNotFoundError(f"{role} locator does not exist: {path}")
    if role == "raw_base_model":
        details = _validate_raw(path)
    elif role == "sft_start":
        details = _validate_sft(path)
    else:
        details = _validate_training_role(role, path)
    return {"role": role, "state": "validated", "source_locator": str(path.resolve()), **details}


def validate_all_source_roles() -> dict[str, Any]:
    return {role: validate_source_role(role) for role in ROLE_ORDER}


def validate_role_bindings(bindings: dict[str, str | None]) -> None:
    """Reject null, implicit, swapped, and non-explicit role locators before I/O."""
    if tuple(bindings) != ROLE_ORDER:
        raise ValueError(f"checkpoint bindings must use the frozen role order {ROLE_ORDER}")
    for role, expected in ROLE_SOURCE_PATHS.items():
        locator = bindings.get(role)
        if not isinstance(locator, str) or not locator.strip():
            raise ValueError(f"M7 checkpoint role {role} cannot be null or implicit")
        if any(marker in locator for marker in ("*", "?", "[", "]")) or "latest" in Path(locator).name:
            raise ValueError(f"M7 checkpoint role {role} must not use glob/latest resolution")
        actual = Path(locator)
        actual = actual if actual.is_absolute() else ROOT / actual
        if actual.resolve() != expected.resolve():
            raise ValueError(f"M7 checkpoint role {role} is swapped or drifted: {actual} != {expected}")


def _merger_source() -> Path:
    import verl.model_merger

    return Path(verl.model_merger.__file__).resolve()


def _tensor_inventory(model_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open

    shard_paths = sorted(model_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"materialized model has no safetensors shards: {model_dir}")
    tensors: list[dict[str, Any]] = []
    total_parameters = 0
    seen: set[str] = set()
    for shard in shard_paths:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"duplicate tensor key across materialized shards: {key}")
                seen.add(key)
                tensor = handle.get_slice(key)
                shape = list(tensor.get_shape())
                count = 1
                for dimension in shape:
                    count *= int(dimension)
                total_parameters += count
                tensors.append({"key": key, "shape": shape, "dtype": str(tensor.get_dtype()), "numel": count})
    return {"tensor_count": len(tensors), "total_parameters": total_parameters, "tensors": tensors}


def _export_files(model_dir: Path) -> list[dict[str, Any]]:
    names = [
        path for path in sorted(model_dir.iterdir())
        if path.is_file() and path.name != "materialization_manifest.json"
    ]
    required = {"config.json"}
    if not required.issubset({path.name for path in names}) or not any(path.suffix == ".safetensors" for path in names):
        raise FileNotFoundError(f"materialized export is not a complete HF model: {model_dir}")
    return [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in names
    ]


def _common_tokenizer_identity() -> dict[str, Any]:
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    _require_files(COMMON_TOKENIZER, names, "common M7 tokenizer")
    return {
        "evaluator_tokenizer_path": str(COMMON_TOKENIZER.resolve()),
        "files": _file_inventory(COMMON_TOKENIZER, names),
    }


def materialize(role: str, minimum_free_bytes: int = 65 * 1024**3) -> dict[str, Any]:
    """Run the pinned official merger once and create a source-to-export identity ledger."""
    if role not in MATERIALIZED_PATHS:
        raise ValueError("only E3/E5 FSDP actor roles require materialization")
    source_summary = validate_source_role(role)
    target = MATERIALIZED_PATHS[role]
    manifest_path = MATERIALIZATION_MANIFESTS[role]
    if target.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing M7 materialization for {role}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(f"M7 materialization disk gate failed: {free_bytes} < {minimum_free_bytes}")
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
        str((ROLE_SOURCE_PATHS[role] / "actor").resolve()),
        "--target_dir",
        str(target.resolve()),
    ]
    started_at = utc_now()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    completed_at = utc_now()
    if process.returncode != 0:
        raise RuntimeError(
            f"official veRL FSDP merge failed for {role} with code {process.returncode}: "
            f"{process.stderr[-4000:]}"
        )
    if not target.is_dir():
        raise RuntimeError("official merger returned success without creating target directory")
    merger_log = target / "materialization.log"
    merger_log.write_text(process.stdout + process.stderr, encoding="utf-8")
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


def validate_materialization(role: str) -> dict[str, Any]:
    if role not in MATERIALIZED_PATHS:
        raise ValueError(f"role is not materialized: {role}")
    validate_source_role(role)
    target = MATERIALIZED_PATHS[role]
    external = MATERIALIZATION_MANIFESTS[role]
    internal = target / "materialization_manifest.json"
    _require_files(target, ("materialization_manifest.json", "config.json"), role)
    if not external.is_file() or external.read_bytes() != internal.read_bytes():
        raise ValueError(f"{role} internal/external materialization ledgers differ")
    manifest = json.loads(external.read_text(encoding="utf-8"))
    if manifest.get("role") != role or Path(manifest.get("target_locator", "")).resolve() != target.resolve():
        raise ValueError(f"{role} materialization manifest role/target mismatch")
    if manifest.get("source", {}).get("source_actor_sha256") != SOURCE_ACTOR_SHA256[role]:
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
        "source_actor_sha256": SOURCE_ACTOR_SHA256[role],
        "export_locator": str(target.resolve()),
        "materialization_manifest": str(external.relative_to(ROOT)),
        "materialization_manifest_sha256": sha256(external),
        "output_file_count": len(actual_files),
        "tensor_count": inventory["tensor_count"],
        "total_parameters": inventory["total_parameters"],
    }


def resolved_checkpoint_roles(require_materialized: bool = True) -> dict[str, Any]:
    roles = validate_all_source_roles()
    for role in MATERIALIZED_PATHS:
        if require_materialized:
            roles[role]["materialization"] = validate_materialization(role)
    roles["raw_base_model"]["inference_locator"] = str(RAW.resolve())
    roles["sft_start"]["inference_locator"] = str(SFT.resolve())
    for role, target in MATERIALIZED_PATHS.items():
        roles[role]["inference_locator"] = str(target.resolve())
    common = _common_tokenizer_identity()
    for role in roles.values():
        role["evaluator_tokenizer"] = common
    return roles


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-sources")
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--role", required=True, choices=tuple(MATERIALIZED_PATHS))
    validate_parser = subparsers.add_parser("validate-materialization")
    validate_parser.add_argument("--role", choices=tuple(MATERIALIZED_PATHS))
    args = parser.parse_args()
    if args.command == "validate-sources":
        result = validate_all_source_roles()
    elif args.command == "materialize":
        result = materialize(args.role)
    elif args.role:
        result = validate_materialization(args.role)
    else:
        result = {role: validate_materialization(role) for role in MATERIALIZED_PATHS}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
