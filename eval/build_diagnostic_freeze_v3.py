"""Build and verify M7's narrow v3 runtime/evaluator binding over frozen v2."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze import git_blob, write_json
from eval.build_diagnostic_freeze_v2 import verify as verify_v2
from eval.checkpoints import (
    COMMON_TOKENIZER,
    MATERIALIZATION_MANIFESTS,
    ROOT,
    resolved_checkpoint_roles,
    sha256,
)

V2_PROTOCOL = ROOT / "eval/diagnostic_protocol_v2.json"
V2_MANIFEST = ROOT / "eval/diagnostic_freeze_v2.sha256"
PROTOCOL_PATH = ROOT / "eval/diagnostic_protocol_v3.json"
MANIFEST_PATH = ROOT / "eval/diagnostic_freeze_v3.sha256"
LEDGER_PATH = ROOT / "plans/M7_DIAGNOSTIC_FREEZE_V3.md"
CONFIG_PATH = ROOT / "eval/m7_eval_config.json"
SCHEMA_PATH = ROOT / "eval/m7_trajectory_schema.json"

EVALUATOR_FILES = (
    "eval/checkpoints.py",
    "eval/generate.py",
    "eval/evaluate.py",
    "eval/metrics.py",
    "analysis/badcase_taxonomy.py",
    "analysis/paired_transitions.py",
    "analysis/plots.py",
    "scripts/m7/run_unified_eval.sh",
    "scripts/m7/watch_unified_eval.sh",
    "eval/m7_eval_config.json",
    "eval/m7_trajectory_schema.json",
    "eval/test_m7_agent_loop_identity.py",
    "eval/test_m7_protocol.py",
    "eval/test_m7_metrics.py",
    "analysis/test_m7_analysis.py",
    "rl/custom/tool_agent_loop.py",
    "rl/custom/turn_credit_agent_loop.py",
)


def _v2_entries() -> list[str]:
    entries: list[str] = []
    for line in V2_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64:
            raise ValueError(f"invalid parent v2 manifest line: {line}")
        entries.append(relative)
    if len(entries) != 37 or len(set(entries)) != 37:
        raise ValueError("v3 parent must be the exact 37-file frozen v2 closure")
    return entries


def _source_hashes() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in EVALUATOR_FILES}


def _runtime_dependencies() -> dict[str, Any]:
    import importlib.metadata

    import sglang.srt.sampling.sampling_params as sglang_sampling
    import verl.model_merger as verl_merger

    sampling_path = Path(sglang_sampling.__file__).resolve()
    merger_path = Path(verl_merger.__file__).resolve()
    versions = {
        "verl": importlib.metadata.version("verl"),
        "sglang": importlib.metadata.version("sglang"),
    }
    if versions != {"verl": "0.8.0", "sglang": "0.5.8"}:
        raise RuntimeError(f"M7 runtime dependency version drift: {versions}")
    return {
        "versions": versions,
        "verl_model_merger_source": str(merger_path),
        "verl_model_merger_sha256": sha256(merger_path),
        "sglang_sampling_params_source": str(sampling_path),
        "sglang_sampling_params_sha256": sha256(sampling_path),
        "sglang_seed_field": "sampling_seed",
    }


def _checkpoint_protocol(roles: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, summary in roles.items():
        entry = copy.deepcopy(summary)
        entry["state"] = "validated"
        entry["role"] = role
        result[role] = entry
    return result


def protocol_v3(frozen_at: str, git_head: str, roles: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(json.loads(V2_PROTOCOL.read_text(encoding="utf-8")))
    value["protocol_version"] = "toolcredit_diagnostic_v3"
    value["schema_version"] = "toolcredit_diagnostic_schema_v3"
    value["frozen_at"] = frozen_at
    value["git_head_at_freeze"] = git_head
    value["supersedes"] = {
        "protocol_version": "toolcredit_diagnostic_v2",
        "manifest_sha256": sha256(V2_MANIFEST),
        "reason": (
            "M6 completed E5; M7 requires a narrow role/materialization/evaluator runtime binding "
            "before any full-panel model output"
        ),
    }
    value["checkpoints"] = _checkpoint_protocol(roles)
    value["checkpoint_locator_gate"] = {
        "timing": "before any M7 GPU smoke or canonical output",
        "rule": (
            "raw/SFT/E3/E5 source roles and E3/E5 materialized exports must all be explicit, "
            "non-null, role-correct, hash/structure/completion validated, and use the common SFT tokenizer"
        ),
        "forbidden_resolution": ["latest", "glob", "directory timestamp", "implicit null", "role swap"],
        "common_tokenizer_path": str(COMMON_TOKENIZER.relative_to(ROOT)),
        "materialization_manifests": {
            role: str(path.relative_to(ROOT)) for role, path in MATERIALIZATION_MANIFESTS.items()
        },
    }
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["evaluator"] = {
        "version": "toolcredit_m7_evaluator_v1",
        "resolved_eval_config": str(CONFIG_PATH.relative_to(ROOT)),
        "resolved_eval_config_sha256": sha256(CONFIG_PATH),
        "trajectory_schema": str(SCHEMA_PATH.relative_to(ROOT)),
        "trajectory_schema_sha256": sha256(SCHEMA_PATH),
        "source_sha256": _source_hashes(),
        "runtime_dependencies": _runtime_dependencies(),
        "agent_loop": {
            "class": "eval.generate.M7MetadataAgentLoop",
            "base_behavior_class": "rl.custom.tool_agent_loop.ToolCreditAgentLoop",
            "metadata_parent_class": "rl.custom.turn_credit_agent_loop.TurnCreditAgentLoop",
            "advantage_wrapper_installed": False,
            "behavioral_identity_fixture": "eval/test_m7_agent_loop_identity.py",
            "allowed_difference": "turn_credit_ledger audit metadata only; fixture suppresses timing telemetry and requires every behavior field/base extra to be identical",
        },
        "diagnostic_evidence": config["diagnostic_evidence"],
        "expected_workload": {
            "roles": 4,
            "questions": 760,
            "greedy_trajectories": 3040,
            "sampled_trajectories": 12160,
            "total_trajectories": 15200,
        },
        "generation_approval_state": "GPU_SMOKE_NOT_AUTHORIZED_AT_V3_FREEZE_BUILD",
    }
    return value


def closure_files() -> list[str]:
    additions = [
        "plans/M7_DIAGNOSTIC_FREEZE_V3.md",
        "eval/diagnostic_protocol_v3.json",
        "eval/diagnostic_freeze_v2.sha256",
        "eval/build_diagnostic_freeze_v3.py",
        "eval/verify_diagnostic_freeze_v3_semantics.py",
        "eval/test_verify_diagnostic_freeze_v3_semantics.py",
        *EVALUATOR_FILES,
        *(str(path.relative_to(ROOT)) for path in MATERIALIZATION_MANIFESTS.values()),
        "rl/runs/e5_turn_credit_20260823_082012/analysis/m6_e5_completion_manifest.json",
        "rl/runs/e5_turn_credit_20260823_082012/wrapper_installation.json",
        "rl/runs/e5_turn_credit_20260823_082012/status.json",
        "rl/runs/e5_turn_credit_20260823_082012/checkpoints/latest_checkpointed_iteration.txt",
        "rl/runs/e5_turn_credit_20260823_082012/resolved_config.yaml",
    ]
    ordered = list(dict.fromkeys([*_v2_entries(), *additions]))
    if "eval/diagnostic_freeze_v3.sha256" in ordered:
        raise AssertionError("v3 manifest cannot include itself")
    return ordered


def build(frozen_at: str) -> None:
    verify_v2()
    roles = resolved_checkpoint_roles(require_materialized=True)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    protocol = protocol_v3(frozen_at, git_head, roles)
    write_json(PROTOCOL_PATH, protocol)
    materializations = {
        role: sha256(path) for role, path in MATERIALIZATION_MANIFESTS.items()
    }
    LEDGER_PATH.write_text(
        "# M7 diagnostic protocol v3 freeze ledger\n\n"
        f"- 冻结时间（UTC）：`{frozen_at}`\n"
        f"- Git HEAD：`{git_head}`\n"
        "- 状态：`FROZEN_V3_READY_FOR_GPU_SMOKE_REVIEW`；尚未运行80条GPU smoke、canonical evaluation或E7。\n"
        f"- parent v2 manifest SHA256：`{sha256(V2_MANIFEST)}`（37-file closure保持逐字节不变）。\n"
        "- v3唯一语义范围：解析M6完成后的E5 step-200 role、绑定E3/E5官方FSDP materialization与统一evaluator closure；panel/generation/tool/verifier/taxonomy/pairing/bootstrap/data-isolation继续沿用v2。\n"
        "- 用户批准解释约束：metadata-only M7 loop必须通过与ToolCreditAgentLoop的behavioral-identity fixture，只有audit metadata可不同。\n"
        "- 诊断证据层级：greedy paired failure transitions为primary；sampled matched-index transitions为secondary/exploratory，不从matched index作强因果主张。\n"
        f"- E3 materialization manifest SHA256：`{materializations['e3_grpo_baseline_step_200']}`。\n"
        f"- E5 materialization manifest SHA256：`{materializations['e5_turn_credit_step_200']}`。\n"
        f"- evaluator config SHA256：`{sha256(CONFIG_PATH)}`；trajectory schema SHA256：`{sha256(SCHEMA_PATH)}`。\n\n"
        "| artifact | SHA256 | Git blob ID |\n|---|---|---|\n"
        f"| `eval/diagnostic_protocol_v3.json` | `{sha256(PROTOCOL_PATH)}` | `{git_blob(PROTOCOL_PATH)}` |\n"
        f"| `eval/m7_eval_config.json` | `{sha256(CONFIG_PATH)}` | `{git_blob(CONFIG_PATH)}` |\n"
        f"| `eval/m7_trajectory_schema.json` | `{sha256(SCHEMA_PATH)}` | `{git_blob(SCHEMA_PATH)}` |\n\n"
        "完整闭包由`eval/diagnostic_freeze_v3.sha256`锁定；manifest不自哈希，verification report另行记录其hash。\n",
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        "\n".join(f"{sha256(ROOT / relative)}  {relative}" for relative in closure_files()) + "\n",
        encoding="utf-8",
    )


def verify() -> dict[str, Any]:
    verify_v2()
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid v3 manifest entry: {line}")
        entries[relative] = digest
    if set(entries) != set(closure_files()):
        raise ValueError("v3 manifest closure differs from declared file set")
    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v3 freeze hash mismatch: {relative}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "toolcredit_diagnostic_v3":
        raise ValueError("unexpected v3 protocol identity")
    roles = resolved_checkpoint_roles(require_materialized=True)
    if protocol.get("checkpoints") != _checkpoint_protocol(roles):
        raise ValueError("v3 checkpoint runtime binding drifted")
    return {
        "verification": "passed",
        "manifest_sha256": sha256(MANIFEST_PATH),
        "closure_file_count": len(entries),
        "checkpoint_roles": list(roles),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-at")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        report = verify()
    else:
        if not args.frozen_at:
            parser.error("--frozen-at is required when building v3")
        build(args.frozen_at)
        report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
