"""Build/verify diagnostic freeze v2 with fail-closed checkpoint locators."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze import canonical_json, git_blob, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL = ROOT / "eval/diagnostic_protocol_v1.json"
PROTOCOL_PATH = ROOT / "eval/diagnostic_protocol_v2.json"
PANEL_PATH = ROOT / "eval/diagnostic_panel_v1.jsonl"
TAXONOMY_PATH = ROOT / "eval/diagnostic_taxonomy_v1.json"
LEDGER_PATH = ROOT / "plans/M7_DIAGNOSTIC_FREEZE_V2.md"
MANIFEST_PATH = ROOT / "eval/diagnostic_freeze_v2.sha256"

RAW_LOCATOR = Path("/home/jovyan/Qwen/Qwen3-1.7B")
SFT_LOCATOR = ROOT / "sft/checkpoints/qwen3-1.7b-sft"
E3_LOCATOR = ROOT / "rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200"
E3_RUN = E3_LOCATOR.parents[1]


def _checkpoint_protocol() -> dict[str, Any]:
    return {
        "raw": {
            "role": "raw_base_model",
            "state": "available",
            "locator": str(RAW_LOCATOR),
            "config_sha256": sha256(RAW_LOCATOR / "config.json"),
            "model_index_sha256": sha256(RAW_LOCATOR / "model.safetensors.index.json"),
        },
        "sft": {
            "role": "sft_start",
            "state": "available",
            "locator": "sft/checkpoints/qwen3-1.7b-sft",
            "model_sha256": sha256(SFT_LOCATOR / "model.safetensors"),
        },
        "e3": {
            "role": "e3_grpo_baseline_step_200",
            "state": "available",
            "locator": "rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200",
            "run_name": "e3_grpo_baseline_20260819_224555",
            "step": 200,
            "resolved_config_sha256": sha256(E3_RUN / "resolved_config.yaml"),
            "status_sha256": sha256(E3_RUN / "status.json"),
            "tracker_sha256": sha256(E3_LOCATOR.parent / "latest_checkpointed_iteration.txt"),
            "required_file_bytes": {
                "actor/model_world_size_1_rank_0.pt": 8127118819,
                "actor/optim_world_size_1_rank_0.pt": 13764626512,
                "actor/extra_state_world_size_1_rank_0.pt": 15205,
                "actor/fsdp_config.json": 46,
                "data.pt": 6868,
            },
        },
        "e5": {
            "role": "e5_turn_credit_step_200",
            "state": "future",
            "locator": None,
            "required_before_diagnostic_evaluation": True,
            "required_locator_pattern": "rl/runs/e5_turn_credit_<UTC>/checkpoints/global_step_200",
        },
    }


def protocol_v2(frozen_at: str, git_head: str) -> dict[str, Any]:
    value = copy.deepcopy(json.loads(V1_PROTOCOL.read_text(encoding="utf-8")))
    value["protocol_version"] = "toolcredit_diagnostic_v2"
    value["schema_version"] = "toolcredit_diagnostic_schema_v2"
    value["frozen_at"] = frozen_at
    value["git_head_at_freeze"] = git_head
    value["supersedes"] = {
        "protocol_version": "toolcredit_diagnostic_v1",
        "manifest_sha256": sha256(ROOT / "eval/diagnostic_freeze_v1.sha256"),
        "reason": "v1 E3 checkpoint locator was nonexistent and checkpoint roles were not fail-closed validated",
    }
    value["checkpoints"] = _checkpoint_protocol()
    value["checkpoint_locator_gate"] = {
        "timing": "before any canonical E5 v2 output",
        "rule": "Every non-null locator must exist and pass its hard-coded role/structure/identity checks; only future E5 may be null.",
        "raw": "existing Qwen3 HF directory, qwen3 config, model index and frozen hashes",
        "sft": "exact project SFT start path, complete HF files and frozen model hash",
        "e3": "exact canonical run/checkpoints/global_step_200 path, completed status, tracker=200, full veRL actor/optimizer/extra/data structure, SFT origin and frozen identity",
        "e5": "pre-training locator must be null/state=future; a later diagnostic protocol version must resolve and validate the canonical step-200 role before evaluation",
    }
    return value


def closure_files() -> list[str]:
    return [
        "plans/M7_DIAGNOSTIC_FREEZE_V2.md",
        "eval/diagnostic_protocol_v2.json",
        "eval/diagnostic_panel_v1.jsonl",
        "eval/diagnostic_taxonomy_v1.json",
        "eval/diagnostic_freeze_v1.sha256",
        "eval/diagnostic_protocol_v1.json",
        "eval/build_diagnostic_freeze_v2.py",
        "eval/__init__.py",
        "rl/configs/e3_grpo_baseline.yaml",
        "rl/configs/e5_turn_credit.yaml",
        "rl/custom/turn_advantage.py",
        "rl/custom/turn_credit_agent_loop.py",
        "rl/custom/turn_credit_agent_loop_config.yaml",
        "rl/launch/e5_turn_credit.py",
        "rl/analyze_e5.py",
        "rl/prepare_e5_adoption_review.py",
        "rewards/verifier.py",
        "rewards/composite_reward.py",
        "rewards/format_reward.py",
        "rl/custom/reward.py",
        "env/sandbox.py",
        "rl/custom/sandbox_tool.py",
        "rl/custom/tool_config.yaml",
        "data/contamination_report_m4.md",
        "rl/data/manifest.json",
        "rl/data/e3_train.parquet",
        "rl/data/e3_val_math500_100.parquet",
        "data/processed/math500.jsonl",
        "data/processed/aime24.jsonl",
        "data/processed/aime25.jsonl",
        "data/processed/gsm8k_test200.jsonl",
        "sft/checkpoints/qwen3-1.7b-sft/model.safetensors",
        "sft/checkpoints/qwen3-1.7b-sft/chat_template.jinja",
        "sft/checkpoints/qwen3-1.7b-sft/tokenizer_config.json",
        "rl/runs/e3_grpo_baseline_20260819_224555/resolved_config.yaml",
        "rl/runs/e3_grpo_baseline_20260819_224555/status.json",
        "rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/latest_checkpointed_iteration.txt",
    ]


def _validate_checkpoint_files(protocol: dict[str, Any]) -> None:
    from rl.launch.e5_turn_credit import _validate_checkpoint_locators

    summary = _validate_checkpoint_locators(protocol)
    if summary["e3"]["state"] != "validated" or summary["e5"]["state"] != "future":
        raise ValueError("checkpoint role validation did not reach the expected v2 states")
    e3 = protocol["checkpoints"]["e3"]
    for relative, expected_bytes in e3["required_file_bytes"].items():
        actual_bytes = (E3_LOCATOR / relative).stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"E3 checkpoint identity drift for {relative}: {actual_bytes} != {expected_bytes}")
    if e3["resolved_config_sha256"] != sha256(E3_RUN / "resolved_config.yaml"):
        raise ValueError("E3 resolved config identity drift")
    if e3["status_sha256"] != sha256(E3_RUN / "status.json"):
        raise ValueError("E3 status identity drift")
    if e3["tracker_sha256"] != sha256(E3_LOCATOR.parent / "latest_checkpointed_iteration.txt"):
        raise ValueError("E3 checkpoint tracker identity drift")
    raw = protocol["checkpoints"]["raw"]
    if raw["config_sha256"] != sha256(RAW_LOCATOR / "config.json"):
        raise ValueError("raw checkpoint config identity drift")
    if raw["model_index_sha256"] != sha256(RAW_LOCATOR / "model.safetensors.index.json"):
        raise ValueError("raw checkpoint index identity drift")


def build(frozen_at: str) -> None:
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    protocol = protocol_v2(frozen_at, git_head)
    write_json(PROTOCOL_PATH, protocol)
    _validate_checkpoint_files(protocol)
    panel_count = len(PANEL_PATH.read_text(encoding="utf-8").splitlines())
    LEDGER_PATH.write_text(
        "# M7 diagnostic protocol v2 freeze ledger\n\n"
        f"- 冻结时间（UTC）：`{frozen_at}`\n"
        f"- Git HEAD：`{git_head}`\n"
        "- 状态：`FROZEN_V2_HUMAN_ADOPTION_REVIEW_PENDING`。\n"
        "- v2唯一协议变更：修正E3 locator并新增所有checkpoint locator的role-aware fail-closed gate；panel、taxonomy、generation、tool、verifier、pairing与bootstrap沿用v1。\n"
        "- E3 locator：`rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200`。\n"
        "- checkpoint gate：raw/SFT/E3非空locator必须存在并匹配冻结role/结构/identity；E5在pre-training freeze中必须显式为`state=future, locator=null`，未来诊断执行前另行version bump解析step-200 locator。\n"
        "- v1 smoke只保留为noncanonical wiring evidence；Codex preliminary adoption labels不计入human precision gate。canonical v2 smoke在真实human review前禁止启动。\n\n"
        f"panel仍为{panel_count}题；v1 panel/taxonomy文件按相同hash复用。\n\n"
        "| artifact | SHA256 | Git blob ID |\n|---|---|---|\n"
        f"| `eval/diagnostic_protocol_v2.json` | `{sha256(PROTOCOL_PATH)}` | `{git_blob(PROTOCOL_PATH)}` |\n"
        f"| `eval/diagnostic_panel_v1.jsonl` | `{sha256(PANEL_PATH)}` | `{git_blob(PANEL_PATH)}` |\n"
        f"| `eval/diagnostic_taxonomy_v1.json` | `{sha256(TAXONOMY_PATH)}` | `{git_blob(TAXONOMY_PATH)}` |\n\n"
        "完整闭包由`eval/diagnostic_freeze_v2.sha256`锁定；manifest不自哈希，其hash由preflight单独记录。\n",
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        "\n".join(f"{sha256(ROOT / relative)}  {relative}" for relative in closure_files()) + "\n",
        encoding="utf-8",
    )


def verify() -> None:
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid v2 manifest entry: {line}")
        entries[relative] = digest
    if set(entries) != set(closure_files()):
        raise ValueError("v2 manifest closure differs from its declared file set")
    for relative, expected in entries.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"v2 freeze hash mismatch: {relative}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "toolcredit_diagnostic_v2":
        raise ValueError("unexpected v2 protocol version")
    _validate_checkpoint_files(protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-at")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    if not args.frozen_at:
        parser.error("--frozen-at is required when building v2")
    build(args.frozen_at)
    verify()
    print(f"manifest_sha256={sha256(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
