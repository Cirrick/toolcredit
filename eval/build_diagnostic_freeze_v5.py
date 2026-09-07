"""Build and verify M9's protocol v5: frozen v4 plus the ``notool`` generation mode and four roles.

v5 changes exactly this relative to v4 (plans/M9.md §3): it adds a ``generation_mode`` field
(``tir`` = the five existing roles, field-for-field equal to v4; ``notool`` = chat template
without tools, single-turn generation), the four ``notool`` roles with their weight bindings
(three reuse frozen v3 weights, one is the new E3-NoTool materialization), binds the M9
evaluator files, and records which M7/M8 canonical artifacts are reused byte-for-byte.
Panel, generation, tool, verifier, taxonomy, pairing, bootstrap, persistence, prompt and
data-isolation semantics are carried over unchanged and proven so by
``eval/verify_diagnostic_freeze_v5_semantics.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze import git_blob, write_json
from eval.build_diagnostic_freeze_v4 import (
    M7_CANONICAL_RUN,
    MANIFEST_PATH as V4_MANIFEST,
    PROTOCOL_PATH as V4_PROTOCOL,
    verify as verify_v4,
    verify_m7_canonical_ledger,
)
from eval.checkpoints import ROOT, sha256
from eval.checkpoints_v5 import (
    GENERATION_MODE_NOTOOL,
    GENERATION_MODE_TIR,
    NOTOOL_MANIFEST,
    NOTOOL_ROLE_NAMES,
    NOTOOL_RUN,
    NOTOOL_WEIGHT_ROLES,
    ROLE_V5,
    resolved_notool_roles,
)

PROTOCOL_PATH = ROOT / "eval/diagnostic_protocol_v5.json"
MANIFEST_PATH = ROOT / "eval/diagnostic_freeze_v5.sha256"
LEDGER_PATH = ROOT / "plans/M9_DIAGNOSTIC_FREEZE_V5.md"
M8_CANONICAL_RUN = ROOT / "eval/runs/m8_e5v2_eval_20260905_134737"
M8_HASHES = M8_CANONICAL_RUN / "hashes.sha256"
V4_CLOSURE_FILE_COUNT = 87
M9_EVALUATOR_FILES = (
    "eval/checkpoints_v5.py",
    "eval/m9_notool_eval.py",
    "eval/test_m9_notool_eval.py",
    "scripts/m9/run_m9_eval.sh",
    "rl/configs/e3notool_grpo.yaml",
    "rl/launch/e3notool_grpo.py",
    "rl/validate_e3notool_run.py",
)
NOTOOL_RUN_FILES = (
    "status.json",
    "resolved_config.yaml",
    "checkpoints/latest_checkpointed_iteration.txt",
    "analysis/formal_completion_gate.json",
    "recovery/resume_from_125_20260907_140752/recovery.json",
)
E3_ROLE = "e3_grpo_baseline_step_200"
RAW_ROLE = "raw_base_model"
# plans/M9.md §3 pairing table.  ``after - before`` is the reported delta.
PAIRINGS = (
    {"before_role": E3_ROLE, "after_role": ROLE_V5, "priority": "primary", "question": "tool_net_contribution_at_rl_endpoint"},
    {"before_role": "sft_start_notool", "after_role": ROLE_V5, "priority": "secondary", "question": "notool_rl_gain"},
    {"before_role": "raw_base_model_notool", "after_role": "sft_start_notool", "priority": "secondary", "question": "start_point_debt_D_start"},
    {"before_role": E3_ROLE, "after_role": "e3_grpo_baseline_step_200_notool", "priority": "secondary", "question": "inference_time_tool_marginal_value"},
    {"before_role": "sft_start_notool", "after_role": "e3_grpo_baseline_step_200_notool", "priority": "secondary", "question": "tir_rl_transfer_to_notool_reasoning"},
    {"before_role": "e3_grpo_baseline_step_200_notool", "after_role": ROLE_V5, "priority": "secondary", "question": "notool_inference_trained_with_vs_without_tool"},
    {"before_role": RAW_ROLE, "after_role": "raw_base_model_notool", "priority": "descriptive", "question": "m1_zero_shot_tool_gain_replication"},
)
NOTOOL_MODE_SEMANTICS = {
    "chat_template_tools": None,
    "system_prompt": None,
    "agent_loop": "verl single_turn_agent semantics: one apply_chat_template(messages, tools=None, enable_thinking=False) call, one generation, no tool turn",
    "max_response_tokens_total": 3072,
    "termination_reasons": ["final_answer", "length"],
    "tool_metadata": "tool_call_counts/tool_success_count/tool_error_count/tool_parse_error_count are constant 0",
    "extra_row_fields": ["tool_tag_emitted", "tool_tag_count"],
    "tool_tag_detection": "rl.validate_e3notool_run.tool_tag_stats on the response text only; detection never changes scoring",
    "prompt_suffix_and_sampling": "identical to the tir mode (frozen v2 generation block)",
}


def _v4_entries() -> list[str]:
    entries: list[str] = []
    for line in V4_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64:
            raise ValueError(f"invalid parent v4 manifest line: {line}")
        entries.append(relative)
    if len(entries) != V4_CLOSURE_FILE_COUNT or len(set(entries)) != V4_CLOSURE_FILE_COUNT:
        raise ValueError(f"v5 parent must be the exact {V4_CLOSURE_FILE_COUNT}-file frozen v4 closure")
    return entries


def verify_m8_canonical_ledger() -> dict[str, Any]:
    """Re-verify every byte of the reused M8 canonical run (E5-v2 3,800 trajectories)."""
    checked = 0
    for line in M8_HASHES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        path = M8_CANONICAL_RUN / relative.lstrip("*")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"M8 canonical artifact drifted: {relative}")
        checked += 1
    status = json.loads((M8_CANONICAL_RUN / "status.json").read_text(encoding="utf-8"))
    if status.get("protocol_version") != "toolcredit_diagnostic_v4" or status.get("status") != "e5v2_evaluation_completed":
        raise ValueError("M8 canonical run is not the completed v4 protocol run")
    if status.get("hashes_sha256") != sha256(M8_HASHES):
        raise ValueError("M8 canonical hash ledger identity drifted")
    return {
        "run_name": M8_CANONICAL_RUN.name,
        "hashes_sha256": sha256(M8_HASHES),
        "status_sha256": sha256(M8_CANONICAL_RUN / "status.json"),
        "verified_artifact_count": checked,
    }


def _source_hashes() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in M9_EVALUATOR_FILES}


def protocol_v5(
    frozen_at: str, git_head: str, roles: dict[str, dict[str, Any]], reuse_m7: dict[str, Any], reuse_m8: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(json.loads(V4_PROTOCOL.read_text(encoding="utf-8")))
    value["protocol_version"] = "toolcredit_diagnostic_v5"
    value["schema_version"] = "toolcredit_diagnostic_schema_v5"
    value["frozen_at"] = frozen_at
    value["git_head_at_freeze"] = git_head
    value["supersedes"] = {
        "protocol_version": "toolcredit_diagnostic_v4",
        "manifest_sha256": sha256(V4_MANIFEST),
        "reason": (
            "M9 completed the E3-NoTool run; v5 adds the notool generation mode and four notool roles "
            "(three over frozen v3 weights, one new materialization), reuses the M7/M8 canonical trajectories "
            "byte-for-byte, and changes no panel/generation/tool/verifier/taxonomy/pairing/bootstrap semantics"
        ),
    }
    value["generation_mode"] = {
        "default": GENERATION_MODE_TIR,
        "rule": "checkpoint roles without a generation_mode entry are tir; the tir mode is the frozen v2 generation/tool/prompt block unchanged",
        "modes": {
            GENERATION_MODE_TIR: {"definition": "frozen v2 generation + tool + prompt semantics (code_interpreter hermes loop)"},
            GENERATION_MODE_NOTOOL: copy.deepcopy(NOTOOL_MODE_SEMANTICS),
        },
        "notool_roles": {role: NOTOOL_WEIGHT_ROLES[role] for role in NOTOOL_ROLE_NAMES},
    }
    for role in NOTOOL_ROLE_NAMES:
        entry = copy.deepcopy(roles[role])
        entry["state"] = "validated"
        entry["role"] = role
        entry["generation_mode"] = GENERATION_MODE_NOTOOL
        value["checkpoints"][role] = entry
    gate = value["checkpoint_locator_gate"]
    gate["materialization_manifests"][ROLE_V5] = str(NOTOOL_MANIFEST.relative_to(ROOT))
    gate["rule"] = (
        gate["rule"]
        + "; v5 additionally requires the E3-NoTool source role, its formal completion gate and its "
        "materialized export to be explicit, non-null, role-correct and hash validated, and binds the three "
        "reused-weight notool roles to the frozen v3 raw/SFT/E3 locators"
    )
    value["evaluator"]["m9_extension"] = {
        "version": "toolcredit_m9_evaluator_v1",
        "new_roles": list(NOTOOL_ROLE_NAMES),
        "new_checkpoint_role": ROLE_V5,
        "source_run": NOTOOL_RUN.name,
        "source_sha256": _source_hashes(),
        "expected_workload": {
            "new_roles": 4,
            "questions": 760,
            "greedy_trajectories_per_role": 760,
            "sampled_trajectories_per_role": 3040,
            "trajectories_per_role": 3800,
            "total_new_trajectories": 15200,
            "reused_canonical_trajectories": 19000,
        },
        "reused_m7_canonical": reuse_m7,
        "reused_m8_canonical": reuse_m8,
        "pairings": [dict(item) for item in PAIRINGS],
        "start_point_debt": "D_start = acc(raw_base_model_notool) - acc(sft_start_notool), greedy FULL760 (plan §8)",
        "generation_approval_state": "AUTHORIZED_BY_APPROVED_M9_PLAN_STEP_4",
        "generation_code": (
            "eval.m9_notool_eval reuses eval.generate.SGLangHTTPServer / to_sglang_sampling_params / "
            "derive_sample_seed / load_panel_questions / _sampling_params unchanged; the notool prompt is "
            "verl.utils.chat_template.apply_chat_template(tokenizer, [user], tools=None, enable_thinking=False)"
        ),
    }
    return value


def closure_files() -> list[str]:
    additions = [
        "plans/M9_DIAGNOSTIC_FREEZE_V5.md",
        "eval/diagnostic_protocol_v5.json",
        "eval/diagnostic_freeze_v4.sha256",
        "eval/build_diagnostic_freeze_v5.py",
        "eval/verify_diagnostic_freeze_v5_semantics.py",
        *M9_EVALUATOR_FILES,
        str(NOTOOL_MANIFEST.relative_to(ROOT)),
        *(str((NOTOOL_RUN / name).relative_to(ROOT)) for name in NOTOOL_RUN_FILES),
        str(M8_HASHES.relative_to(ROOT)),
        str((M8_CANONICAL_RUN / "status.json").relative_to(ROOT)),
    ]
    ordered = list(dict.fromkeys([*_v4_entries(), *additions]))
    if "eval/diagnostic_freeze_v5.sha256" in ordered:
        raise AssertionError("v5 manifest cannot include itself")
    return ordered


def _expected_entries(roles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for role in NOTOOL_ROLE_NAMES:
        entry = copy.deepcopy(roles[role])
        entry["state"] = "validated"
        entry["role"] = role
        entry["generation_mode"] = GENERATION_MODE_NOTOOL
        expected[role] = entry
    return expected


def build(frozen_at: str) -> None:
    verify_v4()
    reuse_m7 = verify_m7_canonical_ledger()
    reuse_m8 = verify_m8_canonical_ledger()
    roles = resolved_notool_roles()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    protocol = protocol_v5(frozen_at, git_head, roles, reuse_m7, reuse_m8)
    write_json(PROTOCOL_PATH, protocol)
    LEDGER_PATH.write_text(
        "# M9 diagnostic protocol v5 freeze ledger\n\n"
        f"- 冻结时间（UTC）：`{frozen_at}`\n"
        f"- Git HEAD：`{git_head}`\n"
        "- 状态：`FROZEN_V5_READY_FOR_NOTOOL_GENERATION`；四个 notool 角色的 15,200 条评测轨迹尚未生成。\n"
        f"- parent v4 manifest SHA256：`{sha256(V4_MANIFEST)}`（{V4_CLOSURE_FILE_COUNT}-file closure 保持逐字节不变）。\n"
        f"- v5 唯一语义范围：新增 `generation_mode`（`tir` = 现有五角色逐字段等于 v4；`notool` = chat template 不传 tools、"
        "单轮生成、tool 计数恒 0、新增 `tool_tag_emitted/tool_tag_count`），新增四个 notool 角色 "
        f"`{'`、`'.join(NOTOOL_ROLE_NAMES)}`（前三者复用冻结的 raw/SFT/E3 权重，`{ROLE_V5}` 为 run "
        f"`{NOTOOL_RUN.name}` 的官方 FSDP materialization），绑定 M9 evaluator 文件，并记录复用的 M7/M8 canonical 产物；"
        "panel/generation/tool/verifier/taxonomy/pairing/bootstrap/persistence/prompt/data-isolation 继续沿用 v4。\n"
        f"- 配对：E3→{ROLE_V5} 为 primary；五组 secondary 与一组 descriptive 见 protocol `evaluator.m9_extension.pairings`；"
        "greedy paired transitions 仍为 primary diagnostic，sampled matched-index 为 secondary/exploratory。\n"
        f"- E3-NoTool materialization manifest SHA256：`{sha256(NOTOOL_MANIFEST)}`。\n"
        f"- 复用的 M7 canonical run：`{reuse_m7['run_name']}`，hashes.sha256 `{reuse_m7['hashes_sha256']}`，"
        f"full exact-key digest `{reuse_m7['full_exact_key_digest']}`；M8 canonical run：`{reuse_m8['run_name']}`，"
        f"hashes.sha256 `{reuse_m8['hashes_sha256']}`。\n\n"
        "| artifact | SHA256 | Git blob ID |\n|---|---|---|\n"
        f"| `eval/diagnostic_protocol_v5.json` | `{sha256(PROTOCOL_PATH)}` | `{git_blob(PROTOCOL_PATH)}` |\n"
        f"| `{NOTOOL_MANIFEST.relative_to(ROOT)}` | `{sha256(NOTOOL_MANIFEST)}` | `{git_blob(NOTOOL_MANIFEST)}` |\n\n"
        "完整闭包由 `eval/diagnostic_freeze_v5.sha256` 锁定；manifest 不自哈希，verification report 另行记录其 hash。\n",
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        "\n".join(f"{sha256(ROOT / relative)}  {relative}" for relative in closure_files()) + "\n",
        encoding="utf-8",
    )


def verify() -> dict[str, Any]:
    verify_v4()
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid v5 manifest entry: {line}")
        entries[relative] = digest
    if set(entries) != set(closure_files()):
        raise ValueError("v5 manifest closure differs from declared file set")
    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v5 freeze hash mismatch: {relative}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "toolcredit_diagnostic_v5":
        raise ValueError("unexpected v5 protocol identity")
    expected_entries = _expected_entries(resolved_notool_roles())
    for role, entry in expected_entries.items():
        if protocol.get("checkpoints", {}).get(role) != entry:
            raise ValueError(f"v5 runtime binding drifted for {role}")
    v4 = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    frozen_roles = {key: value for key, value in protocol["checkpoints"].items() if key not in NOTOOL_ROLE_NAMES}
    if frozen_roles != v4["checkpoints"]:
        raise ValueError("v5 altered a frozen v4 checkpoint role")
    reuse_m7 = verify_m7_canonical_ledger()
    reuse_m8 = verify_m8_canonical_ledger()
    extension = protocol["evaluator"]["m9_extension"]
    if extension["reused_m7_canonical"] != reuse_m7 or extension["reused_m8_canonical"] != reuse_m8:
        raise ValueError("reused M7/M8 canonical ledger identity drifted")
    return {
        "verification": "passed",
        "manifest_sha256": sha256(MANIFEST_PATH),
        "closure_file_count": len(entries),
        "checkpoint_roles": list(protocol["checkpoints"]),
        "reused_m7_canonical": reuse_m7,
        "reused_m8_canonical": reuse_m8,
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
            parser.error("--frozen-at is required when building v5")
        build(args.frozen_at)
        report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
