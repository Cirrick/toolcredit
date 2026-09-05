"""Build and verify M8's protocol v4: frozen v3 plus the single E5-v2 role binding.

v4 changes exactly one thing relative to v3 (plans/M8.md §8 step 6): it adds the role
``e5v2_conserved_credit_step_200`` with its materialized export, binds the M8 evaluator
files that generate/score/pair the new role, and records which M7 canonical artifacts are
reused byte-for-byte.  Panel, generation, tool, verifier, taxonomy, pairing, bootstrap,
persistence, prompt and data-isolation semantics are carried over unchanged and proven so
by ``eval/verify_diagnostic_freeze_v4_semantics.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze import git_blob, write_json
from eval.build_diagnostic_freeze_v3 import MANIFEST_PATH as V3_MANIFEST
from eval.build_diagnostic_freeze_v3 import PROTOCOL_PATH as V3_PROTOCOL
from eval.build_diagnostic_freeze_v3 import verify as verify_v3
from eval.checkpoints import ROOT, sha256
from eval.checkpoints_v4 import E5V2_MANIFEST, E5V2_RUN, ROLE_V4, resolved_e5v2_role

PROTOCOL_PATH = ROOT / "eval/diagnostic_protocol_v4.json"
MANIFEST_PATH = ROOT / "eval/diagnostic_freeze_v4.sha256"
LEDGER_PATH = ROOT / "plans/M8_DIAGNOSTIC_FREEZE_V4.md"
M7_CANONICAL_RUN = ROOT / "eval/runs/m7_unified_eval_20260824_201032"
M7_HASHES = M7_CANONICAL_RUN / "hashes.sha256"
M7_ANALYSIS_HASHES = M7_CANONICAL_RUN / "analysis_hashes.sha256"
M8_EVALUATOR_FILES = (
    "eval/checkpoints_v4.py",
    "eval/m8_e5v2_eval.py",
    "eval/test_m8_e5v2_eval.py",
    "scripts/m8/run_e5v2_eval.sh",
    "rl/custom/turn_advantage_v2.py",
    "rl/custom/turn_credit_v2_agent_loop.py",
    "rl/validate_e5v2_run.py",
)
E5V2_RUN_FILES = (
    "status.json",
    "resolved_config.yaml",
    "wrapper_installation.json",
    "checkpoints/latest_checkpointed_iteration.txt",
    "analysis/formal_completion_gate.json",
)
PAIRINGS = (
    {"before_role": "e3_grpo_baseline_step_200", "after_role": ROLE_V4, "priority": "primary"},
    {"before_role": "e5_turn_credit_step_200", "after_role": ROLE_V4, "priority": "secondary"},
)


def _v3_entries() -> list[str]:
    entries: list[str] = []
    for line in V3_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64:
            raise ValueError(f"invalid parent v3 manifest line: {line}")
        entries.append(relative)
    if len(entries) != 66 or len(set(entries)) != 66:
        raise ValueError("v4 parent must be the exact 66-file frozen v3 closure")
    return entries


def verify_m7_canonical_ledger() -> dict[str, Any]:
    """Re-verify every byte of the reused M7 canonical run before anything touches it."""
    checked = 0
    for ledger in (M7_HASHES, M7_ANALYSIS_HASHES):
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split(maxsplit=1)
            relative = relative.lstrip("*")
            path = M7_CANONICAL_RUN / relative
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"M7 canonical artifact drifted: {relative}")
            checked += 1
    status = json.loads((M7_CANONICAL_RUN / "status.json").read_text(encoding="utf-8"))
    if status.get("protocol_version") != "toolcredit_diagnostic_v3":
        raise ValueError("M7 canonical run is not the v3 protocol run")
    return {
        "run_name": M7_CANONICAL_RUN.name,
        "hashes_sha256": sha256(M7_HASHES),
        "analysis_hashes_sha256": sha256(M7_ANALYSIS_HASHES),
        "status_sha256": sha256(M7_CANONICAL_RUN / "status.json"),
        "verified_artifact_count": checked,
        "full_exact_key_digest": status["full_scored_gate"]["exact_key_digest_sha256"],
    }


def _source_hashes() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in M8_EVALUATOR_FILES}


def protocol_v4(frozen_at: str, git_head: str, role: dict[str, Any], reuse: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(json.loads(V3_PROTOCOL.read_text(encoding="utf-8")))
    value["protocol_version"] = "toolcredit_diagnostic_v4"
    value["schema_version"] = "toolcredit_diagnostic_schema_v4"
    value["frozen_at"] = frozen_at
    value["git_head_at_freeze"] = git_head
    value["supersedes"] = {
        "protocol_version": "toolcredit_diagnostic_v3",
        "manifest_sha256": sha256(V3_MANIFEST),
        "reason": (
            "M8 completed the E5-v2 conserved-credit run; v4 adds exactly one checkpoint role and "
            "its materialization, reuses the M7 canonical raw/SFT/E3/E5-v1 trajectories byte-for-byte, "
            "and changes no panel/generation/tool/verifier/taxonomy/pairing/bootstrap semantics"
        ),
    }
    entry = copy.deepcopy(role)
    entry["state"] = "validated"
    entry["role"] = ROLE_V4
    value["checkpoints"][ROLE_V4] = entry
    gate = value["checkpoint_locator_gate"]
    gate["materialization_manifests"][ROLE_V4] = str(E5V2_MANIFEST.relative_to(ROOT))
    gate["rule"] = (
        gate["rule"]
        + "; v4 additionally requires the E5-v2 source role, its formal completion gate and "
        "its materialized export to be explicit, non-null, role-correct and hash validated"
    )
    value["evaluator"]["m8_extension"] = {
        "version": "toolcredit_m8_evaluator_v1",
        "new_role": ROLE_V4,
        "source_run": E5V2_RUN.name,
        "source_sha256": _source_hashes(),
        "expected_workload": {
            "new_roles": 1,
            "questions": 760,
            "greedy_trajectories": 760,
            "sampled_trajectories": 3040,
            "total_new_trajectories": 3800,
            "reused_canonical_trajectories": 15200,
        },
        "reused_m7_canonical": reuse,
        "pairings": list(PAIRINGS),
        "generation_approval_state": "AUTHORIZED_BY_APPROVED_M8_PLAN_STEP_6",
        "generation_code": (
            "eval.m8_e5v2_eval reuses eval.generate.build_m7_loop / SGLangHTTPServer / derive_sample_seed / "
            "load_panel_questions unchanged; only the role universe, protocol identifiers and the "
            "raw-row validator are extended to the new role"
        ),
    }
    return value


def closure_files() -> list[str]:
    additions = [
        "plans/M8_DIAGNOSTIC_FREEZE_V4.md",
        "eval/diagnostic_protocol_v4.json",
        "eval/diagnostic_freeze_v3.sha256",
        "eval/build_diagnostic_freeze_v4.py",
        "eval/verify_diagnostic_freeze_v4_semantics.py",
        *M8_EVALUATOR_FILES,
        str(E5V2_MANIFEST.relative_to(ROOT)),
        *(str((E5V2_RUN / name).relative_to(ROOT)) for name in E5V2_RUN_FILES),
        str(M7_HASHES.relative_to(ROOT)),
        str(M7_ANALYSIS_HASHES.relative_to(ROOT)),
        str((M7_CANONICAL_RUN / "status.json").relative_to(ROOT)),
    ]
    ordered = list(dict.fromkeys([*_v3_entries(), *additions]))
    if "eval/diagnostic_freeze_v4.sha256" in ordered:
        raise AssertionError("v4 manifest cannot include itself")
    return ordered


def build(frozen_at: str) -> None:
    verify_v3()
    reuse = verify_m7_canonical_ledger()
    role = resolved_e5v2_role()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    protocol = protocol_v4(frozen_at, git_head, role, reuse)
    write_json(PROTOCOL_PATH, protocol)
    LEDGER_PATH.write_text(
        "# M8 diagnostic protocol v4 freeze ledger\n\n"
        f"- 冻结时间（UTC）：`{frozen_at}`\n"
        f"- Git HEAD：`{git_head}`\n"
        "- 状态：`FROZEN_V4_READY_FOR_E5V2_GENERATION`；E5-v2 的 3,800 条评测轨迹尚未生成。\n"
        f"- parent v3 manifest SHA256：`{sha256(V3_MANIFEST)}`（66-file closure 保持逐字节不变）。\n"
        f"- v4 唯一语义范围：新增 role `{ROLE_V4}`（run `{E5V2_RUN.name}`）及其官方 FSDP materialization，"
        "绑定 M8 evaluator 文件，并记录复用的 M7 canonical 产物；panel/generation/tool/verifier/taxonomy/"
        "pairing/bootstrap/persistence/prompt/data-isolation 继续沿用 v3。\n"
        "- 配对：E3→E5-v2 为 primary，E5-v1→E5-v2 为 secondary；greedy paired transitions 仍为 primary "
        "diagnostic，sampled matched-index 为 secondary/exploratory。\n"
        f"- E5-v2 materialization manifest SHA256：`{sha256(E5V2_MANIFEST)}`。\n"
        f"- 复用的 M7 canonical run：`{reuse['run_name']}`，hashes.sha256 `{reuse['hashes_sha256']}`，"
        f"full exact-key digest `{reuse['full_exact_key_digest']}`。\n\n"
        "| artifact | SHA256 | Git blob ID |\n|---|---|---|\n"
        f"| `eval/diagnostic_protocol_v4.json` | `{sha256(PROTOCOL_PATH)}` | `{git_blob(PROTOCOL_PATH)}` |\n"
        f"| `{E5V2_MANIFEST.relative_to(ROOT)}` | `{sha256(E5V2_MANIFEST)}` | `{git_blob(E5V2_MANIFEST)}` |\n\n"
        "完整闭包由 `eval/diagnostic_freeze_v4.sha256` 锁定；manifest 不自哈希，verification report 另行记录其 hash。\n",
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        "\n".join(f"{sha256(ROOT / relative)}  {relative}" for relative in closure_files()) + "\n",
        encoding="utf-8",
    )


def verify() -> dict[str, Any]:
    verify_v3()
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid v4 manifest entry: {line}")
        entries[relative] = digest
    if set(entries) != set(closure_files()):
        raise ValueError("v4 manifest closure differs from declared file set")
    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v4 freeze hash mismatch: {relative}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "toolcredit_diagnostic_v4":
        raise ValueError("unexpected v4 protocol identity")
    role = resolved_e5v2_role()
    expected_entry = copy.deepcopy(role)
    expected_entry["state"] = "validated"
    expected_entry["role"] = ROLE_V4
    if protocol.get("checkpoints", {}).get(ROLE_V4) != expected_entry:
        raise ValueError("v4 E5-v2 checkpoint runtime binding drifted")
    v3 = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
    frozen_roles = {key: value for key, value in protocol["checkpoints"].items() if key != ROLE_V4}
    if frozen_roles != v3["checkpoints"]:
        raise ValueError("v4 altered a frozen v3 checkpoint role")
    reuse = verify_m7_canonical_ledger()
    if protocol["evaluator"]["m8_extension"]["reused_m7_canonical"] != reuse:
        raise ValueError("reused M7 canonical ledger identity drifted")
    return {
        "verification": "passed",
        "manifest_sha256": sha256(MANIFEST_PATH),
        "closure_file_count": len(entries),
        "checkpoint_roles": list(protocol["checkpoints"]),
        "reused_m7_canonical": reuse,
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
            parser.error("--frozen-at is required when building v4")
        build(args.frozen_at)
        report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
