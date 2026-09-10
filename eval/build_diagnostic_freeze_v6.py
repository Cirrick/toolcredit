"""Build and verify M10's protocol v6: frozen v5 plus the two E7 dynamic-filtering roles.

v6 changes exactly this relative to v5 (plans/M10.md §5): it adds two ``tir`` roles,
``e7_dynamic_filter_step_200`` and ``e7_equal_compute``, both materialized from the single E7 run;
binds the M10 evaluator files; records which M7/M8/M9 canonical artifacts are reused
byte-for-byte; and declares the two E3 -> E7 pairings that answer the plan §9 matrix (equal
effective step and equal rollout compute).  Panel, generation, tool, verifier, taxonomy, pairing,
bootstrap, persistence, prompt, data-isolation and the ``notool`` mode block are carried over
unchanged and proven so by ``eval/verify_diagnostic_freeze_v6_semantics.py``.

The equal-compute role's effective step is discovered from the run, never assumed: plan §3.4 only
fixes the *rule* (first update whose cumulative rollout trajectories reach E3's 102,400).
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze import git_blob, write_json
from eval.build_diagnostic_freeze_v4 import M7_CANONICAL_RUN, verify_m7_canonical_ledger
from eval.build_diagnostic_freeze_v5 import (
    MANIFEST_PATH as V5_MANIFEST,
    PROTOCOL_PATH as V5_PROTOCOL,
    verify as verify_v5,
    verify_m8_canonical_ledger,
)
from eval.checkpoints import ROOT, sha256
from eval.checkpoints_v5 import GENERATION_MODE_TIR
from eval.checkpoints_v6 import (
    E3_TOTAL_TRAJECTORIES,
    E7_COMPLETION_GATE,
    E7_LEDGER,
    E7_ROLE_NAMES,
    E7_RUN,
    MANIFEST_PATHS_V6,
    ROLE_EQUAL_COMPUTE,
    ROLE_STEP_200,
    equal_compute_step,
    resolved_e7_roles,
    source_locator,
)

PROTOCOL_PATH = ROOT / "eval/diagnostic_protocol_v6.json"
MANIFEST_PATH = ROOT / "eval/diagnostic_freeze_v6.sha256"
LEDGER_PATH = ROOT / "plans/M10_DIAGNOSTIC_FREEZE_V6.md"
M9_CANONICAL_RUN = ROOT / "eval/runs/m9_notool_eval_20260907_175726"
M9_HASHES = M9_CANONICAL_RUN / "hashes.sha256"
V5_CLOSURE_FILE_COUNT = 107
M10_EVALUATOR_FILES = (
    "eval/checkpoints_v6.py",
    "eval/m10_e7_eval.py",
    "eval/test_m10_e7_eval.py",
    "eval/test_checkpoints_v6.py",
    "scripts/m10/run_m10_eval.sh",
    "analysis/m10_compute_axes.py",
    "rl/configs/e7_dynamic_filtering.yaml",
    "rl/launch/e7_dynamic_filtering.py",
    "rl/custom/dynamic_filter_trainer.py",
    "rl/validate_e7_run.py",
)
E7_RUN_FILES = (
    "status.json",
    "resolved_config.yaml",
    "checkpoints/latest_checkpointed_iteration.txt",
    "analysis/formal_completion_gate.json",
    "e7_ledger.jsonl",
    "e7_trainer_active.json",
    "e7_boundary_installation.json",
)
E3_ROLE = "e3_grpo_baseline_step_200"
# plans/M10.md §5/§9 pairing table.  ``after - before`` is the reported delta.
PAIRINGS = (
    {
        "before_role": E3_ROLE,
        "after_role": ROLE_STEP_200,
        "priority": "primary",
        "question": "equal_effective_step_endpoint_delta",
    },
    {
        "before_role": E3_ROLE,
        "after_role": ROLE_EQUAL_COMPUTE,
        "priority": "primary",
        "question": "equal_rollout_compute_delta",
    },
    {
        "before_role": ROLE_EQUAL_COMPUTE,
        "after_role": ROLE_STEP_200,
        "priority": "descriptive",
        "question": "e7_progress_past_the_equal_compute_budget",
    },
)
DYNAMIC_FILTERING_SEMANTICS = {
    "unit": "UID group (one prompt, 8 rollouts)",
    "zero_variance_metric": "scalar score = rm_scores.sum(-1) (answer + 0.1 format), NOT acc",
    "zero_variance_tolerance": "max - min < 1e-8 and std < 1e-8",
    "chunk": "64 fresh prompts x 8 rollouts, identical in shape to one E3 step",
    "max_generation_batches_per_update": 4,
    "selection": "first 64 informative groups in sampling order across chunks",
    "surplus": "recorded as evidence, never trained on, never carried across actor snapshots",
    "underfilled": "fewer than 64 informative groups after 4 chunks fails before the actor update",
    "snapshot": "every candidate of one effective update comes from the same frozen actor; no weight sync between chunks",
    "equal_compute_rule": (
        "the first effective update whose cumulative rollout trajectories reach E3's total of "
        f"{E3_TOTAL_TRAJECTORIES} saves one extra checkpoint outside the rotation; its step is discovered, not fixed"
    ),
    "infrastructure_only_differences": [
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.ref.fsdp_config.param_offload=false",
    ],
    "evaluation_mode": "tir: both E7 roles evaluate through the frozen v2 generation/tool/prompt block, unchanged",
}


def _v5_entries() -> list[str]:
    entries: list[str] = []
    for line in V5_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64:
            raise ValueError(f"invalid parent v5 manifest line: {line}")
        entries.append(relative)
    if len(entries) != V5_CLOSURE_FILE_COUNT or len(set(entries)) != V5_CLOSURE_FILE_COUNT:
        raise ValueError(f"v6 parent must be the exact {V5_CLOSURE_FILE_COUNT}-file frozen v5 closure")
    return entries


def verify_m9_canonical_ledger() -> dict[str, Any]:
    """Re-verify every byte of the reused M9 canonical run (four notool roles, 15,200 trajectories)."""
    checked = 0
    for line in M9_HASHES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        path = M9_CANONICAL_RUN / relative.lstrip("*")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"M9 canonical artifact drifted: {relative}")
        checked += 1
    status = json.loads((M9_CANONICAL_RUN / "status.json").read_text(encoding="utf-8"))
    if status.get("protocol_version") != "toolcredit_diagnostic_v5" or status.get("status") != "notool_evaluation_completed":
        raise ValueError("M9 canonical run is not the completed v5 protocol run")
    if status.get("hashes_sha256") != sha256(M9_HASHES):
        raise ValueError("M9 canonical hash ledger identity drifted")
    return {
        "run_name": M9_CANONICAL_RUN.name,
        "hashes_sha256": sha256(M9_HASHES),
        "status_sha256": sha256(M9_CANONICAL_RUN / "status.json"),
        "verified_artifact_count": checked,
    }


def e7_run_relatives() -> list[str]:
    """Every small E7 run artifact the closure binds, including per-segment recovery ledgers.

    The recovery archive names carry timestamps that are not knowable before the run, so they are
    enumerated from disk rather than written as literals; both checkpoint ledgers are bound too so
    the two roles' effective steps cannot drift after the freeze.
    """
    relatives = [str((E7_RUN / name).relative_to(ROOT)) for name in E7_RUN_FILES]
    for archive in sorted((E7_RUN / "recovery").glob("resume_from_*")):
        if archive.is_dir() and (archive / "recovery.json").is_file():
            relatives.append(str((archive / "recovery.json").relative_to(ROOT)))
    for role in E7_ROLE_NAMES:
        relatives.append(str((source_locator(role) / "e7_ledger.json").relative_to(ROOT)))
    return relatives


def _source_hashes() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in M10_EVALUATOR_FILES}


def protocol_v6(
    frozen_at: str,
    git_head: str,
    roles: dict[str, dict[str, Any]],
    reuse_m7: dict[str, Any],
    reuse_m8: dict[str, Any],
    reuse_m9: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(json.loads(V5_PROTOCOL.read_text(encoding="utf-8")))
    value["protocol_version"] = "toolcredit_diagnostic_v6"
    value["schema_version"] = "toolcredit_diagnostic_schema_v6"
    value["frozen_at"] = frozen_at
    value["git_head_at_freeze"] = git_head
    value["supersedes"] = {
        "protocol_version": "toolcredit_diagnostic_v5",
        "manifest_sha256": sha256(V5_MANIFEST),
        "reason": (
            "M10 completed the E7 dynamic-filtering run; v6 adds its step-200 and equal-compute roles "
            "(both tir), reuses the M7/M8/M9 canonical trajectories byte-for-byte, and changes no "
            "panel/generation/tool/verifier/taxonomy/pairing/bootstrap semantics"
        ),
    }
    equal_step = equal_compute_step()
    for role in E7_ROLE_NAMES:
        entry = copy.deepcopy(roles[role])
        entry["state"] = "validated"
        entry["role"] = role
        entry["generation_mode"] = GENERATION_MODE_TIR
        value["checkpoints"][role] = entry
    gate = value["checkpoint_locator_gate"]
    for role in E7_ROLE_NAMES:
        gate["materialization_manifests"][role] = str(MANIFEST_PATHS_V6[role].relative_to(ROOT))
    gate["rule"] = (
        gate["rule"]
        + "; v6 additionally requires both E7 source roles, the single formal completion gate they share and "
        "their materialized exports to be explicit, non-null, role-correct and hash validated, and requires the "
        "equal-compute locator to be the one equal_compute_step_<N> directory whose N the run ledger flags"
    )
    value["evaluator"]["m10_extension"] = {
        "version": "toolcredit_m10_evaluator_v1",
        "new_roles": list(E7_ROLE_NAMES),
        "source_run": E7_RUN.name,
        "source_sha256": _source_hashes(),
        "dynamic_filtering": copy.deepcopy(DYNAMIC_FILTERING_SEMANTICS),
        "equal_compute_step": equal_step,
        "equal_compute_locator": str(source_locator(ROLE_EQUAL_COMPUTE).relative_to(ROOT)),
        "expected_workload": {
            "new_roles": 2,
            "questions": 760,
            "greedy_trajectories_per_role": 760,
            "sampled_trajectories_per_role": 3040,
            "trajectories_per_role": 3800,
            "total_new_trajectories": 7600,
            "reused_canonical_trajectories": 34200,
        },
        "reused_m7_canonical": reuse_m7,
        "reused_m8_canonical": reuse_m8,
        "reused_m9_canonical": reuse_m9,
        "pairings": [dict(item) for item in PAIRINGS],
        "interpretation_matrix": {
            "delta_step": (
                "normalized AUC over effective steps 0-200 of the fixed-100 greedy validation curve, "
                "E7 minus E3 (analysis/m10_compute_axes.py)"
            ),
            "delta_compute": (
                "paired FULL760 greedy delta between e7_equal_compute and e3_grpo_baseline_step_200 "
                "with source-stratified question bootstrap"
            ),
            "positive": "delta >= +2pt and the 95% CI excludes 0",
            "negative": "delta <= -2pt and the 95% CI excludes 0",
            "otherwise": "unchanged (a CI crossing 0 means undetected, not absent)",
            "cases": {
                "A": "delta_step positive, delta_compute unchanged",
                "B": "both positive",
                "C": "both unchanged",
                "D": "delta_compute negative",
            },
        },
        "generation_approval_state": "AUTHORIZED_BY_APPROVED_M10_PLAN_STEP_5",
        "generation_code": (
            "eval.m10_e7_eval reuses eval.generate.SGLangHTTPServer / build_m7_loop / derive_sample_seed / "
            "load_panel_questions / _sampling_params unchanged; both E7 roles generate through the same tir "
            "agent loop as every other tir role"
        ),
    }
    return value


def closure_files() -> list[str]:
    additions = [
        "plans/M10_DIAGNOSTIC_FREEZE_V6.md",
        "eval/diagnostic_protocol_v6.json",
        "eval/diagnostic_freeze_v5.sha256",
        "eval/build_diagnostic_freeze_v6.py",
        "eval/verify_diagnostic_freeze_v6_semantics.py",
        *M10_EVALUATOR_FILES,
        *(str(MANIFEST_PATHS_V6[role].relative_to(ROOT)) for role in E7_ROLE_NAMES),
        *e7_run_relatives(),
        str(M9_HASHES.relative_to(ROOT)),
        str((M9_CANONICAL_RUN / "status.json").relative_to(ROOT)),
    ]
    ordered = list(dict.fromkeys([*_v5_entries(), *additions]))
    if "eval/diagnostic_freeze_v6.sha256" in ordered:
        raise AssertionError("v6 manifest cannot include itself")
    return ordered


def _expected_entries(roles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for role in E7_ROLE_NAMES:
        entry = copy.deepcopy(roles[role])
        entry["state"] = "validated"
        entry["role"] = role
        entry["generation_mode"] = GENERATION_MODE_TIR
        expected[role] = entry
    return expected


def build(frozen_at: str) -> None:
    verify_v5()
    reuse_m7 = verify_m7_canonical_ledger()
    reuse_m8 = verify_m8_canonical_ledger()
    reuse_m9 = verify_m9_canonical_ledger()
    roles = resolved_e7_roles()
    equal_step = equal_compute_step()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    protocol = protocol_v6(frozen_at, git_head, roles, reuse_m7, reuse_m8, reuse_m9)
    write_json(PROTOCOL_PATH, protocol)
    ledger = json.loads((source_locator(ROLE_EQUAL_COMPUTE) / "e7_ledger.json").read_text(encoding="utf-8"))
    LEDGER_PATH.write_text(
        "# M10 diagnostic protocol v6 freeze ledger\n\n"
        f"- 冻结时间（UTC）：`{frozen_at}`\n"
        f"- Git HEAD：`{git_head}`\n"
        "- 状态：`FROZEN_V6_READY_FOR_E7_GENERATION`；两个 E7 角色的 7,600 条评测轨迹尚未生成。\n"
        f"- parent v5 manifest SHA256：`{sha256(V5_MANIFEST)}`（{V5_CLOSURE_FILE_COUNT}-file closure 保持逐字节不变）。\n"
        f"- v6 唯一语义范围：新增两个 `tir` 角色 `{'`、`'.join(E7_ROLE_NAMES)}`（均由 run `{E7_RUN.name}` 的官方 FSDP "
        "materialization 得到），绑定 M10 evaluator 文件，记录复用的 M7/M8/M9 canonical 产物，并声明两组 E3→E7 primary 配对；"
        "panel/generation/tool/verifier/taxonomy/pairing/bootstrap/persistence/prompt/data-isolation 与 notool 模式块继续沿用 v5。\n"
        f"- 等算力角色：有效 step `{equal_step}`，累计 rollout 轨迹 `{ledger['cumulative_trajectories']}`（E3 全程为 "
        f"{E3_TOTAL_TRAJECTORIES}）。该 step 由 run ledger 判定后从磁盘发现，计划只冻结规则不冻结数值。\n"
        "- 判读：`Δ_step`（fixed-100 greedy 曲线按有效 step 的 0–200 归一化 AUC 之差）与 `Δ_compute`"
        "（等算力 checkpoint 与 E3 step-200 在 FULL760 greedy 上的配对 Δ 与 bootstrap CI）构成 §9 四格矩阵；"
        "greedy paired transitions 仍为 primary diagnostic，sampled matched-index 为 secondary/exploratory。\n"
        + "".join(
            f"- {role} materialization manifest SHA256：`{sha256(MANIFEST_PATHS_V6[role])}`。\n" for role in E7_ROLE_NAMES
        )
        + f"- 复用的 canonical run：M7 `{reuse_m7['run_name']}`（`{reuse_m7['hashes_sha256']}`）、"
        f"M8 `{reuse_m8['run_name']}`（`{reuse_m8['hashes_sha256']}`）、M9 `{reuse_m9['run_name']}`（`{reuse_m9['hashes_sha256']}`）。\n\n"
        "| artifact | SHA256 | Git blob ID |\n|---|---|---|\n"
        f"| `eval/diagnostic_protocol_v6.json` | `{sha256(PROTOCOL_PATH)}` | `{git_blob(PROTOCOL_PATH)}` |\n"
        + "".join(
            f"| `{MANIFEST_PATHS_V6[role].relative_to(ROOT)}` | `{sha256(MANIFEST_PATHS_V6[role])}` | "
            f"`{git_blob(MANIFEST_PATHS_V6[role])}` |\n"
            for role in E7_ROLE_NAMES
        )
        + "\n完整闭包由 `eval/diagnostic_freeze_v6.sha256` 锁定；manifest 不自哈希，verification report 另行记录其 hash。\n",
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        "\n".join(f"{sha256(ROOT / relative)}  {relative}" for relative in closure_files()) + "\n",
        encoding="utf-8",
    )


def verify() -> dict[str, Any]:
    verify_v5()
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid v6 manifest entry: {line}")
        entries[relative] = digest
    if set(entries) != set(closure_files()):
        raise ValueError("v6 manifest closure differs from declared file set")
    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v6 freeze hash mismatch: {relative}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "toolcredit_diagnostic_v6":
        raise ValueError("unexpected v6 protocol identity")
    expected_entries = _expected_entries(resolved_e7_roles())
    for role, entry in expected_entries.items():
        if protocol.get("checkpoints", {}).get(role) != entry:
            raise ValueError(f"v6 runtime binding drifted for {role}")
    v5 = json.loads(V5_PROTOCOL.read_text(encoding="utf-8"))
    frozen_roles = {key: value for key, value in protocol["checkpoints"].items() if key not in E7_ROLE_NAMES}
    if frozen_roles != v5["checkpoints"]:
        raise ValueError("v6 altered a frozen v5 checkpoint role")
    extension = protocol["evaluator"]["m10_extension"]
    if extension["equal_compute_step"] != equal_compute_step():
        raise ValueError("frozen equal-compute step no longer matches the run")
    reuse_m7 = verify_m7_canonical_ledger()
    reuse_m8 = verify_m8_canonical_ledger()
    reuse_m9 = verify_m9_canonical_ledger()
    if (
        extension["reused_m7_canonical"] != reuse_m7
        or extension["reused_m8_canonical"] != reuse_m8
        or extension["reused_m9_canonical"] != reuse_m9
    ):
        raise ValueError("reused M7/M8/M9 canonical ledger identity drifted")
    return {
        "verification": "passed",
        "manifest_sha256": sha256(MANIFEST_PATH),
        "closure_file_count": len(entries),
        "checkpoint_roles": list(protocol["checkpoints"]),
        "equal_compute_step": extension["equal_compute_step"],
        "reused_m7_canonical": reuse_m7,
        "reused_m8_canonical": reuse_m8,
        "reused_m9_canonical": reuse_m9,
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
            parser.error("--frozen-at is required when building v6")
        build(args.frozen_at)
        report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
