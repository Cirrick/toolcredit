"""Prove v6 is only the approved two-role / evaluator binding over v5.

The E7 roles are ``tir``, so unlike the v4 -> v5 bump this one adds no generation mode: the whole
``generation_mode`` block, including the ``notool`` semantics and the notool role map, must come
through byte-for-byte.  That makes ``generation_mode`` a *frozen* field here, which is the sharpest
single check that v6 did not quietly reshape how anything is generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze_v5 import PROTOCOL_PATH as V5_PROTOCOL
from eval.build_diagnostic_freeze_v6 import (
    DYNAMIC_FILTERING_SEMANTICS,
    MANIFEST_PATH,
    PAIRINGS,
    PROTOCOL_PATH,
    ROOT,
    V5_CLOSURE_FILE_COUNT,
    V5_MANIFEST,
    verify,
)
from eval.checkpoints import sha256
from eval.checkpoints_v5 import GENERATION_MODE_TIR
from eval.checkpoints_v6 import E3_TOTAL_TRAJECTORIES, E7_ROLE_NAMES, ROLE_EQUAL_COMPUTE, ROLE_STEP_200
from eval.generate import expected_keys, load_panel_questions, validate_frozen_runtime_config

E3_ROLE = "e3_grpo_baseline_step_200"
NEW_ROLE_KEY_COUNT = 7600
EXPECTED_CHANGED_TOP_LEVEL = {
    "checkpoint_locator_gate",
    "checkpoints",
    "evaluator",
    "frozen_at",
    "git_head_at_freeze",
    "protocol_version",
    "schema_version",
    "supersedes",
}
FROZEN_SEMANTIC_FIELDS = {
    "bootstrap",
    "data_isolation",
    "file_sha256",
    "generation",
    # v6 adds no mode: the whole v5 generation_mode block (tir default, notool semantics, notool
    # role map) must survive untouched.
    "generation_mode",
    "pairing_and_transitions",
    "panel",
    "persistence",
    "prompt",
    "protocol_change_rule",
    "taxonomy",
    "tool",
    "verifier",
}
EXPECTED_PAIRINGS = {(item["before_role"], item["after_role"], item["priority"]) for item in PAIRINGS}


def semantic_diff(v5: dict[str, Any], v6: dict[str, Any]) -> dict[str, Any]:
    """Pure structural check: v6 == v5 except the declared additive role/evaluator binding."""
    changed = {key for key in set(v5) | set(v6) if v5.get(key) != v6.get(key)}
    if changed != EXPECTED_CHANGED_TOP_LEVEL:
        raise ValueError(f"unexpected v5->v6 semantic diff: {sorted(changed)}")
    if not FROZEN_SEMANTIC_FIELDS.issubset(v5):
        raise ValueError("declared frozen semantic field is absent from v5")
    for field in FROZEN_SEMANTIC_FIELDS:
        if v6[field] != v5[field]:
            raise ValueError(f"v6 drifted frozen v5 semantic field: {field}")

    frozen_roles = {key: value for key, value in v6["checkpoints"].items() if key not in E7_ROLE_NAMES}
    if frozen_roles != v5["checkpoints"]:
        raise ValueError("v6 altered or removed a frozen v5 checkpoint role")
    for role in E7_ROLE_NAMES:
        entry = v6["checkpoints"].get(role)
        if not isinstance(entry, dict) or entry.get("state") != "validated":
            raise ValueError(f"v6 does not bind the validated E7 role {role}")
        if entry.get("generation_mode") != GENERATION_MODE_TIR or entry.get("weight_role") != role:
            raise ValueError(f"v6 E7 role {role} has the wrong mode/weight binding")
        if entry.get("run_name") != v6["evaluator"]["m10_extension"]["source_run"]:
            raise ValueError(f"v6 E7 role {role} does not come from the declared source run")
    if v6["checkpoints"][ROLE_STEP_200]["source_locator"] == v6["checkpoints"][ROLE_EQUAL_COMPUTE]["source_locator"]:
        raise ValueError("the two E7 roles must not share one source checkpoint")

    gate5, gate6 = v5["checkpoint_locator_gate"], v6["checkpoint_locator_gate"]
    manifests6 = {key: value for key, value in gate6["materialization_manifests"].items() if key not in E7_ROLE_NAMES}
    if manifests6 != gate5["materialization_manifests"]:
        raise ValueError("v6 locator gate manifests are not v5 plus the two E7 manifests")
    if any(role not in gate6["materialization_manifests"] for role in E7_ROLE_NAMES):
        raise ValueError("v6 locator gate does not bind both E7 materialization manifests")
    for key in set(gate5) | set(gate6):
        if key in {"materialization_manifests", "rule"}:
            continue
        if gate5.get(key) != gate6.get(key):
            raise ValueError(f"v6 locator gate drifted field: {key}")
    if not str(gate6.get("rule", "")).startswith(str(gate5.get("rule", ""))):
        raise ValueError("v6 locator gate rule does not extend the v5 rule")

    evaluator6 = {key: value for key, value in v6["evaluator"].items() if key != "m10_extension"}
    if evaluator6 != v5["evaluator"]:
        raise ValueError("v6 changed the frozen M7/M8/M9 evaluator binding beyond the m10_extension block")
    extension = v6["evaluator"].get("m10_extension", {})
    if tuple(extension.get("new_roles", ())) != E7_ROLE_NAMES:
        raise ValueError("m10_extension does not name the two E7 roles")
    if extension.get("dynamic_filtering") != DYNAMIC_FILTERING_SEMANTICS:
        raise ValueError("m10_extension dynamic-filtering semantics drifted from the declared block")
    if extension["dynamic_filtering"]["zero_variance_metric"].startswith("acc"):
        raise ValueError("E7 must filter on the scalar score that enters the advantage, not on acc")
    workload = extension.get("expected_workload", {})
    if workload.get("total_new_trajectories") != NEW_ROLE_KEY_COUNT or workload.get("trajectories_per_role") != 3800:
        raise ValueError("m10_extension workload is not 2 x 760 x (1 + 4)")
    step = extension.get("equal_compute_step")
    if not isinstance(step, int) or not 1 <= step <= 200:
        raise ValueError(f"m10_extension equal-compute step is not a valid effective step: {step}")
    locator = str(extension.get("equal_compute_locator", ""))
    if not locator.endswith(f"equal_compute_step_{step}"):
        raise ValueError("m10_extension equal-compute locator does not match its declared step")
    ledger = v6["checkpoints"][ROLE_EQUAL_COMPUTE]["dynamic_filtering"]
    if ledger.get("effective_step") != step:
        raise ValueError("equal-compute role step disagrees with the m10_extension declaration")
    if int(ledger.get("cumulative_trajectories", 0)) < E3_TOTAL_TRAJECTORIES:
        raise ValueError("equal-compute role did not reach the E3 rollout budget")

    pairings = {(item["before_role"], item["after_role"], item["priority"]) for item in extension.get("pairings", [])}
    if pairings != EXPECTED_PAIRINGS:
        raise ValueError("m10_extension pairings differ from plan §5")
    primary = [item for item in extension.get("pairings", []) if item["priority"] == "primary"]
    if len(primary) != 2 or {item["before_role"] for item in primary} != {E3_ROLE}:
        raise ValueError("m10_extension must declare exactly two primary pairings, both from E3")
    if {item["after_role"] for item in primary} != set(E7_ROLE_NAMES):
        raise ValueError("the two primary pairings must target the two E7 roles")
    matrix = extension.get("interpretation_matrix", {})
    if set(matrix.get("cases", {})) != {"A", "B", "C", "D"}:
        raise ValueError("m10_extension interpretation matrix is not the plan §9 four-cell table")
    for axis in ("delta_step", "delta_compute"):
        if not str(matrix.get(axis, "")).strip():
            raise ValueError(f"m10_extension interpretation matrix does not define {axis}")

    if v6["supersedes"].get("protocol_version") != "toolcredit_diagnostic_v5":
        raise ValueError("v6 supersedes the wrong protocol")
    return {
        "changed_top_level_fields": sorted(changed),
        "unchanged_frozen_semantic_fields": sorted(FROZEN_SEMANTIC_FIELDS),
        "equal_compute_step": step,
    }


def verify_semantic_diff() -> dict[str, Any]:
    closure = verify()
    v5 = json.loads(V5_PROTOCOL.read_text(encoding="utf-8"))
    v6 = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    diff = semantic_diff(v5, v6)
    if v6["supersedes"].get("manifest_sha256") != sha256(V5_MANIFEST):
        raise ValueError("v6 parent manifest hash mismatch")
    config = validate_frozen_runtime_config()
    panel = load_panel_questions()
    keys = expected_keys()
    return {
        **closure,
        **diff,
        "v5_protocol_sha256": sha256(V5_PROTOCOL),
        "v5_manifest_sha256": sha256(V5_MANIFEST),
        "v6_protocol_sha256": sha256(PROTOCOL_PATH),
        "v6_manifest_sha256": sha256(MANIFEST_PATH),
        "parent_v5_closure_file_count": V5_CLOSURE_FILE_COUNT,
        "panel_question_count": len(panel),
        "frozen_expected_key_count": len(keys),
        "new_role_expected_key_count": NEW_ROLE_KEY_COUNT,
        "diagnostic_evidence": config["diagnostic_evidence"],
        "semantic_conclusion": (
            "v6 preserves v5 panel, generation, generation_mode, tool, verifier, taxonomy, pairing, bootstrap, "
            "persistence, prompt and data-isolation semantics exactly; it only adds the two validated E7 tir "
            "roles from one run, the M10 evaluator binding, the two E3 -> E7 primary pairings and the plan §9 "
            "interpretation matrix, and records the reused M7/M8/M9 canonical artifacts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_semantic_diff()
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
