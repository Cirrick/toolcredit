"""Prove v5 is only the approved notool mode / four-role / evaluator binding over v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze_v4 import PROTOCOL_PATH as V4_PROTOCOL
from eval.build_diagnostic_freeze_v5 import (
    MANIFEST_PATH,
    NOTOOL_MODE_SEMANTICS,
    PAIRINGS,
    PROTOCOL_PATH,
    ROOT,
    V4_MANIFEST,
    verify,
)
from eval.checkpoints import sha256
from eval.checkpoints_v5 import GENERATION_MODE_NOTOOL, GENERATION_MODE_TIR, NOTOOL_ROLE_NAMES, NOTOOL_WEIGHT_ROLES, ROLE_V5
from eval.generate import expected_keys, load_panel_questions, validate_frozen_runtime_config

EXPECTED_CHANGED_TOP_LEVEL = {
    "checkpoint_locator_gate",
    "checkpoints",
    "evaluator",
    "frozen_at",
    "generation_mode",
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


def semantic_diff(v4: dict[str, Any], v5: dict[str, Any]) -> dict[str, Any]:
    """Pure structural check: v5 == v4 except the declared additive mode/role/evaluator binding."""
    changed = {key for key in set(v4) | set(v5) if v4.get(key) != v5.get(key)}
    if changed != EXPECTED_CHANGED_TOP_LEVEL:
        raise ValueError(f"unexpected v4->v5 semantic diff: {sorted(changed)}")
    if "generation_mode" in v4:
        raise ValueError("v4 must not already carry a generation_mode field")
    if not FROZEN_SEMANTIC_FIELDS.issubset(v4):
        raise ValueError("declared frozen semantic field is absent from v4")
    for field in FROZEN_SEMANTIC_FIELDS:
        if v5[field] != v4[field]:
            raise ValueError(f"v5 drifted frozen v4 semantic field: {field}")
    frozen_roles = {key: value for key, value in v5["checkpoints"].items() if key not in NOTOOL_ROLE_NAMES}
    if frozen_roles != v4["checkpoints"]:
        raise ValueError("v5 altered or removed a frozen v4 checkpoint role")
    for role in NOTOOL_ROLE_NAMES:
        entry = v5["checkpoints"].get(role)
        if not isinstance(entry, dict) or entry.get("state") != "validated":
            raise ValueError(f"v5 does not bind the validated notool role {role}")
        if entry.get("generation_mode") != GENERATION_MODE_NOTOOL or entry.get("weight_role") != NOTOOL_WEIGHT_ROLES[role]:
            raise ValueError(f"v5 notool role {role} has the wrong mode/weight binding")
    for role, entry in frozen_roles.items():
        if "generation_mode" in entry:
            raise ValueError(f"v5 must not stamp generation_mode onto the frozen role {role}")
    mode = v5["generation_mode"]
    if mode.get("default") != GENERATION_MODE_TIR or set(mode.get("modes", {})) != {GENERATION_MODE_TIR, GENERATION_MODE_NOTOOL}:
        raise ValueError("v5 generation_mode block must declare tir (default) and notool")
    notool = mode["modes"][GENERATION_MODE_NOTOOL]
    if notool != NOTOOL_MODE_SEMANTICS:
        raise ValueError("v5 notool mode semantics drifted from the declared block")
    if notool["max_response_tokens_total"] != v4["generation"]["max_response_tokens_total"]:
        raise ValueError("notool response budget differs from the frozen tir budget")
    if mode.get("notool_roles") != dict(NOTOOL_WEIGHT_ROLES):
        raise ValueError("v5 notool role -> weight role map drifted")
    gate4, gate5 = v4["checkpoint_locator_gate"], v5["checkpoint_locator_gate"]
    manifests5 = {key: value for key, value in gate5["materialization_manifests"].items() if key != ROLE_V5}
    if manifests5 != gate4["materialization_manifests"] or ROLE_V5 not in gate5["materialization_manifests"]:
        raise ValueError("v5 locator gate manifests are not v4 plus the E3-NoTool manifest")
    for key in set(gate4) | set(gate5):
        if key in {"materialization_manifests", "rule"}:
            continue
        if gate4.get(key) != gate5.get(key):
            raise ValueError(f"v5 locator gate drifted field: {key}")
    if not str(gate5.get("rule", "")).startswith(str(gate4.get("rule", ""))):
        raise ValueError("v5 locator gate rule does not extend the v4 rule")
    evaluator5 = {key: value for key, value in v5["evaluator"].items() if key != "m9_extension"}
    if evaluator5 != v4["evaluator"]:
        raise ValueError("v5 changed the frozen M7/M8 evaluator binding beyond the m9_extension block")
    extension = v5["evaluator"].get("m9_extension", {})
    if tuple(extension.get("new_roles", ())) != NOTOOL_ROLE_NAMES or extension.get("new_checkpoint_role") != ROLE_V5:
        raise ValueError("m9_extension does not name the four notool roles")
    workload = extension.get("expected_workload", {})
    if workload.get("total_new_trajectories") != 15200 or workload.get("trajectories_per_role") != 3800:
        raise ValueError("m9_extension workload is not 4 x 760 x (1 + 4)")
    pairings = {(item["before_role"], item["after_role"], item["priority"]) for item in extension.get("pairings", [])}
    if pairings != EXPECTED_PAIRINGS:
        raise ValueError("m9_extension pairings differ from plan §3")
    primary = [item for item in extension.get("pairings", []) if item["priority"] == "primary"]
    if len(primary) != 1 or primary[0]["before_role"] != "e3_grpo_baseline_step_200" or primary[0]["after_role"] != ROLE_V5:
        raise ValueError("m9_extension primary pairing is not E3 -> E3-NoTool")
    if v5["supersedes"].get("protocol_version") != "toolcredit_diagnostic_v4":
        raise ValueError("v5 supersedes the wrong protocol")
    return {"changed_top_level_fields": sorted(changed), "unchanged_frozen_semantic_fields": sorted(FROZEN_SEMANTIC_FIELDS)}


def verify_semantic_diff() -> dict[str, Any]:
    closure = verify()
    v4 = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    v5 = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    diff = semantic_diff(v4, v5)
    if v5["supersedes"].get("manifest_sha256") != sha256(V4_MANIFEST):
        raise ValueError("v5 parent manifest hash mismatch")
    config = validate_frozen_runtime_config()
    panel = load_panel_questions()
    keys = expected_keys()
    return {
        **closure,
        **diff,
        "v4_protocol_sha256": sha256(V4_PROTOCOL),
        "v4_manifest_sha256": sha256(V4_MANIFEST),
        "v5_protocol_sha256": sha256(PROTOCOL_PATH),
        "v5_manifest_sha256": sha256(MANIFEST_PATH),
        "parent_v4_closure_file_count": 87,
        "panel_question_count": len(panel),
        "frozen_expected_key_count": len(keys),
        "new_role_expected_key_count": 15200,
        "diagnostic_evidence": config["diagnostic_evidence"],
        "semantic_conclusion": (
            "v5 preserves v4 panel, generation, tool, verifier, taxonomy, pairing, bootstrap, persistence, "
            "prompt and data-isolation semantics exactly; it only adds the notool generation mode, the four "
            "validated notool roles (three over frozen v3 weights, one new materialization), the M9 evaluator "
            "binding, and records the reused M7/M8 canonical artifacts."
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
