"""Prove v4 is only the approved E5-v2 role/materialization/evaluator binding over v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze_v3 import PROTOCOL_PATH as V3_PROTOCOL
from eval.build_diagnostic_freeze_v4 import MANIFEST_PATH, PROTOCOL_PATH, ROOT, V3_MANIFEST, verify
from eval.checkpoints import sha256
from eval.checkpoints_v4 import ROLE_V4
from eval.generate import expected_keys, load_panel_questions, validate_frozen_runtime_config

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
    "pairing_and_transitions",
    "panel",
    "persistence",
    "prompt",
    "protocol_change_rule",
    "taxonomy",
    "tool",
    "verifier",
}


def semantic_diff(v3: dict[str, Any], v4: dict[str, Any]) -> dict[str, Any]:
    """Pure structural check: v4 == v3 except the declared additive role/evaluator binding."""
    changed = {key for key in set(v3) | set(v4) if v3.get(key) != v4.get(key)}
    if changed != EXPECTED_CHANGED_TOP_LEVEL:
        raise ValueError(f"unexpected v3->v4 semantic diff: {sorted(changed)}")
    if not FROZEN_SEMANTIC_FIELDS.issubset(v3):
        raise ValueError("declared frozen semantic field is absent from v3")
    for field in FROZEN_SEMANTIC_FIELDS:
        if v4[field] != v3[field]:
            raise ValueError(f"v4 drifted frozen v3 semantic field: {field}")
    frozen_roles = {key: value for key, value in v4["checkpoints"].items() if key != ROLE_V4}
    if frozen_roles != v3["checkpoints"]:
        raise ValueError("v4 altered or removed a frozen v3 checkpoint role")
    if ROLE_V4 not in v4["checkpoints"] or v4["checkpoints"][ROLE_V4].get("state") != "validated":
        raise ValueError("v4 does not bind the validated E5-v2 role")
    gate3, gate4 = v3["checkpoint_locator_gate"], v4["checkpoint_locator_gate"]
    manifests4 = {key: value for key, value in gate4["materialization_manifests"].items() if key != ROLE_V4}
    if manifests4 != gate3["materialization_manifests"] or ROLE_V4 not in gate4["materialization_manifests"]:
        raise ValueError("v4 locator gate manifests are not v3 plus the E5-v2 manifest")
    for key in set(gate3) | set(gate4):
        if key in {"materialization_manifests", "rule"}:
            continue
        if gate3.get(key) != gate4.get(key):
            raise ValueError(f"v4 locator gate drifted field: {key}")
    if not str(gate4.get("rule", "")).startswith(str(gate3.get("rule", ""))):
        raise ValueError("v4 locator gate rule does not extend the v3 rule")
    evaluator4 = {key: value for key, value in v4["evaluator"].items() if key != "m8_extension"}
    if evaluator4 != v3["evaluator"]:
        raise ValueError("v4 changed the frozen M7 evaluator binding beyond the m8_extension block")
    extension = v4["evaluator"].get("m8_extension", {})
    if extension.get("new_role") != ROLE_V4:
        raise ValueError("m8_extension does not name the E5-v2 role")
    if extension.get("expected_workload", {}).get("total_new_trajectories") != 3800:
        raise ValueError("m8_extension workload is not 760 x (1 + 4)")
    pairings = {(item["before_role"], item["after_role"], item["priority"]) for item in extension.get("pairings", [])}
    if pairings != {
        ("e3_grpo_baseline_step_200", ROLE_V4, "primary"),
        ("e5_turn_credit_step_200", ROLE_V4, "secondary"),
    }:
        raise ValueError("m8_extension pairings differ from plan §8 step 6")
    if v4["supersedes"].get("protocol_version") != "toolcredit_diagnostic_v3":
        raise ValueError("v4 supersedes the wrong protocol")
    return {"changed_top_level_fields": sorted(changed), "unchanged_frozen_semantic_fields": sorted(FROZEN_SEMANTIC_FIELDS)}


def verify_semantic_diff() -> dict[str, Any]:
    closure = verify()
    v3 = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
    v4 = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    diff = semantic_diff(v3, v4)
    if v4["supersedes"].get("manifest_sha256") != sha256(V3_MANIFEST):
        raise ValueError("v4 parent manifest hash mismatch")
    config = validate_frozen_runtime_config()
    panel = load_panel_questions()
    keys = expected_keys()
    return {
        **closure,
        **diff,
        "v3_protocol_sha256": sha256(V3_PROTOCOL),
        "v3_manifest_sha256": sha256(V3_MANIFEST),
        "v4_protocol_sha256": sha256(PROTOCOL_PATH),
        "v4_manifest_sha256": sha256(MANIFEST_PATH),
        "parent_v3_closure_file_count": 66,
        "panel_question_count": len(panel),
        "frozen_expected_key_count": len(keys),
        "new_role_expected_key_count": 3800,
        "diagnostic_evidence": config["diagnostic_evidence"],
        "semantic_conclusion": (
            "v4 preserves v3 panel, generation, tool, verifier, taxonomy, pairing, bootstrap, persistence, "
            "prompt and data-isolation semantics exactly; it only adds the validated E5-v2 role, its "
            "materialization and the M8 evaluator binding, and records the reused M7 canonical artifacts."
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
