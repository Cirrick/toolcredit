"""Prove v3 is only the approved E5/materialization/evaluator binding over v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze_v3 import (
    MANIFEST_PATH,
    PROTOCOL_PATH,
    ROOT,
    V2_MANIFEST,
    V2_PROTOCOL,
    verify,
)
from eval.checkpoints import sha256
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


def verify_semantic_diff() -> dict[str, Any]:
    closure = verify()
    v2 = json.loads(V2_PROTOCOL.read_text(encoding="utf-8"))
    v3 = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    changed = {key for key in set(v2) | set(v3) if v2.get(key) != v3.get(key)}
    if changed != EXPECTED_CHANGED_TOP_LEVEL:
        raise ValueError(f"unexpected v2->v3 semantic diff: {sorted(changed)}")
    if not FROZEN_SEMANTIC_FIELDS.issubset(v2):
        raise ValueError("declared frozen semantic field is absent from v2")
    for field in FROZEN_SEMANTIC_FIELDS:
        if v3[field] != v2[field]:
            raise ValueError(f"v3 drifted frozen v2 semantic field: {field}")
    if not isinstance(v3.get("supersedes"), dict):
        raise TypeError("supersedes must be a mapping")
    if v3["supersedes"].get("protocol_version") != "toolcredit_diagnostic_v2":
        raise ValueError("v3 supersedes the wrong protocol")
    if v3["supersedes"].get("manifest_sha256") != sha256(V2_MANIFEST):
        raise ValueError("v3 parent manifest hash mismatch")
    evaluator = v3.get("evaluator", {})
    evidence = evaluator.get("diagnostic_evidence", {})
    if evidence.get("greedy_paired_transitions") != "primary_diagnostic":
        raise ValueError("v3 does not bind greedy transitions as primary evidence")
    if evidence.get("sampled_matched_index_transitions") != "secondary_exploratory":
        raise ValueError("v3 does not bind sampled transitions as exploratory")
    if evaluator.get("agent_loop", {}).get("advantage_wrapper_installed") is not False:
        raise ValueError("M7 evaluator must not install the E5 advantage wrapper")
    config = validate_frozen_runtime_config()
    panel = load_panel_questions()
    keys = expected_keys()
    return {
        **closure,
        "v2_protocol_sha256": sha256(V2_PROTOCOL),
        "v2_manifest_sha256": sha256(V2_MANIFEST),
        "v3_protocol_sha256": sha256(PROTOCOL_PATH),
        "v3_manifest_sha256": sha256(MANIFEST_PATH),
        "parent_v2_closure_file_count": 37,
        "panel_question_count": len(panel),
        "expected_key_count": len(keys),
        "changed_top_level_fields": sorted(changed),
        "unchanged_frozen_semantic_fields": sorted(FROZEN_SEMANTIC_FIELDS),
        "diagnostic_evidence": config["diagnostic_evidence"],
        "gpu_generation_performed": False,
        "semantic_conclusion": (
            "v3 preserves v2 panel, generation, tool, verifier, taxonomy, pairing, bootstrap, "
            "persistence, prompt, and data-isolation semantics exactly; it only resolves E5, "
            "binds role-validated inference materializations, and freezes the unified evaluator."
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
