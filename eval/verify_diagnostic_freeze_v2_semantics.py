"""Verify the frozen v2 closure and its exact semantic delta from v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from eval.build_diagnostic_freeze_v2 import ROOT, verify
from rl.launch.e5_turn_credit import validate_freeze_bundle

V1_PATH = ROOT / "eval/diagnostic_protocol_v1.json"
V2_PATH = ROOT / "eval/diagnostic_protocol_v2.json"
V1_MANIFEST = ROOT / "eval/diagnostic_freeze_v1.sha256"
V2_MANIFEST = ROOT / "eval/diagnostic_freeze_v2.sha256"
EXPECTED_CHANGED_TOP_LEVEL = {
    "checkpoint_locator_gate",
    "checkpoints",
    "frozen_at",
    "protocol_version",
    "schema_version",
    "supersedes",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_semantic_diff() -> dict[str, Any]:
    verify()
    closure = validate_freeze_bundle(V2_MANIFEST)
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))

    top_level = set(v1) | set(v2)
    changed = {key for key in top_level if v1.get(key) != v2.get(key)}
    if changed != EXPECTED_CHANGED_TOP_LEVEL:
        raise ValueError(
            f"unexpected v1->v2 semantic diff: {sorted(changed)} != "
            f"{sorted(EXPECTED_CHANGED_TOP_LEVEL)}"
        )
    shared_unchanged = sorted(top_level - changed)
    if any(v1[key] != v2[key] for key in shared_unchanged):
        raise AssertionError("a supposedly unchanged diagnostic field drifted")

    if v1["protocol_version"] != "toolcredit_diagnostic_v1":
        raise ValueError("unexpected v1 protocol identity")
    if v2["protocol_version"] != "toolcredit_diagnostic_v2":
        raise ValueError("unexpected v2 protocol identity")
    if v2["git_head_at_freeze"] != v1["git_head_at_freeze"]:
        raise ValueError("v2 unexpectedly changed the frozen Git HEAD")
    if v1["checkpoints"]["e3"] != (
        "rl/runs/e3_grpo_baseline_20260819_093123/global_step_200"
    ):
        raise ValueError("v1 no longer preserves the recorded bad E3 locator")
    e3_v2 = v2["checkpoints"]["e3"]
    if e3_v2["locator"] != (
        "rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200"
    ) or e3_v2["role"] != "e3_grpo_baseline_step_200":
        raise ValueError("v2 does not contain the approved corrected E3 role/locator")
    if v2["checkpoints"]["sft"]["locator"] != v1["checkpoints"]["sft"]:
        raise ValueError("v2 changed the frozen SFT start locator")
    if v2["checkpoints"]["e5"]["state"] != "future" or v2["checkpoints"]["e5"]["locator"] is not None:
        raise ValueError("v2 pre-training E5 role must remain future/null")
    supersedes = v2["supersedes"]
    if supersedes["protocol_version"] != "toolcredit_diagnostic_v1":
        raise ValueError("v2 supersedes metadata points to the wrong protocol")
    if supersedes["manifest_sha256"] != _sha256(V1_MANIFEST):
        raise ValueError("v2 supersedes metadata has the wrong v1 manifest hash")

    return {
        "verification": "passed",
        "v1_protocol_sha256": _sha256(V1_PATH),
        "v1_manifest_sha256": _sha256(V1_MANIFEST),
        "v2_protocol_sha256": _sha256(V2_PATH),
        "v2_manifest_sha256": _sha256(V2_MANIFEST),
        "v2_closure_file_count": closure["closure_file_count"],
        "panel_question_count": closure["panel_question_count"],
        "changed_top_level_fields": sorted(changed),
        "unchanged_top_level_fields": shared_unchanged,
        "checkpoint_roles": closure["checkpoint_roles"],
        "semantic_conclusion": (
            "Panel, taxonomy, generation, tool, verifier, persistence, pairing, bootstrap, "
            "and data-isolation semantics are byte-equivalent at the parsed JSON level; v2 "
            "only versions the protocol and replaces checkpoint descriptions with the "
            "approved role-aware fail-closed locator gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_semantic_diff()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
