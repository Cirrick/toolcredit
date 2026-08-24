from __future__ import annotations

from eval.verify_diagnostic_freeze_v2_semantics import (
    EXPECTED_CHANGED_TOP_LEVEL,
    verify_semantic_diff,
)


def test_v1_to_v2_semantic_diff_is_exact_and_locators_validate() -> None:
    report = verify_semantic_diff()
    assert report["verification"] == "passed"
    assert set(report["changed_top_level_fields"]) == EXPECTED_CHANGED_TOP_LEVEL
    assert report["v2_closure_file_count"] == 37
    assert report["checkpoint_roles"]["raw"]["state"] == "validated"
    assert report["checkpoint_roles"]["sft"]["state"] == "validated"
    assert report["checkpoint_roles"]["e3"]["state"] == "validated"
    assert report["checkpoint_roles"]["e5"]["state"] == "future"
