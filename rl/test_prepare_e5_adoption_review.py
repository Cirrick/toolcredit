from __future__ import annotations

from rl.prepare_e5_adoption_review import FORBIDDEN_KEYS, _assert_blinded, _relevant_excerpt


def test_relevant_excerpt_is_compact_and_keeps_match() -> None:
    text = "a" * 1000 + "used result 42 in equation" + "b" * 1000
    excerpt = _relevant_excerpt(text, "42")
    assert "42" in excerpt
    assert "earlier span omitted" in excerpt
    assert "later span omitted" in excerpt
    assert len(excerpt) < len(text)


def test_blinded_schema_rejects_forbidden_fields() -> None:
    item = {
        "review_case_id": "TC-ADOPT-001",
        "minimal_context_before_tool": "call context",
        "visible_post_truncation_tool_response": "42",
        "relevant_later_assistant_span": "using 42 gives the result",
    }
    _assert_blinded(item)
    for forbidden in FORBIDDEN_KEYS:
        leaked = {**item, forbidden: "leak"}
        try:
            _assert_blinded(leaked)
        except ValueError:
            pass
        else:
            raise AssertionError(f"forbidden field was accepted: {forbidden}")
