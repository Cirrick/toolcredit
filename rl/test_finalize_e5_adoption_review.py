from __future__ import annotations

from rl.finalize_e5_adoption_review import join_rows, wilson_interval


def _fixtures() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    templates = []
    keys = []
    blinded = []
    candidates = []
    for index in range(15):
        review_id = f"TC-ADOPT-{index + 1:03d}"
        candidate_id = f"candidate-{index:02d}"
        templates.append(
            {
                "review_case_id": review_id,
                "human_label": "false_adoption" if index == 6 else "true_adoption",
                "human_reason": "user-confirmed decision",
            }
        )
        keys.append({"review_case_id": review_id, "candidate_id": candidate_id})
        blinded.append({"review_case_id": review_id})
        candidates.append({"candidate_id": candidate_id})
    return templates, keys, blinded, candidates


def test_join_rows_requires_exact_unique_15_case_alignment() -> None:
    templates, keys, blinded, candidates = _fixtures()
    joined = join_rows(templates, keys, blinded, candidates)
    assert len(joined) == len({row["candidate_id"] for row in joined}) == 15
    assert sum(row["human_label"] == "true_adoption" for row in joined) == 14
    assert sum(row["human_label"] == "false_adoption" for row in joined) == 1
    assert all(row["adjudicator"] == "user" for row in joined)

    keys[-1] = dict(keys[0])
    try:
        join_rows(templates, keys, blinded, candidates)
    except ValueError as exc:
        assert "duplicate review_case_id" in str(exc)
    else:
        raise AssertionError("duplicate review case was accepted")


def test_wilson_interval_for_fourteen_of_fifteen() -> None:
    lower, upper = wilson_interval(14, 15)
    assert 0.68 < lower < 0.72
    assert 0.98 < upper < 1.0
