"""M9 / E3-NoTool run gates — plans/M9.md §6 fixture N plus the rollout invariants.

Fixture N (tag detection): complete, unpaired and absent ``<tool_call>`` tags are counted
correctly, and detection reads only the response text — it never touches scoring.
The remaining tests drive the rollout validator over a synthetic run directory so the
fail-closed paths are exercised without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.validate_e3notool_run import (
    TOOL_TAG_GATE_FROM_STEP,
    TOOL_TAG_RATE_LIMIT,
    behavior_metrics,
    prompt_tool_schema_markers,
    tool_tag_gate,
    tool_tag_stats,
    validate_row_is_tool_free,
    validate_rollouts_notool,
    write_prompt_audit,
    zero_variance_group_fraction,
)

COMPLETE_CALL = 'Let me compute.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "1+1"}}\n</tool_call>'
UNPAIRED_CALL = 'Let me compute.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "1+1"'
NO_CALL = "Purely verbal reasoning, so the answer is \\boxed{2}."


# --------------------------------------------------------------------------- #
# Fixture N — tag detection
# --------------------------------------------------------------------------- #


def test_fixture_n_complete_block_counts_once() -> None:
    stats = tool_tag_stats(COMPLETE_CALL)
    assert stats["tool_tag_emitted"] is True
    assert stats["tool_tag_count"] == 1
    assert stats["complete_tool_call_blocks"] == 1
    assert stats["unpaired_tool_call_tags"] == 0


def test_fixture_n_unpaired_opening_tag_still_counts() -> None:
    stats = tool_tag_stats(UNPAIRED_CALL)
    assert stats["tool_tag_emitted"] is True
    assert stats["tool_tag_count"] == 1
    assert stats["complete_tool_call_blocks"] == 0
    assert stats["unpaired_tool_call_tags"] == 1


def test_fixture_n_clean_response_has_no_tags() -> None:
    stats = tool_tag_stats(NO_CALL)
    assert stats["tool_tag_emitted"] is False
    assert stats["tool_tag_count"] == 0
    assert stats["other_hermes_tag_count"] == 0


def test_fixture_n_counts_repeated_and_mixed_tags() -> None:
    stats = tool_tag_stats(COMPLETE_CALL + "\n" + COMPLETE_CALL + "\n<tool_call>")
    assert stats["tool_tag_count"] == 3
    assert stats["complete_tool_call_blocks"] == 2
    assert stats["unpaired_tool_call_tags"] == 1
    assert tool_tag_stats("<tool_response>ok</tool_response>")["other_hermes_tag_count"] == 2


def test_fixture_n_detection_does_not_change_scoring() -> None:
    """The detector is pure text analysis; the reward never sees it."""
    from rl.custom.reward import compute_score

    scored = compute_score(
        data_source="toolcredit_math",
        solution_str=COMPLETE_CALL + " \\boxed{2}",
        ground_truth="2",
        extra_info={},
    )
    assert scored["score"] == pytest.approx(1.1)
    assert scored["n_tool_calls"] == 0.0
    assert tool_tag_stats(COMPLETE_CALL + " \\boxed{2}")["tool_tag_emitted"] is True


def test_prompt_marker_detection() -> None:
    assert prompt_tool_schema_markers("user\nWhat is 1+1?\nassistant\n") == []
    markers = prompt_tool_schema_markers("system\n# Tools\n<tools>{...code_interpreter...}</tools>\nuser\n")
    assert set(markers) == {"<tools>", "</tools>", "code_interpreter", "# Tools"}


# --------------------------------------------------------------------------- #
# rollout invariants
# --------------------------------------------------------------------------- #


def _row(step: int, sample_id: str, *, output: str = NO_CALL, **overrides) -> dict:
    row = {
        "step": step,
        "sample_id": sample_id,
        "input": f"user\nQuestion {sample_id}\nassistant\n<think>\n\n</think>\n\n",
        "output": output,
        "score": 1.1,
        "acc": 1.0,
        "format_ok": 1.0,
        "invalid": 0.0,
        "n_tool_calls": 0.0,
        "tool_error_rate": 0.0,
        "tool_parse_errors": 0.0,
        "truncated": 0.0,
    }
    row.update(overrides)
    return row


def _write_run(tmp_path: Path, steps: int, *, tagged_per_step: dict[int, int] | None = None) -> Path:
    run_dir = tmp_path / "e3notool_smoke_run"
    train = run_dir / "predictions/train"
    train.mkdir(parents=True)
    (run_dir / "tensorboard").mkdir(parents=True)
    tagged_per_step = tagged_per_step or {}
    for step in range(1, steps + 1):
        rows = []
        tagged = tagged_per_step.get(step, 0)
        for prompt in range(64):
            for rollout in range(8):
                index = prompt * 8 + rollout
                output = COMPLETE_CALL if index < tagged else NO_CALL
                score = 1.1 if rollout % 2 == 0 else 0.0
                rows.append(
                    _row(step, f"math_train_{prompt:06d}", output=output, score=score, acc=float(rollout % 2 == 0))
                )
        (train / f"{step}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    return run_dir


def test_validate_row_rejects_tool_activity() -> None:
    validate_row_is_tool_free(_row(1, "math_train_000000"))
    with pytest.raises(ValueError, match="nonzero n_tool_calls"):
        validate_row_is_tool_free(_row(1, "math_train_000000", n_tool_calls=1.0))
    with pytest.raises(ValueError, match="nonzero truncated"):
        validate_row_is_tool_free(_row(1, "math_train_000000", truncated=1.0))
    with pytest.raises(ValueError, match="nonzero tool_call_counts"):
        validate_row_is_tool_free(_row(1, "math_train_000000", tool_call_counts=2.0))
    with pytest.raises(ValueError, match="tool schema markers"):
        validate_row_is_tool_free(
            _row(1, "math_train_000000", input="system\n# Tools\n<tools>code_interpreter</tools>\nuser\n")
        )


def test_validate_rollouts_counts_groups_and_tags(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, steps=2, tagged_per_step={1: 64, 2: 0})
    report = validate_rollouts_notool(run_dir, expected_steps=2)
    assert report["steps"] == [1, 2]
    assert report["trajectory_count"] == 1024 == report["expected_trajectory_count"]
    assert report["tool_tag_rate_by_step"] == {"1": 0.125, "2": 0.0}
    assert report["tool_tag_trajectory_rate"] == pytest.approx(64 / 1024)
    assert report["accuracy"] == pytest.approx(0.5)


def test_validate_rollouts_rejects_incomplete_groups(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, steps=1)
    path = run_dir / "predictions/train/1.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="instead of 512 rollouts"):
        validate_rollouts_notool(run_dir, expected_steps=1)


def test_validate_rollouts_rejects_missing_steps(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, steps=2)
    with pytest.raises(ValueError, match="incomplete"):
        validate_rollouts_notool(run_dir, expected_steps=3)


def test_tool_tag_gate_only_applies_after_step_50() -> None:
    curve = {"1": 0.9, "50": 0.9, "51": 0.2, "60": 0.01}
    gate = tool_tag_gate(curve)
    assert gate["limit"] == TOOL_TAG_RATE_LIMIT
    assert gate["applies_after_step"] == TOOL_TAG_GATE_FROM_STEP
    assert gate["pause_required"] is True
    assert set(gate["violating_steps"]) == {"51"}
    assert tool_tag_gate({"1": 0.9, "51": 0.0})["pause_required"] is False


def test_behavior_and_zero_variance_metrics(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, steps=1)
    behavior = behavior_metrics(run_dir, [1])
    assert behavior["rows"] == 512
    assert behavior["accuracy"] == pytest.approx(0.5)
    assert behavior["tool_tag_trajectory_rate"] == 0.0
    variance = zero_variance_group_fraction(run_dir, [1])
    assert variance["groups"] == 64
    assert variance["zero_variance_fraction"] == 0.0


def test_prompt_audit_writes_ten_schema_free_prompts(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, steps=1)
    path = write_prompt_audit(run_dir, [1])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 10
    assert all(row["tool_schema_markers"] == [] for row in rows)
    assert (run_dir / "analysis/prompt_audit.md").is_file()


def test_prompt_audit_fails_closed_on_a_tool_schema(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, steps=1)
    path = run_dir / "predictions/train/1.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows[0]["input"] = "system\n# Tools\n<tools>code_interpreter</tools>\nuser\n"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tool schema"):
        write_prompt_audit(run_dir, [1])


def test_tool_tag_rate_by_step_matches_the_rollout_report(tmp_path: Path) -> None:
    from rl.validate_e3notool_run import tool_tag_rate_by_step

    run_dir = _write_run(tmp_path, steps=2, tagged_per_step={1: 128, 2: 8})
    curve = tool_tag_rate_by_step(run_dir, [1, 2])
    assert curve == {"1": 0.25, "2": 0.015625}
    assert curve == validate_rollouts_notool(run_dir, expected_steps=2)["tool_tag_rate_by_step"]
