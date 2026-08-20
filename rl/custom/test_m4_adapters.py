from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rl.custom.reward import compute_score
from rl.custom.sandbox_tool import ToolCreditSandboxTool
from rl.custom.tool_agent_loop import count_hermes_parse_errors, ensure_tool_count_fields


def test_agent_loop_adds_zero_counts_when_no_tool_is_called() -> None:
    extra_fields = {"raw_prompt": [{"role": "user", "content": "Solve this."}]}
    ensure_tool_count_fields(extra_fields)
    assert extra_fields["tool_call_counts"] == 0
    assert extra_fields["tool_success_count"] == 0
    assert extra_fields["tool_error_count"] == 0
    assert extra_fields["tool_parse_error_count"] == 0


def test_agent_loop_counts_malformed_hermes_calls() -> None:
    malformed = '<tool_call>{"name":"code_interpreter","arguments":{"code":"1+1"}</tool_call>'
    assert count_hermes_parse_errors(malformed, parsed_count=0) == 1
    valid = '<tool_call>{"name":"code_interpreter","arguments":{"code":"1+1"}}</tool_call>'
    assert count_hermes_parse_errors(valid, parsed_count=1) == 0


def test_reward_marks_tool_parse_failure_invalid() -> None:
    result = compute_score(
        data_source="toolcredit_math",
        solution_str="Reasoning. \\boxed{2}",
        ground_truth="2",
        extra_info={"tool_parse_error_count": 1},
    )
    assert result["score"] == 1.0
    assert result["format_ok"] == 0.0
    assert result["invalid"] == 1.0
    assert result["tool_parse_errors"] == 1.0


def test_reward_exposes_strict_components() -> None:
    result = compute_score(
        data_source="toolcredit_math",
        solution_str="Reasoning. \\boxed{2}",
        ground_truth="2",
        extra_info={
            "id": "math_train_1",
            "tool_call_counts": 2,
            "tool_success_count": 1,
            "tool_error_count": 1,
        },
    )
    assert result["score"] == pytest.approx(1.1)
    assert result["acc"] == 1.0
    assert result["format_ok"] == 1.0
    assert result["tool_error_rate"] == 0.5
    assert result["sample_id"] == "math_train_1"


def test_reward_marks_budget_exhaustion_without_final_answer() -> None:
    result = compute_score(
        data_source="toolcredit_math",
        solution_str="still trying",
        ground_truth="2",
        extra_info={"tool_call_counts": 4, "tool_success_count": 4},
    )
    assert result["score"] == 0.0
    assert result["truncated"] == 1.0


def test_sandbox_tool_reuses_hardened_runner_and_records_metrics() -> None:
    tool = ToolCreditSandboxTool(config={"type": "native", "timeout": 2.0}, tool_schema=None)
    agent_data = SimpleNamespace(extra_fields={})
    response, reward, metrics = asyncio.run(
        tool.execute("instance", {"code": "x = 20\nx + 22"}, agent_data=agent_data)
    )
    assert response.text.strip() == "42"
    assert reward == 0.0
    assert metrics["tool_success"] == 1.0
    assert agent_data.extra_fields == {
        "tool_call_counts": 1,
        "tool_success_count": 1,
        "tool_error_count": 0,
    }


def test_sandbox_tool_records_execution_error() -> None:
    tool = ToolCreditSandboxTool(config={"type": "native", "timeout": 2.0}, tool_schema=None)
    agent_data = SimpleNamespace(extra_fields={})
    response, _, metrics = asyncio.run(
        tool.execute("instance", {"code": "raise RuntimeError('boom')"}, agent_data=agent_data)
    )
    assert response.text.startswith("error:")
    assert metrics["tool_error"] == 1.0
    assert agent_data.extra_fields["tool_error_count"] == 1
