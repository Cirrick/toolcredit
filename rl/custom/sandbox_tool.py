"""veRL native tool adapter for ToolCredit's single hardened Python sandbox."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from transformers.utils import get_json_schema
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from env.sandbox import DEFAULT_TIMEOUT_S, prepare_tool_code


def _call_helper(request: bytes, timeout: float) -> subprocess.CompletedProcess[bytes]:
    """Start the helper without applying preexec_fn in the PyTorch process."""
    return subprocess.run(
        [sys.executable, "-m", "rl.custom.sandbox_worker"],
        input=request,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


class ToolCreditSandboxTool(BaseTool):
    """Execute Python through ``env.sandbox`` and expose trajectory tool metrics."""

    def __init__(self, config: dict[str, Any], tool_schema: OpenAIFunctionToolSchema | None):
        super().__init__(config, tool_schema)
        self.timeout = float(config.get("timeout", DEFAULT_TIMEOUT_S))
        if self.timeout <= 0:
            raise ValueError("sandbox timeout must be positive")
        # Calls inside one AgentLoop worker are intentionally synchronous because
        # forking helpers from PyTorch executor threads can deadlock.  veRL's eight
        # AgentLoop workers still execute tools concurrently across trajectories.

    async def _run(self, code: str) -> dict[str, Any]:
        """Use a lightweight helper process around the canonical synchronous sandbox.

        ``env.sandbox.run_python`` intentionally uses ``preexec_fn`` for rlimits,
        which is unsafe when called from an executor thread.  The helper remains
        single-threaded; concurrency comes from veRL's AgentLoop worker processes.
        """
        request = json.dumps({"code": code, "timeout": self.timeout}).encode()
        process = _call_helper(request, self.timeout + 10.0)
        if process.returncode != 0:
            diagnostic = process.stderr.decode(errors="replace")[:2000]
            raise RuntimeError(f"sandbox helper failed: {diagnostic}")
        return dict(json.loads(process.stdout))

    async def code_interpreter(self, code: str) -> str:
        """Execute Python code in the ToolCredit sandbox.

        Args:
            code: Python source, optionally enclosed in a Python Markdown fence.

        Returns:
            Captured stdout, or an explicit error/timeout message.
        """
        prepared = prepare_tool_code(code)
        result = await self._run(prepared)
        if result["status"] == "ok":
            return result["stdout"] or "(no output)"
        detail = result["stderr"] or result["stdout"] or "no diagnostic output"
        return f"{result['status']}: {detail}"

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(get_json_schema(self.code_interpreter))

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResponse, float, dict[str, float]]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            raise ValueError("ToolAgentLoop must pass agent_data to ToolCreditSandboxTool")

        extras = agent_data.extra_fields
        extras["tool_call_counts"] = int(extras.get("tool_call_counts", 0)) + 1
        extras.setdefault("tool_success_count", 0)
        extras.setdefault("tool_error_count", 0)

        if set(parameters) != {"code"} or not isinstance(parameters.get("code"), str):
            extras["tool_error_count"] += 1
            response = "error: code_interpreter expects exactly one string argument named 'code'"
            return ToolResponse(text=response), 0.0, {"tool_success": 0.0, "tool_error": 1.0}

        prepared = prepare_tool_code(parameters["code"])
        result = await self._run(prepared)
        success = result["status"] == "ok"
        if success:
            extras["tool_success_count"] += 1
            text = result["stdout"] or "(no output)"
        else:
            extras["tool_error_count"] += 1
            detail = result["stderr"] or result["stdout"] or "no diagnostic output"
            text = f"{result['status']}: {detail}"

        return ToolResponse(text=text), 0.0, {
            "tool_success": float(success),
            "tool_error": float(not success),
            "tool_wall_time": result["wall_time"],
        }
