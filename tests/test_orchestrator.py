import asyncio
import json
import pytest
from backend.tools.base import Tool, ToolContext, ToolResult
from backend.tools.registry import ToolRegistry
from backend.tools.orchestrator import ToolOrchestrator
from pydantic import BaseModel, Field


class DummyArgs(BaseModel):
    value: int = Field(ge=1)


class DummyTool(Tool):
    name = "dummy_tool"
    description = "A dummy tool for testing."
    args_model = DummyArgs
    timeout_s = 0.5
    examples = [("run dummy 5", {"value": 5})]

    def gate(self, text: str) -> bool:
        return "dummy" in text.lower()

    def fallback_arguments(self, text: str, ctx: ToolContext):
        return {"value": 42}

    async def run(self, args: DummyArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, data={"res": args.value * 2})


class FailingTool(Tool):
    name = "failing_tool"
    description = "A failing tool."
    args_model = DummyArgs
    timeout_s = 0.1

    def gate(self, text: str) -> bool:
        return "fail" in text.lower()

    async def run(self, args: DummyArgs, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("Crash")


class SlowTool(Tool):
    name = "slow_tool"
    description = "A slow tool."
    args_model = DummyArgs
    timeout_s = 0.1

    def gate(self, text: str) -> bool:
        return "slow" in text.lower()

    async def run(self, args: DummyArgs, ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(1.0)
        return ToolResult(ok=True, data={})


class MockLLM:
    def __init__(self, response_json: str = ""):
        self.response_json = response_json
        self.call_count = 0

    async def decide_json(self, messages, model=None, num_predict=96):
        self.call_count += 1
        return self.response_json


@pytest.mark.asyncio
async def test_orchestrator_gate_noop_costs_no_llm_call():
    llm = MockLLM('{"tool": "dummy_tool", "arguments": {"value": 5}}')
    registry = ToolRegistry([DummyTool()])
    orch = ToolOrchestrator(registry, llm)
    ctx = ToolContext(owner_id="u1", session_id="s1")

    # Text that does not match gate
    outcome = await orch.run("Hello there, what is the weather?", ctx)
    assert outcome is None
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_orchestrator_valid_json_executes_tool():
    llm = MockLLM('{"tool": "dummy_tool", "arguments": {"value": 5}}')
    registry = ToolRegistry([DummyTool()])
    orch = ToolOrchestrator(registry, llm)
    ctx = ToolContext(owner_id="u1", session_id="s1")

    outcome = await orch.run("Please test dummy", ctx)
    assert outcome is not None
    assert outcome.name == "dummy_tool"
    assert outcome.source == "llm"
    assert outcome.result.ok is True
    assert outcome.result.data["res"] == 10
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_invalid_json_uses_fallback():
    llm = MockLLM("THIS IS NOT JSON")
    registry = ToolRegistry([DummyTool()])
    orch = ToolOrchestrator(registry, llm)
    ctx = ToolContext(owner_id="u1", session_id="s1")

    outcome = await orch.run("Please test dummy", ctx)
    assert outcome is not None
    assert outcome.name == "dummy_tool"
    assert outcome.source == "fallback"
    assert outcome.result.data["res"] == 84  # 42 * 2


@pytest.mark.asyncio
async def test_orchestrator_schema_violation_uses_fallback():
    # Value must be >= 1, sending 0 is a validation error
    llm = MockLLM('{"tool": "dummy_tool", "arguments": {"value": -5}}')
    registry = ToolRegistry([DummyTool()])
    orch = ToolOrchestrator(registry, llm)
    ctx = ToolContext(owner_id="u1", session_id="s1")

    outcome = await orch.run("Please test dummy", ctx)
    assert outcome is not None
    assert outcome.source == "fallback"
    assert outcome.result.data["res"] == 84


@pytest.mark.asyncio
async def test_orchestrator_tool_exception_handled():
    llm = MockLLM('{"tool": "failing_tool", "arguments": {"value": 1}}')
    registry = ToolRegistry([FailingTool()])
    orch = ToolOrchestrator(registry, llm)
    ctx = ToolContext(owner_id="u1", session_id="s1")

    outcome = await orch.run("test fail", ctx)
    assert outcome is not None
    assert outcome.result.ok is False
    assert outcome.result.error_code == "tool_error"


@pytest.mark.asyncio
async def test_orchestrator_tool_timeout_handled():
    llm = MockLLM('{"tool": "slow_tool", "arguments": {"value": 1}}')
    registry = ToolRegistry([SlowTool()])
    orch = ToolOrchestrator(registry, llm)
    ctx = ToolContext(owner_id="u1", session_id="s1")

    outcome = await orch.run("test slow", ctx)
    assert outcome is not None
    assert outcome.result.ok is False
    assert outcome.result.error_code == "timeout"


@pytest.mark.asyncio
async def test_orchestrator_fault_injection():
    llm = MockLLM('{"tool": "dummy_tool", "arguments": {"value": 5}}')
    registry = ToolRegistry([DummyTool()])
    orch = ToolOrchestrator(registry, llm, fault="dummy_tool:timeout")
    ctx = ToolContext(owner_id="u1", session_id="s1")

    outcome = await orch.run("test dummy", ctx)
    assert outcome is not None
    assert outcome.result.ok is False
    assert outcome.result.error_code == "timeout"
