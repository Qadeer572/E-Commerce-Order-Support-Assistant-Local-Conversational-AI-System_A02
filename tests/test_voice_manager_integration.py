import pytest
from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import Session
from backend.tools.base import Tool, ToolContext, ToolResult
from backend.tools.registry import ToolRegistry
from backend.tools.orchestrator import ToolOrchestrator
from pydantic import BaseModel
from tests.test_rag_conversation import Retrieval
from tests.test_prompt_budget import Tokenizer


class EchoLLM:
    def __init__(self):
        self.last_messages = []

    async def stream(self, messages):
        self.last_messages = messages
        yield "Answer based on tool."

    async def decide_json(self, messages, model=None, num_predict=96):
        return '{"tool": "calc", "arguments": {"expr": "100"}}'


class CalcArgs(BaseModel):
    expr: str


class CalcTool(Tool):
    name = "calc"
    description = "calculator"
    args_model = CalcArgs

    def gate(self, text: str) -> bool:
        return "calc" in text.lower() or "+" in text or "%" in text

    async def run(self, args: CalcArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, data={"result": "100.00"})


@pytest.mark.asyncio
async def test_manager_bypasses_out_of_scope_when_tool_runs():
    llm = EchoLLM()
    orch = ToolOrchestrator(ToolRegistry([CalcTool()]), llm)
    cm = ConversationManager(llm, Retrieval("ok"), Tokenizer(), tools=orch)
    session = Session()

    # "what is 15% of 200" would normally be OUT_OF_SCOPE without tools
    tokens = [t async for t in cm.handle_turn(session, "what is 15% of 200")]
    reply = "".join(tokens)
    assert "ShopBot is an e-commerce customer support assistant" not in reply
    assert "Answer based on tool" in reply

    # Verify tool result was injected into prompt
    assert any("Tool result [calc]" in m["content"] for m in llm.last_messages if m["role"] == "system")


@pytest.mark.asyncio
async def test_manager_tool_failure_produces_fallback():
    class FailingCalc(CalcTool):
        async def run(self, args: CalcArgs, ctx: ToolContext) -> ToolResult:
            return ToolResult(ok=False, error_code="timeout", user_message="Calculator timed out.")

    llm = EchoLLM()
    orch = ToolOrchestrator(ToolRegistry([FailingCalc()]), llm)
    cm = ConversationManager(llm, Retrieval("ok"), Tokenizer(), tools=orch)
    session = Session()

    tokens = [t async for t in cm.handle_turn(session, "calc 100")]
    reply = "".join(tokens)
    assert reply != ""


@pytest.mark.asyncio
async def test_manager_unchanged_when_tools_none():
    llm = EchoLLM()
    cm = ConversationManager(llm, Retrieval("ok"), Tokenizer(), tools=None)
    session = Session()

    tokens = [t async for t in cm.handle_turn(session, "What is the capital of Pakistan?")]
    reply = "".join(tokens)
    # Refusal redirect for out of scope
    assert "order support assistant for this store" in reply
