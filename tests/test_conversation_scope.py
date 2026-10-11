import pytest

from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import Session
from tests.test_rag_conversation import Retrieval
from tests.test_prompt_budget import Tokenizer


class GreetingLLM:
    def __init__(self):
        self.messages = None

    async def stream(self, messages):
        self.messages = messages
        yield "Hello Qadeer Raza! How can I help with your shopping today?"


class ForbiddenLLM:
    async def stream(self, messages):
        raise AssertionError("Unrelated questions must not reach the model")
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("previous", [None, "PRODUCT_QUERY", "ORDER_TRACKING", "RETURNS_POLICY"])
@pytest.mark.parametrize("message", [
    "ok what is capital of pakistan ?",
    "What is the size of Pakistan?",
    "Hi, how much does a trip to Paris cost?",
])
async def test_off_topic_reply_is_guarded_without_losing_support_context(previous, message):
    session = Session(active_intent=previous, last_order_id="ORD-10002")
    manager = ConversationManager(ForbiddenLLM(), Retrieval(), Tokenizer())
    reply = "".join([token async for token in manager.handle_turn(
        session, message
    )])
    assert "Islamabad" not in reply
    assert "product" in reply.lower() and "policy" in reply.lower()
    assert session.active_intent == previous
    assert session.last_order_id == "ORD-10002"
    assert session.history[-1].content == reply


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["hi", "my name is Qadeer Raza"])
async def test_greetings_reach_model_with_social_instructions_and_keep_support_context(message):
    llm = GreetingLLM()
    session = Session(active_intent="ORDER_TRACKING", last_order_id="ORD-10002")
    manager = ConversationManager(llm, Retrieval(), Tokenizer())
    reply = "".join([token async for token in manager.handle_turn(session, message)])
    assert "Hello Qadeer Raza" in reply
    assert llm.messages[-1] == {"role": "user", "content": message}
    assert "greeting" in llm.messages[0]["content"].lower()
    assert session.active_intent == "ORDER_TRACKING"
    assert session.last_order_id == "ORD-10002"
    assert session.history[-2].content == message
