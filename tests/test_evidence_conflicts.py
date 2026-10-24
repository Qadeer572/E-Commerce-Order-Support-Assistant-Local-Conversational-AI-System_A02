import pytest
from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import Session
from backend.retrieval.types import Hit, RetrievalResult
from tests.test_prompt_budget import Tokenizer
from tests.test_rag_conversation import LLM


@pytest.mark.asyncio
async def test_conflicting_same_product_prices_require_clarification_without_generation():
    class Evidence:
        async def retrieve(self,*args,**kwargs):
            return RetrievalResult('ok',[Hit('a','a.md','Earbuds','Earbuds price $73.21.',.9,'product','ELEC-001'),
                Hit('b','b.md','Earbuds','Earbuds price $99.99.',.9,'product','ELEC-001')],1)
    llm=LLM()
    cm=ConversationManager(llm,Evidence(),Tokenizer())
    reply=''.join([t async for t in cm.handle_turn(Session(),'earbuds price?')])
    assert 'conflicting' in reply.lower()
    assert not llm.messages
