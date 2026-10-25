import pytest
from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import Session
from backend.retrieval.types import Hit, RetrievalResult
from tests.test_prompt_budget import Tokenizer
from tests.test_rag_conversation import LLM
from backend.conversation_manager.manager import has_conflicting_facts


def test_different_order_item_prices_are_not_conflicting_order_facts():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'data/corpus/orders/ORD-10001.md').read_text(encoding='utf-8')
    hit = Hit('order', 'orders/ORD-10001.md', 'Order', text, .9, 'order', '', 'ORD-10001')
    assert not has_conflicting_facts([hit])


def test_contradictory_order_statuses_still_require_clarification():
    hits = [Hit('a', 'a.md', 'Order', 'Status: Delivered', .9, 'order', '', 'ORD-10001'),
            Hit('b', 'b.md', 'Order', 'Status: Cancelled', .9, 'order', '', 'ORD-10001')]
    assert has_conflicting_facts(hits)


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
