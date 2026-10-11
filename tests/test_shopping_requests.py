import pytest

from backend.conversation_manager.intent_triage import detect_intent
from backend.retrieval.service import build_retrieval_query
from backend.conversation_manager.session import Session
from backend.conversation_manager.manager import ConversationManager
from tests.test_rag_conversation import LLM, Retrieval
from tests.test_prompt_budget import Tokenizer
from backend.retrieval.types import Hit


@pytest.mark.parametrize('query', [
    'i need cheap product', 'list if prodcut', 'list of prodcuts',
    'show your catalog', 'I need something cheap', 'any budget options?',
    'recommend something affordable', 'what can I buy under $20?',
    'show me all products', 'list products', 'cheapest items',
])
def test_shopping_is_in_scope(query):
    assert detect_intent(query) == 'PRODUCT_QUERY'


@pytest.mark.parametrize('query', ['cheap flights to Paris', 'budget for my wedding', 'price of Bitcoin'])
def test_nonshopping_budget_requests_remain_out_of_scope(query):
    assert detect_intent(query) == 'OUT_OF_SCOPE'


@pytest.mark.asyncio
async def test_catalog_request_uses_catalog_evidence():
    retrieval = Retrieval()
    manager = ConversationManager(LLM(), retrieval, Tokenizer())
    async for _ in manager.handle_turn(Session(last_sku='ELEC-001'), 'list if prodcut'):
        pass
    assert retrieval.queries[0][1]['doc_types'] == ('catalog',)
    assert retrieval.queries[0][1]['sku'] is None
    assert 'catalog' in retrieval.queries[0][0].lower()


@pytest.mark.asyncio
async def test_no_match_streams_conversation_without_business_evidence():
    llm = LLM()
    manager = ConversationManager(llm, Retrieval('no_match'), Tokenizer())
    tokens = [t async for t in manager.handle_turn(Session(), 'do you sell telescopes?')]
    assert len(tokens) > 1
    assert llm.messages
    prompt = llm.messages[0]['content']
    assert 'No reliable document matches' in prompt
    assert 'Retrieved evidence [' not in prompt
    assert 'Sources:' not in ''.join(tokens)


def test_budget_evidence_filters_and_sorts_retrieved_prices():
    from backend.conversation_manager.manager import catalog_evidence
    hits = [Hit('catalog', 'catalog/one.md', 'Catalog',
                '- SPRT-001 | Expensive thing | $54.99 | Sports\n'
                '- ACC-001 | Affordable item | $12.99 | Accessories\n'
                '- ACC-002 | Cheapest item | $9.99 | Accessories', .9, 'catalog')]
    selected = catalog_evidence(hits, 'what can I buy under $20?')
    text = '\n'.join(h.text for h in selected)
    assert 'Expensive thing' not in text
    assert text.index('Cheapest item') < text.index('Affordable item')
    assert selected[0].source == hits[0].source


def test_catalog_evidence_does_not_invent_items_when_budget_has_no_match():
    from backend.conversation_manager.manager import catalog_evidence
    hit = Hit('catalog', 'catalog/one.md', 'Catalog', '- ACC-001 | Thing | $9.99 | Home', .9, 'catalog')
    assert catalog_evidence([hit], 'products under $5') == []


@pytest.mark.asyncio
async def test_no_match_cannot_claim_an_unlisted_product_is_sold():
    class InventingLLM:
        async def stream(self, messages):
            yield 'Yes, I sell telescopes for $15. What model do you want?'
    cm = ConversationManager(InventingLLM(), Retrieval('no_match'), Tokenizer())
    reply = ''.join([t async for t in cm.handle_turn(Session(), 'do you sell telescopes?')])
    assert 'sell telescopes' not in reply
    assert '$15' not in reply
    assert 'reliable information' in reply


@pytest.mark.asyncio
async def test_typo_is_normalized_in_the_generation_question():
    llm = LLM()
    cm = ConversationManager(llm, Retrieval(), Tokenizer())
    async for _ in cm.handle_turn(Session(), 'list if prodcut'):
        pass
    assert llm.messages[-1]['content'] == 'list of products'


@pytest.mark.asyncio
async def test_catalog_listing_streams_exact_retrieved_rows_without_model_refusal():
    from backend.retrieval.types import RetrievalResult
    class CatalogRetrieval:
        async def retrieve(self, query, **kwargs):
            hit = Hit('catalog', 'catalog/one.md', 'Catalog',
                      '- ACC-002 | CablePro Cable | $9.99 | Accessories\n'
                      '- ACC-001 | Phone Case | $12.99 | Accessories', .9, 'catalog')
            return RetrievalResult('ok', [hit], 1)
    class RefusingLLM:
        async def stream(self, messages):
            raise AssertionError('Catalog lists must not depend on generated facts')
            yield
    cm = ConversationManager(RefusingLLM(), CatalogRetrieval(), Tokenizer())
    tokens = [t async for t in cm.handle_turn(Session(), 'list if prodcut')]
    reply = ''.join(tokens)
    assert 'CablePro Cable' in reply and '$9.99' in reply
    assert 'Phone Case' in reply and '$12.99' in reply
    assert 'Sources:' in reply
    assert len(tokens) >= 3
