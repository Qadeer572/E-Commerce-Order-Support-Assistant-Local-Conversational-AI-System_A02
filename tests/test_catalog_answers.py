import pytest

from backend import shop_db
from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import Session
from backend.tools.base import ToolContext
from backend.tools.catalog import CatalogTool
from backend.tools.orchestrator import ToolOrchestrator
from backend.tools.registry import ToolRegistry
from tests.test_prompt_budget import Tokenizer


class UnreliableLLM:
    async def decide_json(self, messages, **kwargs):
        return '{"tool":"catalog_query","arguments":{"query":"currently available products in your store"}}'

    async def stream(self, messages):
        yield "I don't have access to the store data."


class NoRetrieval:
    async def retrieve(self, *args, **kwargs):
        raise AssertionError('A live catalog answer must not retrieve an unrelated FAQ')


@pytest.fixture
def catalog(tmp_path):
    db = tmp_path / 'shop.db'
    shop_db.build_database(db)
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize('question', [
    'list the products that are available in your store',
    'I am telling you that please provide a list of products that are currently available in your store',
    'show me all products',
    'list the product that are availanle in your store',
    'What products are available?',
    'What can I buy in your store?',
    'show your catalog',
])
async def test_catalog_listing_is_answered_from_live_rows(catalog, question):
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    manager = ConversationManager(UnreliableLLM(), NoRetrieval(), Tokenizer(), tools=tools)
    session = Session(last_sku='ELEC-001')
    answer = ''.join([token async for token in manager.handle_turn(session, question)])
    conn = shop_db.connect(catalog)
    rows = conn.execute('SELECT name, price FROM products WHERE stock > 0').fetchall()
    conn.close()
    for row in rows:
        assert row['name'] in answer
        assert f"${row['price']:.2f}" in answer
    assert "don't have access" not in answer
    assert 'faq-' not in answer
    assert session.last_sku is None
    assert session.history[-1].content == answer


@pytest.mark.asyncio
async def test_catalog_browsing_filters_stock_category_and_budget(catalog):
    tool = CatalogTool(catalog)
    tools = ToolOrchestrator(ToolRegistry([tool]), UnreliableLLM())
    outcome = await tools.run('show available electronics under $50', ToolContext('u', 's'))
    assert outcome.result.data['products']
    for product in outcome.result.data['products']:
        assert product['category'] == 'Electronics'
        assert product['price'] <= 50
        assert product['total_stock'] > 0


@pytest.mark.asyncio
async def test_single_sku_stock_uses_tool_facts_and_remembers_product(catalog):
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    manager = ConversationManager(UnreliableLLM(), NoRetrieval(), Tokenizer(), tools=tools)
    session = Session()
    answer = ''.join([token async for token in manager.handle_turn(session, 'Is ELEC-001 in stock?')])
    assert 'ProSound' in answer and '$49.99' in answer
    assert session.last_sku == 'ELEC-001'
    followup = ''.join([token async for token in manager.handle_turn(session, 'How much is it?')])
    assert 'ProSound' in followup and '$49.99' in followup


@pytest.mark.asyncio
@pytest.mark.parametrize('question', ['recommend something affordable', 'what can I buy under $20?'])
async def test_broad_budget_requests_do_not_search_conversational_words(catalog, question):
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    outcome = await tools.run(question, ToolContext('u', 's'))
    assert outcome is not None
    assert outcome.result.data['products']
    if '$20' in question:
        assert all(row['price'] <= 20 for row in outcome.result.data['products'])


@pytest.mark.asyncio
async def test_no_stock_and_tool_errors_are_reported_truthfully(catalog):
    with shop_db.connect(catalog) as conn:
        conn.execute('UPDATE products SET stock = 0')
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    manager = ConversationManager(UnreliableLLM(), NoRetrieval(), Tokenizer(), tools=tools)
    answer = ''.join([token async for token in manager.handle_turn(Session(), 'list available products')])
    assert 'matching products' in answer and 'access' not in answer
    tools.fault = 'catalog_query:error'
    failed = ''.join([token async for token in manager.handle_turn(Session(), 'list available products')])
    assert 'lookup failed' in failed
    assert 'Source:' not in failed


@pytest.mark.asyncio
@pytest.mark.parametrize('question, category, max_price', [
    ('show all home goods products', 'Home', None),
    ('show products priced under $50', None, 50),
    ('show sports equipment products', 'Sports', None),
])
async def test_category_aliases_and_price_wording(catalog, question, category, max_price):
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    outcome = await tools.run(question, ToolContext('u', 's'))
    assert outcome.result.data['products']
    for row in outcome.result.data['products']:
        assert category is None or row['category'] == category
        assert max_price is None or row['price'] <= max_price


@pytest.mark.asyncio
async def test_sku_does_not_override_customer_budget(catalog):
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    outcome = await tools.run('show ELEC-001 under $20', ToolContext('u', 's'))
    assert outcome.result.data['products'] == []


@pytest.mark.asyncio
async def test_out_of_stock_list_does_not_return_in_stock_rows(catalog):
    with shop_db.connect(catalog) as conn:
        conn.execute("UPDATE products SET stock=0 WHERE sku='ELEC-001'")
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    outcome = await tools.run('show all products not in stock', ToolContext('u', 's'))
    assert [row['sku'] for row in outcome.result.data['products']] == ['ELEC-001']


@pytest.mark.asyncio
async def test_unknown_product_filter_is_not_dropped(catalog):
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    outcome = await tools.run('list telescope products', ToolContext('u', 's'))
    assert outcome.result.data['products'] == []


@pytest.mark.asyncio
async def test_catalog_direct_route_does_not_preempt_other_tools(catalog):
    from backend.tools.calculator import CalculatorTool

    class Router:
        called = False
        async def decide_json(self, messages, **kwargs):
            self.called = True
            return '{"tool":"calculator","arguments":{"expression":"49.99 * 0.85"}}'

    router = Router()
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog), CalculatorTool()]), router)
    outcome = await tools.run('calculate 15% off the price of ELEC-001', ToolContext('u', 's'))
    assert router.called
    assert outcome.name == 'calculator'


@pytest.mark.asyncio
async def test_catalog_price_range(catalog):
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    outcome = await tools.run('show products between $10 and $20', ToolContext('u', 's'))
    assert outcome.result.data['products']
    assert all(10 <= row['price'] <= 20 for row in outcome.result.data['products'])


@pytest.mark.asyncio
async def test_catalog_multiple_categories(catalog):
    tools = ToolOrchestrator(ToolRegistry([CatalogTool(catalog)]), UnreliableLLM())
    outcome = await tools.run('list electronics and apparel products', ToolContext('u', 's'))
    assert {row['category'] for row in outcome.result.data['products']} == {'Electronics', 'Apparel'}
