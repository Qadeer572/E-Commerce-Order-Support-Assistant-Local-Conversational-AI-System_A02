import asyncio
import subprocess
import sys
import pytest
from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import Session
from backend.conversation_manager.intent_triage import detect_intent
from backend.conversation_manager.prompt_builder import build_messages
from backend.retrieval.service import build_retrieval_query, is_entity_followup
from tests.test_rag_conversation import LLM, Retrieval
from tests.test_prompt_budget import Tokenizer


@pytest.mark.asyncio
async def test_unknown_order_replaces_prior_followup_entity():
    s=Session(last_order_id='ORD-10001',active_intent='ORDER_TRACKING')
    r=Retrieval('no_match')
    cm=ConversationManager(LLM(),r,Tokenizer())
    async for _ in cm.handle_turn(s,'Track ORD-99999'): pass
    async for _ in cm.handle_turn(s,'When will it arrive?'): pass
    assert r.queries[-1][1]['order_id']=='ORD-99999'


@pytest.mark.parametrize('query',['What colors does it come in?','Tell me more','Are they in stock?'])
def test_product_followup_classifier_is_shared(query):
    assert is_entity_followup(query)
    assert 'ELEC-001' in build_retrieval_query(query,Session(last_sku='ELEC-001'),'PRODUCT_QUERY')


@pytest.mark.parametrize('query',['Do you ship internationally?','What warranty coverage do you offer?','How do I cancel my order?'])
def test_store_policy_questions_are_routed_to_policy(query):
    assert detect_intent(query)=='RETURNS_POLICY'


@pytest.mark.asyncio
async def test_summary_cursor_never_skips_unincluded_turns():
    s=Session()
    for content in ['A'*1000,'B'*1000,'UNIQUE_LATE_FACT','reply','recent','reply']:
        s.add_turn('user' if len(s.history)%2==0 else 'assistant',content)
    class Capture(LLM):
        async def stream(self,messages):
            self.messages=messages
            yield 'summary'
    llm=Capture(); cm=ConversationManager(llm,Retrieval(),Tokenizer())
    await cm._summarize(s)
    inputs=''.join(m['content'] for m in llm.messages)
    assert 'UNIQUE_LATE_FACT' in inputs or any(t.content=='UNIQUE_LATE_FACT' for t in s.history[s.summarized_turn_count:])


def test_shutdown_exits_even_with_permanently_blocked_retrieval():
    code='''
import asyncio,threading
from backend.retrieval.service import Retriever
class E:
 revision='x'
 def encode(self,texts): threading.Event().wait()
class S: revision='x'
async def run():
 r=Retriever(E(),S(),timeout=.01)
 await r.retrieve('test',doc_types=('product',))
 r.close()
asyncio.run(run())
'''
    subprocess.run([sys.executable,'-c',code],timeout=3,check=True)
