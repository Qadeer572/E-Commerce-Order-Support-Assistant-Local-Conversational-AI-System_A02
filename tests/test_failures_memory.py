import asyncio
import json
import pytest
from backend.conversation_manager.session import Session
from backend.conversation_manager.manager import ConversationManager
from backend import history_store
from backend.retrieval.embeddings import ConservativeTokenizer
from tests.test_rag_conversation import Retrieval, LLM
from tests.test_prompt_budget import Tokenizer


def test_missing_tokenizer_uses_conservative_unicode_count():
    tokenizer=ConservativeTokenizer()
    assert len(tokenizer.encode('漢'))==3
    assert len(tokenizer.apply_chat_template([{'role':'user','content':'漢'}]))>=3


def test_old_history_format_restores(tmp_path,monkeypatch):
    file=tmp_path/'history.json'
    file.write_text(json.dumps({'old':{'messages':[{'role':'user','content':'hi'}]}}))
    monkeypatch.setattr(history_store,'HISTORY_FILE',file)
    s=Session(session_id='old')
    assert history_store.restore_session(s)
    assert s.history[0].content=='hi'
    assert s.summarized_turn_count==0


@pytest.mark.asyncio
async def test_summary_preserves_full_transcript_and_advances_cursor():
    session=Session()
    for i in range(4):
        session.add_turn('user','customer question '*30)
        session.add_turn('assistant','previous answer '*30)
    original=list(session.history)
    cm=ConversationManager(LLM(),Retrieval(),Tokenizer())
    await cm._summarize(session)
    assert session.history==original
    assert 0 < session.summarized_turn_count < len(session.history)-2
    assert session.conversation_summary
