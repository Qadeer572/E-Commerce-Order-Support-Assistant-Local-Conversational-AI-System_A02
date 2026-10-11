from fastapi.testclient import TestClient
from backend import main, history_store
from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import SessionStore
from tests.test_rag_conversation import LLM, Retrieval
from tests.test_prompt_budget import Tokenizer


def client(monkeypatch,tmp_path,status="ok"):
    monkeypatch.setattr(main,"_session_store",SessionStore())
    monkeypatch.setattr(main,"_conv_manager",ConversationManager(LLM(),Retrieval(status),Tokenizer()))
    monkeypatch.setattr(history_store,"HISTORY_FILE",tmp_path/'history.json')
    return TestClient(main.app)


def test_websocket_streams_tokens_then_done(monkeypatch,tmp_path):
    with client(monkeypatch,tmp_path).websocket_connect('/ws/chat') as ws:
        ws.send_json(dict(session_id='stream',message='earbuds price?'))
        frames=[]
        while not frames or frames[-1]['type']=='token':
            frames.append(ws.receive_json())
        assert len(frames)>=3
        assert all(f['type']=='token' for f in frames[:-1])
        assert frames[-1]['type']=='done'


def test_websocket_failure_has_safe_terminal_frame(monkeypatch,tmp_path):
    with client(monkeypatch,tmp_path,"timeout").websocket_connect('/ws/chat') as ws:
        ws.send_json(dict(session_id='failure',message='earbuds price?'))
        assert ws.receive_json()['type']=='token'
        assert ws.receive_json()['type']=='done'


def test_health_exposes_rag_readiness(monkeypatch,tmp_path):
    class Engine:
        async def health_check(self):
            return dict(status="ok",available=True)
    monkeypatch.setattr(main,"_llm_engine",Engine())
    c=client(monkeypatch,tmp_path)
    response=c.get('/api/health').json()
    assert 'rag' in response
