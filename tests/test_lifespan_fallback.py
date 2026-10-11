import pytest
from fastapi.testclient import TestClient
from backend import main, history_store


def test_missing_tokenizer_still_serves_health_and_safe_chat(monkeypatch,tmp_path):
    def missing():
        raise OSError('missing tokenizer')
    class Engine:
        async def health_check(self):
            return dict(status='ok',available=True,model='test')
    monkeypatch.setattr(main,'load_generation_tokenizer',missing)
    monkeypatch.setattr(main,'LLMEngine',Engine)
    monkeypatch.setattr(history_store,'HISTORY_FILE',tmp_path/'history.json')
    with TestClient(main.app) as client:
        assert client.get('/api/health').json()['rag']['ready'] is False
        with client.websocket_connect('/ws/chat') as ws:
            ws.send_json(dict(session_id='missing-model',message='earbuds price?'))
            assert 'right now' in ws.receive_json()['content']
            assert ws.receive_json()['type']=='done'
