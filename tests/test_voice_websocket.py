import asyncio
import pytest
from fastapi.testclient import TestClient
from backend import main, history_store
from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import SessionStore
from backend.voice.asr import ASREvent
from tests.test_rag_conversation import LLM, Retrieval
from tests.test_prompt_budget import Tokenizer


class FakeASRStream:
    def __init__(self):
        self.chunk_count = 0

    async def accept(self, chunk: bytes):
        self.chunk_count += 1
        if self.chunk_count == 1:
            return [ASREvent(kind="partial", text="earbuds")]
        return [ASREvent(kind="final", text="earbuds price", finalize_ms=15.0, trailing_silence_ms=600.0)]

    async def flush(self):
        return ASREvent(kind="final", text="earbuds price", finalize_ms=10.0, trailing_silence_ms=600.0)


class FakeASR:
    def __init__(self, ready=True):
        self.ready = ready

    def new_stream(self):
        return FakeASRStream()

    def close(self):
        pass


class FakeTTS:
    def __init__(self, ready=True):
        self.ready = ready
        self.sample_rate = 22050

    async def synthesize(self, text: str) -> bytes:
        # Return 8000 bytes (4000 samples) of fake PCM16
        return b"\x00\x00" * 4000

    def close(self):
        pass


def voice_client(monkeypatch, tmp_path, asr_ready=True, tts_ready=True):
    monkeypatch.setattr(main, "_session_store", SessionStore())
    monkeypatch.setattr(main, "_conv_manager", ConversationManager(LLM(), Retrieval("ok"), Tokenizer()))
    monkeypatch.setattr(main, "_asr", FakeASR(ready=asr_ready))
    monkeypatch.setattr(main, "_tts", FakeTTS(ready=tts_ready))
    monkeypatch.setattr(history_store, "HISTORY_FILE", tmp_path / "history.json")
    return TestClient(main.app)


def test_voice_missing_assets_clean_error(monkeypatch, tmp_path):
    c = voice_client(monkeypatch, tmp_path, asr_ready=False)
    with c.websocket_connect("/ws/voice") as ws:
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert "Voice models are not installed" in frame["content"]


def test_voice_review_uses_only_confirmed_text(monkeypatch, tmp_path):
    c = voice_client(monkeypatch, tmp_path)
    with c.websocket_connect('/ws/voice') as ws:
        ws.send_json(dict(type='start', session_id='review', tts=False, review_transcript=True))
        assert ws.receive_json()['type'] == 'ready'
        ws.send_bytes(b'\x00\x00' * 1600)
        assert ws.receive_json()['type'] == 'partial_transcript'
        ws.send_bytes(b'\x00\x00' * 1600)
        preview = ws.receive_json()
        assert preview['type'] == 'final_transcript'
        assert preview.get('requires_confirmation') is True
        assert main._session_store.get('review') is None
        for invalid in ['', ' ' * 3, 'x' * 2001, 12]:
            ws.send_json(dict(type='confirm_transcript', text=invalid))
            assert ws.receive_json()['type'] == 'error'
            assert main._session_store.get('review') is None
        ws.send_json(dict(type='confirm_transcript', text='My name is Qadeer Raza'))
        assert ws.receive_json()['text'] == 'My name is Qadeer Raza'
        while ws.receive_json()['type'] != 'metrics':
            pass
        session = main._session_store.get('review')
        assert session.history[0].content == 'My name is Qadeer Raza'
        ws.send_json(dict(type='confirm_transcript', text='Duplicate'))
        assert ws.receive_json()['type'] == 'error'
        assert len(session.history) == 2


def test_voice_full_pipeline(monkeypatch, tmp_path):
    c = voice_client(monkeypatch, tmp_path)
    with c.websocket_connect("/ws/voice") as ws:
        # 1. Send start
        ws.send_json({"type": "start", "session_id": "v1", "tts": True})
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["asr"] is True
        assert ready["tts"] is True

        # 2. Send 2 audio chunks (100ms each = 3200 bytes)
        ws.send_bytes(b"\x00\x00" * 1600)
        p = ws.receive_json()
        assert p["type"] == "partial_transcript"
        assert p["text"] == "earbuds"

        ws.send_bytes(b"\x00\x00" * 1600)
        f = ws.receive_json()
        assert f["type"] == "final_transcript"

        # 3. Read remaining frames until audio_end and metrics
        frames = []
        while True:
            frame = ws.receive_json()
            frames.append(frame)
            if frame["type"] == "metrics":
                break

        types = [f["type"] for f in frames]
        assert "token" in types
        assert "audio" in types
        assert "done" in types
        assert "audio_end" in types
        assert frames[-1]["type"] == "metrics"


def test_formatted_transcript_keeps_name_spelling_and_punctuation(monkeypatch, tmp_path):
    async def formatted(self, chunk):
        return [ASREvent('final', 'My name is Qadeer Raza.', text_is_formatted=True)]
    monkeypatch.setattr(FakeASRStream, 'accept', formatted)
    c = voice_client(monkeypatch, tmp_path)
    with c.websocket_connect('/ws/voice') as ws:
        ws.send_json(dict(type='start', session_id='formatted', review_transcript=True))
        assert ws.receive_json()['type'] == 'ready'
        ws.send_bytes(b'\0\0' * 1600)
        preview = ws.receive_json()
        assert preview['text'] == 'My name is Qadeer Raza.'
        assert preview['requires_confirmation'] is True


def test_chat_and_voice_shared_session_history(monkeypatch, tmp_path):
    c = voice_client(monkeypatch, tmp_path)
    session_id = "shared-session-1"

    # Turn 1 on typed chat
    with c.websocket_connect("/ws/chat") as ws_chat:
        ws_chat.send_json({"session_id": session_id, "message": "My name is Sheharyar"})
        while ws_chat.receive_json()["type"] != "done":
            pass

    # Verify session store has history
    session = main._session_store.get(session_id)
    assert len(session.history) >= 2

    # Turn 2 on voice
    with c.websocket_connect("/ws/voice") as ws_voice:
        ws_voice.send_json({"type": "start", "session_id": session_id, "tts": False})
        ws_voice.receive_json()  # ready
        ws_voice.send_bytes(b"\x00\x00" * 1600)
        ws_voice.receive_json()  # partial
        ws_voice.send_bytes(b"\x00\x00" * 1600)
        ws_voice.receive_json()  # final

        while True:
            frame = ws_voice.receive_json()
            if frame["type"] == "metrics":
                break

    # Now history has 4 turns!
    assert len(session.history) >= 4


def test_chat_websocket_contract_retained(monkeypatch, tmp_path):
    """Confirm /ws/chat strictly outputs token frames followed by done (R3)."""
    c = voice_client(monkeypatch, tmp_path)
    with c.websocket_connect("/ws/chat") as ws:
        ws.send_json({"session_id": "chat-only", "message": "earbuds price?"})
        frames = []
        while not frames or frames[-1]["type"] == "token":
            frames.append(ws.receive_json())
        assert len(frames) >= 2
        assert all(f["type"] == "token" for f in frames[:-1])
        assert frames[-1]["type"] == "done"
