"""Comprehensive smoke test against live running ShopBot backend (http://127.0.0.1:8000)."""
import asyncio
import json
import urllib.request
import websockets

BASE_HTTP = "http://127.0.0.1:8000"
WS_CHAT = "ws://127.0.0.1:8000/ws/chat"
WS_VOICE = "ws://127.0.0.1:8000/ws/voice"


def test_health():
    req = urllib.request.urlopen(f"{BASE_HTTP}/api/health")
    data = json.loads(req.read().decode())
    print("\n[1] /api/health:")
    print(f"  Status: {data['status']}")
    print(f"  RAG Ready: {data['rag']['ready']}")
    print(f"  Voice Ready: ASR={data['voice']['asr_ready']}, TTS={data['voice']['tts_ready']}")
    print(f"  Tools: {data['tools']}")
    assert data["status"] == "ok"
    assert data["rag"]["ready"] is True
    assert data["voice"]["asr_ready"] is True
    assert data["voice"]["tts_ready"] is True


async def test_chat_query(session_id: str, query: str):
    async with websockets.connect(WS_CHAT) as ws:
        await ws.send(json.dumps({"session_id": session_id, "message": query}))
        tokens = []
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "token":
                tokens.append(msg["content"])
            elif msg["type"] == "done":
                reply = "".join(tokens)
                return reply
            elif msg["type"] == "error":
                raise RuntimeError(f"Chat error: {msg['content']}")


async def test_voice_turn(session_id: str, query: str):
    """Test /ws/voice control and audio streaming frames."""
    async with websockets.connect(WS_VOICE) as ws:
        await ws.send(json.dumps({"type": "start", "session_id": session_id, "tts": True}))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", f"Unexpected frame: {ready}"

        # Synthesize a speech chunk or send end_of_speech with dummy pcm
        # Simulate 1 second of 16 kHz silence
        silence = b"\x00" * 32000
        await ws.send(silence)
        # End speech
        await ws.send(json.dumps({"type": "end_of_speech"}))

        received_types = []
        tokens = []
        audio_chunks = 0
        while True:
            raw = await asyncio.wait_for(ws.recv(), 30)
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            t = msg["type"]
            received_types.append(t)
            if t == "no_speech":
                break
            if t == "token":
                tokens.append(msg["content"])
            elif t == "audio":
                audio_chunks += 1
            elif t == "done":
                break
        return {"received_types": received_types, "reply": "".join(tokens), "audio_chunks": audio_chunks}


async def main():
    print("=" * 60)
    print("RUNNING LIVE SHOPBOT SMOKE TESTS")
    print("=" * 60)

    # 1. Health check
    test_health()

    # 2. Calculator test
    print("\n[2] Testing Calculator Tool over /ws/chat:")
    calc_reply = await test_chat_query("smoke-calc", "What is 15% off $49.99?")
    print(f"  User: What is 15% off $49.99?")
    print(f"  Bot:  {calc_reply.strip()}")
    assert "42.49" in calc_reply or "42.50" in calc_reply or "42" in calc_reply

    # 3. Catalog Live Stock test
    print("\n[3] Testing Catalog Query Tool over /ws/chat:")
    stock_reply = await test_chat_query("smoke-stock", "Is ELEC-001 in stock right now?")
    print(f"  User: Is ELEC-001 in stock right now?")
    print(f"  Bot:  {stock_reply.strip()}")
    assert "ELEC-001" in stock_reply or "in stock" in stock_reply.lower() or "ProSound" in stock_reply

    # 4. RAG Document Retrieval test
    print("\n[4] Testing RAG Evidence Retrieval over /ws/chat:")
    rag_reply = await test_chat_query("smoke-rag", "What is the battery life of ProSound earbuds?")
    print(f"  User: What is the battery life of ProSound earbuds?")
    print(f"  Bot:  {rag_reply.strip()}")
    assert "Sources:" in rag_reply
    assert "battery" in rag_reply.lower() or "30" in rag_reply or "hours" in rag_reply

    # 5. Out of scope redirect
    print("\n[5] Testing Out of Scope Guard over /ws/chat:")
    oos_reply = await test_chat_query("smoke-oos", "What is the capital of France?")
    print(f"  User: What is the capital of France?")
    print(f"  Bot:  {oos_reply.strip()}")
    assert "order" in oos_reply.lower() or "support" in oos_reply.lower()

    # 6. Voice WebSocket test
    print("\n[6] Testing Voice WebSocket Handshake & Frames over /ws/voice:")
    voice_res = await test_voice_turn("smoke-voice", "silence")
    print(f"  Frames received: {voice_res['received_types']}")
    assert "no_speech" in voice_res["received_types"]

    print("\n" + "=" * 60)
    print("ALL LIVE SMOKE TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
