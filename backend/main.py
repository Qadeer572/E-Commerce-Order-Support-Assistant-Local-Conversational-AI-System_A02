"""
main.py — FastAPI application entry point.

Exposes:
  WS   /ws/chat                       — streaming chat WebSocket
  POST /api/session                   — create a new session
  DELETE /api/session/{session_id}    — reset/clear a session
  GET  /api/health                    — liveness + Ollama health check
  GET  /                              — serve frontend index.html
  GET  /static/*                      — serve frontend static files
"""

import json
import logging
import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import ALLOWED_ORIGINS
from backend.llm_engine import LLMEngine, LLMEngineError
from backend.conversation_manager.session import SessionStore, Turn
from backend.conversation_manager import ConversationManager
from backend import history_store
from backend.config import VECTOR_DIR, MODEL_DIR
from backend.retrieval.embeddings import LocalEmbedder, load_generation_tokenizer, ConservativeTokenizer
from backend.retrieval.store import VectorStore
from backend.retrieval.service import Retriever

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("shopbot.main")

# ── Global singletons (created once at startup) ───────────────────────────────
_llm_engine:    LLMEngine           = None
_session_store: SessionStore        = None
_conv_manager:  ConversationManager = None

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ── Application lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm_engine, _session_store, _conv_manager

    logger.info("ShopBot starting up…")
    _llm_engine    = LLMEngine()
    warmup = asyncio.create_task(_llm_engine.warmup()) if hasattr(_llm_engine, "warmup") else None
    _session_store = SessionStore()
    def load_rag():
        try:
            tokenizer = load_generation_tokenizer()
        except Exception:
            logger.warning("Generation tokenizer unavailable; using conservative fallback count")
            return Retriever(), ConservativeTokenizer()
        try:
            embedder = LocalEmbedder()
            store = VectorStore(VECTOR_DIR, reader=True)
            if store.manifest["settings"]["model_revision"] != embedder.revision:
                store.close()
                raise RuntimeError("Index model revision mismatch")
            return Retriever(embedder, store), tokenizer
        except Exception:
            logger.exception("RAG unavailable; chat will use safe fallbacks")
            return Retriever(), tokenizer
    retriever, tokenizer = await asyncio.to_thread(load_rag)
    _conv_manager = ConversationManager(_llm_engine, retriever, tokenizer)

    # Periodic session cleanup task (every 10 minutes)
    async def _cleanup_task():
        while True:
            await asyncio.sleep(600)
            removed = _session_store.purge_expired()
            if removed:
                logger.info(f"Purged {removed} expired sessions")

    cleanup = asyncio.create_task(_cleanup_task())

    health = await _llm_engine.health_check()
    if health["status"] == "ok" and health["available"]:
        logger.info(f"Ollama OK — model '{health['model']}' is available")
    elif health["status"] == "ok":
        logger.warning(
            f"Ollama reachable but model '{health['model']}' not found. "
            f"Run: ollama pull {health['model']}"
        )
    else:
        logger.error(f"Ollama unreachable: {health.get('error')}")

    yield  # — server is running —

    cleanup.cancel()
    if warmup:
        warmup.cancel()
        await asyncio.gather(warmup, return_exceptions=True)
    await asyncio.to_thread(retriever.close)
    logger.info("ShopBot shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShopBot — E-Commerce Order Support Assistant",
    version="1.0.0",
    description="Local conversational AI for product Q&A, order tracking, and return policy.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if the directory exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend chat UI."""
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"message": "ShopBot API is running. Frontend not found at /frontend/index.html"}


@app.get("/api/health")
async def health():
    """Liveness check + Ollama model availability."""
    llm_health = await _llm_engine.health_check()
    return {
        "status": "ok",
        "service": "ShopBot",
        "active_sessions": _session_store.count(),
        "llm": llm_health,
        "rag": {"ready": _conv_manager.retriever.ready if hasattr(_conv_manager.retriever, "ready") else True},
    }


@app.post("/api/session")
async def create_session():
    """Create a new session and return its ID."""
    session = _session_store.create()
    logger.info(f"Created session {session.session_id}")
    return {"session_id": session.session_id}


@app.delete("/api/session/{session_id}")
async def reset_session(session_id: str):
    """Reset (clear) an existing session, or return 404 if not found."""
    session = _session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    async with session.lock:
        session.reset()
        await asyncio.to_thread(history_store.save_session, session)
    logger.info(f"Reset session {session_id}")
    return {"status": "reset", "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    """List past conversations (for the frontend history sidebar)."""
    return {"sessions": await asyncio.to_thread(history_store.list_sessions)}


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Return the persisted turn list for a past conversation, and hydrate the
    live SessionStore so the conversation can continue with full context if
    the user keeps chatting in it.
    """
    messages = await asyncio.to_thread(history_store.get_session_messages, session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="No history found for this session")

    session = _session_store.get_or_create(session_id)
    async with session.lock:
        if not session.history and messages:
            await asyncio.to_thread(history_store.restore_session, session)

    return {"session_id": session_id, "messages": messages}


@app.delete("/api/history/{session_id}")
async def delete_session_history(session_id: str):
    """Permanently remove a conversation from the history sidebar."""
    removed = await asyncio.to_thread(history_store.delete_session, session_id)
    _session_store.delete(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Session not found in history")
    return {"status": "deleted", "session_id": session_id}


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Streaming chat WebSocket endpoint.

    Incoming JSON:  { "session_id": "...", "message": "user text" }
    Outgoing JSON (streamed):
        { "type": "token",  "content": "partial text" }
        { "type": "done",   "content": "" }
        { "type": "error",  "content": "error description" }
    """
    await websocket.accept()
    client = websocket.client
    logger.info(f"WebSocket connected: {client}")

    try:
        while True:
            # ── Receive message ────────────────────────────────────────────
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {client}")
                break

            # ── Parse & validate JSON ──────────────────────────────────────
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON — expected {session_id, message}")
                continue

            if not isinstance(data, dict) or not isinstance(data.get("session_id", ""), str) or not isinstance(data.get("message", ""), str):
                await _send_error(websocket, "Expected string session_id and message fields")
                continue

            session_id = data.get("session_id", "").strip()
            user_msg   = data.get("message", "").strip()

            if not session_id:
                await _send_error(websocket, "Missing 'session_id' field")
                continue
            if not user_msg:
                await _send_error(websocket, "Missing or empty 'message' field")
                continue
            if len(user_msg) > 2000:
                await _send_error(websocket, "Message too long (max 2000 characters)")
                continue

            # ── Get or create session ──────────────────────────────────────
            session = _session_store.get_or_create(session_id)
            logger.info(f"[{session_id[:8]}] User: {user_msg[:80]!r}")

            # ── Stream response ────────────────────────────────────────────
            try:
                async for token in _conv_manager.handle_turn(session, user_msg):
                    await websocket.send_json({"type": "token", "content": token})

                async with session.lock:
                    await asyncio.to_thread(history_store.save_session, session)
                await websocket.send_json({"type": "done", "content": ""})
                logger.info(f"[{session_id[:8]}] Response complete")

            except LLMEngineError as e:
                logger.error(f"LLMEngineError in session {session_id[:8]}: {e}")
                async with session.lock:
                    await asyncio.to_thread(history_store.save_session, session)
                await _send_error(websocket, f"Model error: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error in session {session_id[:8]}: {e}")
                async with session.lock:
                    await asyncio.to_thread(history_store.save_session, session)
                await _send_error(websocket, "An unexpected error occurred. Please try again.")

    except Exception as e:
        logger.exception(f"Fatal WebSocket error: {e}")
    finally:
        logger.info(f"WebSocket handler exiting for {client}")


async def _send_error(websocket: WebSocket, message: str):
    """Send a structured error frame over the WebSocket."""
    try:
        await websocket.send_json({"type": "error", "content": message})
    except Exception:
        pass  # Connection may already be closed
