"""
Configuration settings for the ShopBot backend.
All tuneable parameters are centralized here.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

PRODUCTS_FILE = DATA_DIR / "products.json"
ORDERS_FILE   = DATA_DIR / "orders.json"
POLICY_FILE   = DATA_DIR / "policy.md"

# Persisted chat history (generated at runtime — powers the frontend history sidebar)
HISTORY_FILE  = DATA_DIR / "chat_history.json"

# ── Ollama / LLM ─────────────────────────────────────────────────────────────
_raw_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
OLLAMA_HOST = (_raw_host if _raw_host.startswith("http") else f"http://{_raw_host}").replace("0.0.0.0", "127.0.0.1")
MODEL_NAME   = os.getenv("SHOPBOT_MODEL", "qwen2.5:0.5b")

# Generation parameters
TEMPERATURE  = float(os.getenv("SHOPBOT_TEMPERATURE", "0.3"))
TOP_P        = float(os.getenv("SHOPBOT_TOP_P", "0.9"))
MAX_TOKENS   = int(os.getenv("SHOPBOT_MAX_TOKENS", "512"))

# ── Context window management ─────────────────────────────────────────────────
# Approximate token budget for conversation history (system prompt takes ~500)
MAX_HISTORY_TOKENS = int(os.getenv("SHOPBOT_MAX_HISTORY_TOKENS", "1800"))
# Always keep at least this many recent turns regardless of token budget
MIN_HISTORY_TURNS  = int(os.getenv("SHOPBOT_MIN_HISTORY_TURNS", "2"))

# ── LLM engine tuning ────────────────────────────────────────────────────────
# Number of CPU threads for llama.cpp inference (0 = auto-detect)
NUM_THREADS = int(os.getenv("SHOPBOT_NUM_THREADS", "0"))
MODEL_KEEP_ALIVE = os.getenv("SHOPBOT_KEEP_ALIVE", "30m")

# ── Session ───────────────────────────────────────────────────────────────────
# Session timeout in seconds (30 minutes of inactivity)
SESSION_TIMEOUT_SECONDS = int(os.getenv("SHOPBOT_SESSION_TIMEOUT", "1800"))

# ── Server ───────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = ["*"]   # tighten in production

# Local RAG assets. Downloads occur only through scripts/prepare_models.py.
MODEL_DIR = BASE_DIR / "models"
CORPUS_DIR = DATA_DIR / "corpus"
VECTOR_DIR = DATA_DIR / "vector_store"
CONTEXT_TOKENS = 4096
RAG_THRESHOLD = float(os.getenv("SHOPBOT_RAG_THRESHOLD", "0.40"))
RAG_TIMEOUT = float(os.getenv("SHOPBOT_RAG_TIMEOUT", "1.5"))

# ── Assignment 3: voice ───────────────────────────────────────────────────────
def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


VOICE_ENABLED = _flag("SHOPBOT_VOICE_ENABLED", "1")
ASR_MODEL_DIR = Path(os.getenv("SHOPBOT_ASR_MODEL_DIR", str(MODEL_DIR / "vosk-model-small-en-us-0.15")))
ASR_SAMPLE_RATE = 16000
# Trailing silence (ms) after which the Vosk endpointer ends an utterance.
ASR_ENDPOINT_SILENCE_MS = int(os.getenv("SHOPBOT_ASR_ENDPOINT_MS", "600"))
ASR_MAX_JOBS = int(os.getenv("SHOPBOT_ASR_MAX_JOBS", "2"))
TTS_VOICE_PATH = Path(os.getenv("SHOPBOT_TTS_VOICE", str(MODEL_DIR / "piper" / "en_US-lessac-medium.onnx")))
TTS_MAX_JOBS = int(os.getenv("SHOPBOT_TTS_MAX_JOBS", "2"))
TTS_THREADS = int(os.getenv("SHOPBOT_TTS_THREADS", "2"))
AUDIO_CHUNK_SAMPLES = int(os.getenv("SHOPBOT_AUDIO_CHUNK_SAMPLES", "4000"))
VOICE_MAX_UTTERANCE_S = int(os.getenv("SHOPBOT_VOICE_MAX_UTTERANCE_S", "30"))

# ── Assignment 3: tools ───────────────────────────────────────────────────────
TOOLS_ENABLED = _flag("SHOPBOT_TOOLS_ENABLED", "1")
# Optional larger model used only for the tool-decision call.
TOOL_MODEL = os.getenv("SHOPBOT_TOOL_MODEL", "").strip() or None
TOOL_DECISION_TIMEOUT = float(os.getenv("SHOPBOT_TOOL_DECISION_TIMEOUT", "5.0"))
TOOL_TIMEOUT = float(os.getenv("SHOPBOT_TOOL_TIMEOUT", "2.0"))
TOOL_FAULT = os.getenv("SHOPBOT_TOOL_FAULT", "").strip()  # "<tool>:timeout|error"
SHOP_DB_FILE = Path(os.getenv("SHOPBOT_SHOP_DB", str(DATA_DIR / "shop.db")))
CALENDAR_FILE = Path(os.getenv("SHOPBOT_CALENDAR_FILE", str(DATA_DIR / "calendar.json")))
CRM_DB_FILE = Path(os.getenv("SHOPBOT_CRM_DB", str(DATA_DIR / "crm.db")))
