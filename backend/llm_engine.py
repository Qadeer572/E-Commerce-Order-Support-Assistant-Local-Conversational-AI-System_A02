"""
llm_engine.py — Async streaming wrapper around the Ollama API.

Provides LLMEngine.stream() which yields text tokens as they arrive
from the locally running Ollama server.  The Conversation Manager
calls this; no LLM tool-calling or retrieval pipeline is used.
"""

import time
import asyncio
import logging
from typing import AsyncIterator

import ollama

from backend.config import (
    OLLAMA_HOST,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    NUM_THREADS,
    MODEL_KEEP_ALIVE,
)

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Thin async wrapper over the Ollama client for streaming chat completions.

    The engine is stateless — all session state lives in the Session objects
    managed by the ConversationManager.
    """

    def __init__(self):
        self.client = ollama.AsyncClient(host=OLLAMA_HOST)
        self.model  = MODEL_NAME
        logger.info(f"LLMEngine initialized — model={self.model}, host={OLLAMA_HOST}")

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """
        Stream a chat completion.

        Args:
            messages: List of {role, content} dicts assembled by the PromptBuilder.

        Yields:
            Text tokens (strings) as they arrive from the model.

        Raises:
            LLMEngineError: On connection failure or model error.
        """
        t_start = time.perf_counter()
        first_token = True
        # Ordinary support answers should be brief. Full catalog lists need room.
        from backend.conversation_manager.intent_triage import is_catalog_request
        question = messages[-1]["content"] if messages else ""
        catalog_list = is_catalog_request(question) and any(
            word in question.lower().split() for word in ("list", "all", "catalog", "catalogue"))
        output_limit = min(MAX_TOKENS, 512 if catalog_list else 256)

        try:
            async for chunk in await self.client.chat(
                model   = self.model,
                messages= messages,
                stream  = True,
                keep_alive = MODEL_KEEP_ALIVE,
                options = {
                    "temperature": TEMPERATURE,
                    "top_p":       TOP_P,
                    "num_predict": output_limit,
                    "num_ctx": 4096,
                    "num_gpu": 0,
                    "num_thread":  NUM_THREADS if NUM_THREADS > 0 else None,
                },
            ):
                token = chunk.get("message", {}).get("content", "")
                if token:
                    if first_token:
                        ttft = time.perf_counter() - t_start
                        logger.info(f"Time to first token: {ttft:.3f}s")
                        first_token = False
                    yield token

            total_time = time.perf_counter() - t_start
            logger.info(f"LLM generation complete in {total_time:.2f}s")

        except ollama.ResponseError as e:
            logger.error(f"Ollama ResponseError: {e}")
            raise LLMEngineError(f"Model error: {e.error}") from e
        except Exception as e:
            logger.error(f"LLMEngine unexpected error: {e}")
            raise LLMEngineError(f"Inference failed: {str(e)}") from e

    async def warmup(self):
        """Load CPU weights ahead of the first chat; failure leaves chat available."""
        from backend.conversation_manager.prompt_builder import RULES
        try:
            await asyncio.wait_for(self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": RULES}, {"role": "user", "content": "hi"}],
                keep_alive=MODEL_KEEP_ALIVE,
                options={"num_predict": 1, "num_ctx": 4096, "num_gpu": 0}), 60)
        except Exception:
            logger.warning("Model warmup failed; inference will retry on the next chat")

    async def health_check(self) -> dict:
        """
        Verify that Ollama is reachable and the model is available.

        Returns:
            {"status": "ok", "model": ..., "available": True/False}
        """
        try:
            models_response = await self.client.list()
            model_names = [m["model"] for m in models_response.get("models", [])]
            available = any(self.model in m for m in model_names)
            return {
                "status": "ok",
                "model": self.model,
                "available": available,
                "ollama_host": OLLAMA_HOST,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.model,
                "available": False,
            }


class LLMEngineError(Exception):
    """Raised when the LLM engine encounters a non-recoverable error."""
    pass
