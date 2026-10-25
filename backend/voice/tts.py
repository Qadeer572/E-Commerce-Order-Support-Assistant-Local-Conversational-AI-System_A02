"""Local text-to-speech with Piper (ONNX). One sentence in, 16-bit mono PCM out.

The voice is loaded once at startup. espeak phonemization is not thread-safe, so it runs under a
lock; the ONNX inference runs outside the lock in a dedicated bounded thread pool. The ONNX
session is re-created with a limited thread count so TTS does not starve Ollama on a shared CPU.
"""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from backend.config import TTS_MAX_JOBS, TTS_THREADS, TTS_VOICE_PATH

logger = logging.getLogger("shopbot.tts")


class TTSError(Exception):
    """Synthesis failure for one sentence; the handler skips that audio and warns once."""


class PiperTTS:
    def __init__(self, voice_path: Path = TTS_VOICE_PATH, max_jobs: int = TTS_MAX_JOBS,
                 threads: int = TTS_THREADS):
        self.voice_path, self.threads = Path(voice_path), threads
        self.executor = ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix="tts")
        self._phonemize_lock = threading.Lock()
        self.voice = None
        self.sample_rate = 22050
        self.ready = False

    def load(self) -> None:
        """Blocking voice load + warm-up; call from a thread. Leaves ready=False on failure."""
        try:
            import onnxruntime
            from piper import PiperVoice
            if not self.voice_path.is_file():
                raise FileNotFoundError(f"TTS voice not found at {self.voice_path}")
            self.voice = PiperVoice.load(self.voice_path)
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = self.threads
            self.voice.session = onnxruntime.InferenceSession(
                str(self.voice_path), sess_options=options, providers=["CPUExecutionProvider"])
            self.sample_rate = self.voice.config.sample_rate
            self.synthesize_sync("Hello.")  # warm-up so the first user does not pay start-up cost
            self.ready = True
            logger.info("Piper voice ready: %s (%d Hz)", self.voice_path.name, self.sample_rate)
        except Exception as error:
            logger.warning("TTS unavailable: %s", error)
            self.ready = False

    def synthesize_sync(self, text: str) -> bytes:
        """Return PCM16 mono bytes for the text (all of its sentences, in order)."""
        with self._phonemize_lock:
            sentences = self.voice.phonemize(text)
        pieces = []
        for phonemes in sentences:
            if not phonemes:
                continue
            audio = self.voice.phoneme_ids_to_audio(self.voice.phonemes_to_ids(phonemes))
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > 1e-8:
                pieces.append((audio / peak * 32767 * 0.95).astype(np.int16).tobytes())
        return b"".join(pieces)

    async def synthesize(self, text: str) -> bytes:
        try:
            return await asyncio.get_running_loop().run_in_executor(self.executor, self.synthesize_sync, text)
        except Exception as error:
            raise TTSError(f"Speech synthesis failed: {error}") from error

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
