"""Local streaming speech recognition with Vosk (Kaldi).

One shared read-only Model is loaded at startup. Each WebSocket connection gets its own
recognizer (recognizers are not thread-safe). Recognizer calls run in a small bounded thread
pool so the event loop never blocks; each call decodes a single ~100 ms chunk.

The installed vosk wrapper (0.3.45) exposes no endpointer setters, so end-of-speech is the
earlier of (a) Vosk's built-in endpoint and (b) our own trailing-silence counter, which makes
the silence length configurable and measurable.
"""
import asyncio
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import ASR_ENDPOINT_SILENCE_MS, ASR_MAX_JOBS, ASR_MODEL_DIR, ASR_SAMPLE_RATE
from backend.config import ASR_FINAL_ENABLED, ASR_FINAL_MODEL_DIR, ASR_FINAL_THREADS, VOICE_MAX_UTTERANCE_S
from backend.config import ASR_HOTWORDS
from backend.config import ASR_DIAGNOSTIC_CAPTURE, BASE_DIR

logger = logging.getLogger("shopbot.asr")
SILENCE_RMS = 400  # int16 RMS below this counts as silence for the trailing-silence counter


class ASRError(Exception):
    """Recognizer failure; the WebSocket handler turns it into an error frame."""


@dataclass
class ASREvent:
    kind: str                        # "partial" or "final"
    text: str
    finalize_ms: float = 0.0         # chunk received -> final string ready (final events only)
    trailing_silence_ms: float = 0.0  # silence observed before the endpoint (final events only)
    text_is_formatted: bool = False


class ASRStream:
    """Per-connection recognizer wrapper."""

    def __init__(self, recognizer, executor: ThreadPoolExecutor, endpoint_silence_ms: int,
                 final_transcriber=None):
        self._rec, self._executor, self._endpoint_ms = recognizer, executor, endpoint_silence_ms
        self._last_partial = ""
        self._heard_speech = False
        self._silence_ms = 0.0
        self._final_transcriber = final_transcriber
        self._audio = bytearray()

    def _is_silent(self, chunk: bytes) -> float:
        """Update the silence counter and return the chunk duration in ms."""
        samples = np.frombuffer(chunk, dtype=np.int16)
        duration_ms = len(samples) * 1000 / ASR_SAMPLE_RATE
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
        self._silence_ms = self._silence_ms + duration_ms if rms < SILENCE_RMS else 0.0
        return duration_ms

    def _final(self, started: float, method) -> ASREvent:
        text = json.loads(method()).get("text", "")
        pcm, self._audio = self._audio, bytearray()
        silence = self._silence_ms
        self._last_partial, self._heard_speech, self._silence_ms = "", False, 0.0
        formatted = False
        if self._final_transcriber is not None:
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            # Avoid decoding empty/silent input, which generative ASR may hallucinate over.
            if len(samples) and np.max(np.abs(samples)) > 0.005:
                try:
                    text = self._final_transcriber(samples)
                    formatted = True
                except Exception:
                    logger.exception("Whisper final decoding failed; retaining Vosk draft for review")
            else:
                text = ""
        return ASREvent("final", text, (time.perf_counter() - started) * 1000, silence,
                        formatted)

    def _accept_sync(self, chunk: bytes) -> list[ASREvent]:
        started = time.perf_counter()
        if len(chunk) % 2:
            raise ValueError("Expected PCM16 audio with an even byte length")
        if self._final_transcriber is not None:
            if len(self._audio) + len(chunk) > (VOICE_MAX_UTTERANCE_S + 1) * ASR_SAMPLE_RATE * 2:
                raise ValueError("Voice utterance exceeds the audio buffer limit")
            self._audio.extend(chunk)
        self._is_silent(chunk)
        if self._rec.AcceptWaveform(chunk):  # Vosk's built-in endpoint fired
            return [self._final(started, self._rec.Result)]
        partial = json.loads(self._rec.PartialResult()).get("partial", "")
        if partial:
            self._heard_speech = True
        if self._heard_speech and self._silence_ms >= self._endpoint_ms:
            return [self._final(started, self._rec.FinalResult)]
        if partial and partial != self._last_partial:
            self._last_partial = partial
            return [ASREvent("partial", partial)]
        return []

    def _flush_sync(self) -> ASREvent:
        return self._final(time.perf_counter(), self._rec.FinalResult)

    async def accept(self, chunk: bytes) -> list[ASREvent]:
        """Feed one PCM16 mono 16 kHz chunk; returns partial/final events."""
        try:
            return await asyncio.get_running_loop().run_in_executor(self._executor, self._accept_sync, chunk)
        except Exception as error:
            raise ASRError(f"Speech recognition failed: {error}") from error

    async def flush(self) -> ASREvent:
        """Force a final transcript (client pressed stop)."""
        try:
            return await asyncio.get_running_loop().run_in_executor(self._executor, self._flush_sync)
        except Exception as error:
            raise ASRError(f"Speech recognition failed: {error}") from error


class VoskASR:
    def __init__(self, model_dir: Path = ASR_MODEL_DIR, max_jobs: int = ASR_MAX_JOBS,
                 endpoint_silence_ms: int = ASR_ENDPOINT_SILENCE_MS):
        self.model_dir, self.endpoint_silence_ms = Path(model_dir), endpoint_silence_ms
        self.executor = ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix="asr")
        self.model = None
        self.ready = False
        self.final_model = None
        self._final_lock = threading.Lock()

    def load(self) -> None:
        """Blocking model load + warm-up; call from a thread. Leaves ready=False on failure."""
        try:
            from vosk import Model, SetLogLevel
            if not self.model_dir.is_dir():
                raise FileNotFoundError(f"ASR model not found at {self.model_dir}")
            SetLogLevel(-1)
            self.model = Model(str(self.model_dir))
            self._warm_up()
            if ASR_FINAL_ENABLED:
                try:
                    from faster_whisper import WhisperModel
                    self.final_model = WhisperModel(str(ASR_FINAL_MODEL_DIR), device="cpu",
                        compute_type="int8", cpu_threads=ASR_FINAL_THREADS, num_workers=1,
                        local_files_only=True)
                    logger.info("Whisper final transcription ready: %s (CPU int8)", ASR_FINAL_MODEL_DIR.name)
                except Exception as error:
                    logger.warning("Whisper final transcription unavailable; using Vosk: %s", error)
            self.ready = True
            logger.info("Vosk model ready: %s", self.model_dir.name)
        except Exception as error:
            logger.warning("ASR unavailable: %s", error)
            self.ready = False

    def _warm_up(self) -> None:
        recognizer = self._recognizer()
        recognizer.AcceptWaveform(b"\x00\x00" * ASR_SAMPLE_RATE)  # one second of silence
        recognizer.FinalResult()

    def _recognizer(self):
        from vosk import KaldiRecognizer
        return KaldiRecognizer(self.model, ASR_SAMPLE_RATE)

    def new_stream(self) -> ASRStream:
        if not self.ready:
            raise ASRError("Speech recognition is not available")
        return ASRStream(self._recognizer(), self.executor, self.endpoint_silence_ms,
                         self._transcribe_final if self.final_model is not None else None)

    def _transcribe_final(self, samples: np.ndarray) -> str:
        # Serialize the lazy segment iterator to bound CPU/memory use across sessions.
        with self._final_lock:
            segments, _ = self.final_model.transcribe(samples, language="en", beam_size=5,
                temperature=0.0, condition_on_previous_text=False, vad_filter=True,
                hotwords=ASR_HOTWORDS or None)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if ASR_DIAGNOSTIC_CAPTURE:
                self._capture_diagnostic(samples, text)
            return text

    @staticmethod
    def _capture_diagnostic(samples: np.ndarray, text: str) -> None:
        """Opt-in single local clip; an existing capture is never overwritten."""
        import wave
        target = BASE_DIR / 'artifacts' / 'asr-diagnostic.wav'
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        pcm = (samples * 32768).clip(-32768, 32767).astype('<i2')
        with wave.open(str(target), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(ASR_SAMPLE_RATE)
            output.writeframes(pcm.tobytes())
        details = dict(transcript=text, sample_rate=ASR_SAMPLE_RATE,
                       duration_seconds=len(samples) / ASR_SAMPLE_RATE,
                       rms=float(np.sqrt(np.mean(samples ** 2))),
                       peak=float(np.max(np.abs(samples))),
                       clipped_fraction=float(np.mean(np.abs(samples) >= .999)))
        target.with_suffix('.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
        logger.info('Saved one diagnostic microphone clip locally: %s', target)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
