"""/ws/voice: microphone audio in, transcript + streamed text + spoken audio out.

Voice is not a separate system: the final transcript goes through the same
ConversationManager.handle_turn and SessionStore as typed chat. Text frames never wait for audio.

Client -> server: {"type":"start","session_id",...}, binary PCM16 mono 16 kHz frames,
                  {"type":"end_of_speech"}, {"type":"stop_audio"}
                  Set review_transcript=true on start to defer chat until
                  {"type":"confirm_transcript","text":"edited text"}.
Server -> client: ready, partial_transcript, final_transcript, no_speech, tool_call, tool_result,
                  token, audio, done, audio_end, metrics, tts_error, error
"""
import asyncio
import base64
import json
import logging
import time
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from backend import history_store
from backend.config import AUDIO_CHUNK_SAMPLES, VOICE_MAX_UTTERANCE_S, ASR_SAMPLE_RATE
from backend.llm_engine import LLMEngineError
from backend.metrics import TurnMetrics
from .asr import ASREvent, ASRError
from .speech_text import SentenceSegmenter, clean_for_speech
from .transcript import normalize
from .tts import TTSError

logger = logging.getLogger("shopbot.voice")
MAX_MESSAGE_CHARS = 2000
SENTENCE_QUEUE_SIZE = 16
NOT_INSTALLED = "Voice models are not installed; run scripts/prepare_voice_models.py"


class VoiceConnection:
    """State and behavior of one /ws/voice socket."""

    def __init__(self, websocket: WebSocket, asr, tts, session_store, manager):
        self.ws, self.asr, self.tts = websocket, asr, tts
        self.sessions, self.manager = session_store, manager
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.speak = True
        self.review_transcript = False
        self.pending_transcript: Optional[ASREvent] = None
        self.stream = None
        self.turn_task: Optional[asyncio.Task] = None
        self.audio_stopped = asyncio.Event()
        self.utterance_ms = 0.0
        self._send_lock = asyncio.Lock()

    async def send(self, frame: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(frame)

    @property
    def busy(self) -> bool:
        return self.turn_task is not None and not self.turn_task.done()

    async def run(self) -> None:
        if self.asr is None or not self.asr.ready:
            await self.send({"type": "error", "content": NOT_INSTALLED})
            await self.ws.close()
            return
        try:
            await self._receive_loop()
        except WebSocketDisconnect:
            logger.info("Voice client disconnected")
        finally:
            await self._shutdown()

    async def _receive_loop(self) -> None:
        while True:
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if message.get("bytes") is not None:
                if self.session_id and not self.busy and self.pending_transcript is None:
                    await self._on_audio(message["bytes"])
            elif message.get("text") is not None:
                await self._on_control(message["text"])

    async def _on_control(self, raw: str) -> None:
        try:
            data = json.loads(raw)
            kind = data["type"]
        except (json.JSONDecodeError, KeyError, TypeError):
            await self.send({"type": "error", "content": "Invalid control message"})
            return
        if kind == "start":
            await self._on_start(data)
        elif self.session_id is None:
            await self.send({"type": "error", "content": "Send a start message first"})
        elif kind == "confirm_transcript":
            text = data.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_CHARS:
                await self.send({"type": "error", "content": "Enter a transcript between 1 and 2000 characters"})
            elif self.pending_transcript is None or self.busy:
                await self.send({"type": "error", "content": "No voice transcript is awaiting confirmation"})
            else:
                final, self.pending_transcript = self.pending_transcript, None
                self.turn_task = asyncio.create_task(self._run_turn(time.perf_counter(), final, text.strip()))
        elif kind == "end_of_speech":
            if self.pending_transcript is not None:
                return
            if not self.busy and self.stream is not None:
                received = time.perf_counter()
                await self._finish_utterance(received, await self._safe(self.stream.flush()))
        elif kind == "stop_audio":
            self.audio_stopped.set()
        else:
            await self.send({"type": "error", "content": f"Unknown message type: {kind}"})

    async def _on_start(self, data: dict) -> None:
        session_id, user_id = data.get("session_id"), data.get("user_id")
        if not isinstance(session_id, str) or not session_id.strip() or (
                user_id is not None and (not isinstance(user_id, str) or len(user_id) > 100)):
            await self.send({"type": "error", "content": "Expected string session_id (and optional user_id)"})
            return
        self.session_id = session_id.strip()
        self.review_transcript = data.get("review_transcript") is True
        self.pending_transcript = None
        self.user_id = user_id.strip() if user_id else None
        self.speak = bool(data.get("tts", True)) and self.tts is not None and self.tts.ready
        if self.stream is None:
            self.stream = self.asr.new_stream()
        await self.send({"type": "ready", "asr": True, "tts": self.speak,
                         "sample_rate": self.tts.sample_rate if self.tts is not None else 22050})

    async def _safe(self, awaitable) -> Optional[ASREvent]:
        """Await an ASR call; a recognizer error becomes an error frame and the socket survives."""
        try:
            return await awaitable
        except ASRError as error:
            logger.error("%s", error)
            await self.send({"type": "error", "content": str(error)})
            return None

    async def _on_audio(self, chunk: bytes) -> None:
        received = time.perf_counter()
        self.utterance_ms += len(chunk) / 2 * 1000 / ASR_SAMPLE_RATE
        try:
            events = await self.stream.accept(chunk)
        except ASRError as error:
            logger.error("%s", error)
            await self.send({"type": "error", "content": str(error)})
            return
        for event in events:
            if event.kind == "partial":
                await self.send({"type": "partial_transcript", "text": event.text})
            else:
                await self._finish_utterance(received, event)
                return
        if self.utterance_ms > VOICE_MAX_UTTERANCE_S * 1000:
            await self._finish_utterance(received, await self._safe(self.stream.flush()))

    async def _finish_utterance(self, detected_at: float, final: Optional[ASREvent]) -> None:
        self.utterance_ms = 0.0
        if final is None:
            return
        if not final.text.strip():
            await self.send({"type": "no_speech"})
            return
        if self.review_transcript:
            self.pending_transcript = final
            await self.send({"type": "final_transcript", "text": final.text.strip() if final.text_is_formatted else normalize(final.text),
                             "raw_text": final.text, "requires_confirmation": True,
                             "asr_ms": round(final.finalize_ms, 1)})
            return
        self.turn_task = asyncio.create_task(self._run_turn(detected_at, final))

    async def _run_turn(self, detected_at: float, final: ASREvent, confirmed_text: Optional[str] = None) -> None:
        metrics = TurnMetrics(asr_endpoint_silence_ms=final.trailing_silence_ms, asr_finalize_ms=final.finalize_ms)
        text = confirmed_text if confirmed_text is not None else (
            final.text.strip() if final.text_is_formatted else normalize(final.text))
        await self.send({"type": "final_transcript", "text": text, "raw_text": final.text,
                         "asr_ms": round(final.finalize_ms, 1)})
        logger.info("[%s] voice raw=%r normalized=%r", self.session_id[:8], final.text, text)
        if len(text) > MAX_MESSAGE_CHARS:
            await self.send({"type": "error", "content": "Message too long (max 2000 characters)"})
            return
        session = self.sessions.get_or_create(self.session_id)
        self.audio_stopped.clear()
        sentences: asyncio.Queue = asyncio.Queue(maxsize=SENTENCE_QUEUE_SIZE)
        timing = {"first_sentence": None}
        consumer = asyncio.create_task(self._speak(sentences, metrics, timing, detected_at))
        try:
            await self._stream_text(session, text, sentences, metrics, timing)
        except LLMEngineError as error:
            logger.error("LLMEngineError in voice session: %s", error)
            await self._persist(session)
            await self.send({"type": "error", "content": f"Model error: {error}"})
        except Exception:
            logger.exception("Unexpected voice turn error")
            await self._persist(session)
            await self.send({"type": "error", "content": "An unexpected error occurred. Please try again."})
        finally:
            await sentences.put(None)
            await consumer
        await self.send({"type": "audio_end"})
        await self.send(metrics.to_frame())
        logger.info("[%s] voice metrics %s", self.session_id[:8], metrics.to_frame())

    async def _stream_text(self, session, text, sentences, metrics, timing) -> None:
        segmenter = SentenceSegmenter()

        def enqueue(batch: list[str]) -> None:
            for sentence in batch:
                if not self.speak:
                    return
                if timing["first_sentence"] is None:
                    timing["first_sentence"] = time.perf_counter()
                try:
                    sentences.put_nowait(sentence)
                except asyncio.QueueFull:
                    logger.warning("TTS queue full; dropping one sentence of audio")  # never drop text

        async for token in self.manager.handle_turn(session, text, user_id=self.user_id, metrics=metrics,
                                                    on_event=self.send):
            await self.send({"type": "token", "content": token})
            enqueue(segmenter.feed(token))
        enqueue(segmenter.flush())
        await self._persist(session)
        await self.send({"type": "done", "content": ""})

    async def _speak(self, sentences: asyncio.Queue, metrics: TurnMetrics, timing: dict, detected_at: float) -> None:
        """Synthesize queued sentences in order and stream PCM frames; text never waits for this."""
        seq, warned = 0, False
        frame_bytes = AUDIO_CHUNK_SAMPLES * 2
        while True:
            sentence = await sentences.get()
            if sentence is None:
                return
            if self.audio_stopped.is_set():
                continue
            spoken = clean_for_speech(sentence)
            if not spoken:
                continue
            try:
                pcm = await self.tts.synthesize(spoken)
            except TTSError as error:
                logger.error("%s", error)
                if not warned:
                    warned = True
                    await self._try_send({"type": "tts_error", "content": "Speech synthesis failed for part of the reply."})
                continue
            for start in range(0, len(pcm), frame_bytes):
                if self.audio_stopped.is_set():
                    break
                if not await self._try_send({"type": "audio", "seq": seq, "sample_rate": self.tts.sample_rate,
                                             "format": "pcm_s16le",
                                             "data": base64.b64encode(pcm[start:start + frame_bytes]).decode()}):
                    return
                if seq == 0:
                    now = time.perf_counter()
                    metrics.tts_first_audio_ms = (now - timing["first_sentence"]) * 1000
                    metrics.end_of_speech_to_first_audio_ms = (now - detected_at) * 1000
                    metrics.user_stop_to_first_audio_ms = (
                        metrics.end_of_speech_to_first_audio_ms + (metrics.asr_endpoint_silence_ms or 0))
                seq += 1

    async def _try_send(self, frame: dict) -> bool:
        try:
            await self.send(frame)
            return True
        except Exception:
            return False

    async def _persist(self, session) -> None:
        async with session.lock:
            await asyncio.to_thread(history_store.save_session, session)

    async def _shutdown(self) -> None:
        """Cancel any in-flight turn, then persist so a mid-turn disconnect never loses history."""
        task = self.turn_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self.session_id:
            session = self.sessions.get(self.session_id)
            if session is not None:
                try:
                    await self._persist(session)
                except Exception:
                    logger.exception("Could not persist voice session on disconnect")


async def handle_voice(websocket: WebSocket, asr, tts, session_store, manager) -> None:
    await websocket.accept()
    await VoiceConnection(websocket, asr, tts, session_store, manager).run()
