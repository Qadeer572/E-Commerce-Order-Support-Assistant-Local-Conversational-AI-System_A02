"""Per-turn latency breakdown filled by the manager and the voice handler."""
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class TurnMetrics:
    """All durations are milliseconds; None means the stage did not run for this turn."""
    asr_endpoint_silence_ms: Optional[float] = None
    asr_finalize_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    tool_gate_ms: Optional[float] = None
    tool_decision_ms: Optional[float] = None
    tool_exec_ms: Optional[float] = None
    llm_first_token_ms: Optional[float] = None
    tts_first_audio_ms: Optional[float] = None
    end_of_speech_to_first_audio_ms: Optional[float] = None
    user_stop_to_first_audio_ms: Optional[float] = None
    total_ms: Optional[float] = None
    tool_name: Optional[str] = None
    tool_source: Optional[str] = None

    @property
    def tool_ms(self) -> Optional[float]:
        parts = [self.tool_gate_ms, self.tool_decision_ms, self.tool_exec_ms]
        return None if all(part is None for part in parts) else sum(part or 0 for part in parts)

    def to_frame(self) -> dict:
        frame = {key: (round(value, 1) if isinstance(value, float) else value) for key, value in asdict(self).items()}
        tool_ms = self.tool_ms
        frame["tool_ms"] = None if tool_ms is None else round(tool_ms, 1)
        return {"type": "metrics", **frame}
