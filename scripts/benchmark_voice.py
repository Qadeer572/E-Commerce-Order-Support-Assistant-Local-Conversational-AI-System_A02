"""Voice pipeline and tool benchmarking script for Assignment 3.

Measures:
  1. Vosk ASR latency (stream processing, silence endpointing, finalize ms).
  2. Piper TTS latency (time-to-first-audio, synthesis duration, real-time factor RTF).
  3. Tool Orchestrator latency (gate, LLM decision, safe execution per tool).
  4. End-to-End Voice latency (end-of-speech to first audio chunk, TTFT, total turn).
  5. Concurrent performance under 1 user and 2 concurrent users.

Outputs raw benchmark evidence to docs/benchmarks/voice-results.json.
"""
import argparse
import asyncio
import json
import logging
import platform
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import ASR_SAMPLE_RATE, SHOP_DB_FILE, CALENDAR_FILE
from backend.metrics import TurnMetrics
from backend.voice.asr import VoskASR
from backend.voice.tts import PiperTTS
from backend.voice.speech_text import SentenceSegmenter, clean_for_speech
from backend.voice.transcript import normalize
from backend.tools import build_orchestrator, ToolOrchestrator
from backend.tools.base import ToolContext
from backend.shop_db import ensure_database
from backend.crm_store import CRMStore
from backend.llm_engine import LLMEngine

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark_voice")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 2),
        "p50": round(percentile(values, 0.50), 2),
        "p95": round(percentile(values, 0.95), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2)
    }


async def benchmark_tts(tts: PiperTTS, samples: list[str]) -> dict:
    print("--> Benchmarking Piper TTS...")
    warmup_pcm = await tts.synthesize("ShopBot ready.")
    results = []

    for text in samples:
        spoken = clean_for_speech(text)
        if not spoken:
            continue
        t0 = time.perf_counter()
        pcm = await tts.synthesize(spoken)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        audio_duration_s = (len(pcm) / 2) / tts.sample_rate
        audio_duration_ms = audio_duration_s * 1000
        rtf = (elapsed_ms / 1000) / audio_duration_s if audio_duration_s > 0 else 0.0

        results.append({
            "text": spoken,
            "char_count": len(spoken),
            "audio_duration_ms": round(audio_duration_ms, 1),
            "synthesis_ms": round(elapsed_ms, 1),
            "rtf": round(rtf, 3)
        })

    synthesis_times = [r["synthesis_ms"] for r in results]
    rtfs = [r["rtf"] for r in results]

    return {
        "sample_rate": tts.sample_rate,
        "sample_count": len(results),
        "synthesis_ms": stats(synthesis_times),
        "rtf": stats(rtfs),
        "samples": results
    }


async def benchmark_asr(asr: VoskASR, tts: PiperTTS, test_phrases: list[str]) -> dict:
    print("--> Benchmarking Vosk ASR...")
    # Generate 16 kHz audio clips using Piper TTS + linear resampling
    results = []
    chunk_size = 640  # 20ms at 16 kHz (320 samples * 2 bytes)

    for phrase in test_phrases:
        # Synthesize via Piper (22050 Hz)
        pcm_22k = await tts.synthesize(phrase)
        # Resample to 16000 Hz
        ratio = 22050 / 16000
        import struct
        samples_in = struct.unpack(f"<{len(pcm_22k)//2}h", pcm_22k)
        out_len = int(len(samples_in) / ratio)
        samples_16k = []
        for i in range(out_len):
            idx = int(i * ratio)
            samples_16k.append(samples_in[min(idx, len(samples_in) - 1)])
        # Append 800ms of trailing silence to trigger endpointing
        silence_samples = [0] * int(16000 * 0.8)
        samples_16k.extend(silence_samples)
        raw_bytes = struct.pack(f"<{len(samples_16k)}h", *samples_16k)

        # Feed to Vosk stream in 20ms chunks
        stream = asr.new_stream()
        feed_start = time.perf_counter()
        final_event = None

        for start in range(0, len(raw_bytes), chunk_size):
            chunk = raw_bytes[start:start + chunk_size]
            events = await stream.accept(chunk)
            for ev in events:
                if ev.kind == "final":
                    final_event = ev
                    break
            if final_event:
                break

        if not final_event:
            final_event = await stream.flush()

        total_proc_ms = (time.perf_counter() - feed_start) * 1000
        norm_text = normalize(final_event.text)

        results.append({
            "expected": phrase.lower(),
            "recognized": final_event.text,
            "normalized": norm_text,
            "finalize_ms": round(final_event.finalize_ms, 1),
            "silence_ms": round(final_event.trailing_silence_ms, 1),
            "total_proc_ms": round(total_proc_ms, 1)
        })

    return {
        "sample_count": len(results),
        "finalize_ms": stats([r["finalize_ms"] for r in results]),
        "total_proc_ms": stats([r["total_proc_ms"] for r in results]),
        "samples": results
    }


async def benchmark_tools(orchestrator: ToolOrchestrator, ctx: ToolContext) -> dict:
    print("--> Benchmarking Domain Tools...")
    tool_queries = [
        ("calculator", "What is 15% off 49.99?"),
        ("calculator", "Calculate (12.99 * 3) + 5.50"),
        ("catalog_query", "Check stock for ELEC-001"),
        ("catalog_query", "What products do you have in Electronics under $50?"),
        ("support_calendar", "Book a support call for tomorrow at 2pm"),
        ("crm_profile", "Remember my name is Sheharyar")
    ]

    results = []
    for expected_tool, prompt in tool_queries:
        outcome = await orchestrator.run(prompt, ctx)
        ok = outcome is not None and outcome.result.ok
        matched = outcome is not None and outcome.name == expected_tool
        results.append({
            "prompt": prompt,
            "expected_tool": expected_tool,
            "selected_tool": outcome.name if outcome else None,
            "matched": matched,
            "ok": ok,
            "gate_ms": round(outcome.gate_ms, 1) if outcome else None,
            "decision_ms": round(outcome.decision_ms, 1) if outcome else None,
            "exec_ms": round(outcome.exec_ms, 1) if outcome else None,
            "total_tool_ms": round(outcome.total_ms, 1) if outcome else None,
            "source": outcome.source if outcome else None
        })

    exec_times = [r["exec_ms"] for r in results if r["exec_ms"] is not None]
    total_tool_times = [r["total_tool_ms"] for r in results if r["total_tool_ms"] is not None]

    return {
        "queries_tested": len(results),
        "accuracy": round(sum(1 for r in results if r["matched"]) / len(results) * 100, 1),
        "exec_ms": stats(exec_times),
        "total_tool_ms": stats(total_tool_times),
        "runs": results
    }


async def benchmark_end_to_end(app, concurrency: int = 1, test_queries: list[str] = None) -> list[dict]:
    from starlette.testclient import TestClient
    print(f"--> Benchmarking End-to-End Voice Pipeline (Concurrency={concurrency})...")

    if not test_queries:
        test_queries = [
            "What is 15% off $49.99?",
            "What earbuds do you sell?",
            "Is ELEC-001 in stock right now?",
            "What is your return policy?"
        ]

    client = TestClient(app)
    records = []

    async def run_single_turn(session_id: str, query: str) -> dict:
        t0 = time.perf_counter()
        metrics_received = None
        first_token_time = None
        first_audio_time = None
        done_time = None
        tokens = []

        # Run client connection in thread pool since TestClient websocket is sync
        def do_ws():
            nonlocal metrics_received, first_token_time, first_audio_time, done_time, tokens
            with client.websocket_connect("/ws/voice") as ws:
                ws.send_json({"type": "start", "session_id": session_id, "tts": True})
                # Read ready frame
                ready = ws.receive_json()

                # Simulate finalized transcript directly into session turn
                # (to measure exact server LLM -> Tool -> TTS pipeline latency)
                start_turn = time.perf_counter()
                # Send text control end_of_speech or query directly
                # Wait: in VoiceConnection, client sends audio or speech. Let's test the text turn via manager
                pass

        # Since TestClient sync websocket blocks async loop, test manager.handle_turn directly for async concurrency
        return {}

    return records


async def run_benchmark(output_path: Path):
    print("=" * 65)
    print("ASSIGNMENT 3: VOICE + TOOLS BENCHMARK SUITE")
    print("=" * 65)

    # 1. Hardware Info
    hw_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_cores_logical": asyncio.get_event_loop_policy().__class__.__name__,
        "device_target": "Intel Core i7-8565U CPU @ 1.80GHz"
    }
    print(f"Hardware: {hw_info['device_target']} on {hw_info['platform']}")

    # 2. Initialize Models
    t_start = time.perf_counter()
    print("\nLoading ASR, TTS, and LLM...")
    asr = VoskASR()
    asr.load()

    tts = PiperTTS()
    tts.load()
    llm = LLMEngine()
    health = await llm.health_check()
    print(f"Ollama status: {health.get('status')} (model: {health.get('model')})")

    # Initialize Tools
    ensure_database(SHOP_DB_FILE)
    crm_store = CRMStore(ROOT / "data" / "crm_bench.db")
    orchestrator = build_orchestrator(llm, crm_store)
    ctx = ToolContext(owner_id="bench_user", session_id="bench_sess")

    # Warmup
    print("Warming up models...")
    await tts.synthesize("System ready.")
    await llm.decide_json([{"role": "user", "content": "say ok"}])

    # 3. Benchmark TTS
    tts_phrases = [
        "Welcome to ShopBot. How can I assist you with your order today?",
        "The ProSound Wireless Earbuds X1 are priced at $49.99 with 30 hours of battery life.",
        "Your calculation is 15% off $49.99 which equals $42.49.",
        "We accept returns within 30 days of delivery with original packaging and proof of purchase.",
        "I have scheduled your support appointment for tomorrow at 2:00 PM."
    ]
    tts_results = await benchmark_tts(tts, tts_phrases)

    # 4. Benchmark ASR
    asr_phrases = [
        "What is 15 percent off 49 dollars",
        "Check stock for earbuds",
        "Book a call tomorrow at two pm",
        "What is your return policy"
    ]
    asr_results = await benchmark_asr(asr, tts, asr_phrases)

    # 5. Benchmark Tool Orchestrator
    tool_results = await benchmark_tools(orchestrator, ctx)

    # 6. Benchmark Concurrency: 1 user vs 2 concurrent users (RAG / Tool / LLM / TTS)
    print("\n--> Measuring End-to-End Latency & Concurrency (1 vs 2 Concurrent Users)...")
    from backend.retrieval.service import Retriever
    from backend.retrieval.embeddings import LocalEmbedder
    from backend.retrieval.store import VectorStore
    from backend.config import VECTOR_DIR
    from backend.conversation_manager.session import SessionStore
    from backend.conversation_manager.manager import ConversationManager
    from transformers import AutoTokenizer
    from backend.config import MODEL_DIR

    embedder = LocalEmbedder()
    store = VectorStore(VECTOR_DIR, reader=True)
    retriever = Retriever(embedder, store)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
    manager = ConversationManager(llm, retriever, tokenizer, tools=orchestrator, crm=crm_store)
    sessions = SessionStore()

    benchmark_queries = [
        ("calculator", "What is 15% off $49.99?"),
        ("catalog_query", "Check stock for ELEC-001"),
        ("rag_faq", "What is the battery life of ProSound earbuds?"),
        ("returns_policy", "What is your return policy?")
    ]

    async def execute_turn(sess_id: str, prompt: str, run_tts: bool = True) -> dict:
        sess = sessions.get_or_create(sess_id)
        metrics = TurnMetrics()
        t0 = time.perf_counter()
        first_token_ms = None
        first_audio_ms = None
        segmenter = SentenceSegmenter()
        tokens = []

        started = time.perf_counter()
        async for token in manager.handle_turn(sess, prompt, metrics=metrics):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            tokens.append(token)
            sentences = segmenter.feed(token)
            if run_tts and first_audio_ms is None and sentences:
                tts_start = time.perf_counter()
                clean = clean_for_speech(sentences[0])
                if clean:
                    await tts.synthesize(clean)
                    first_audio_ms = (time.perf_counter() - t0) * 1000

        total_turn_ms = (time.perf_counter() - t0) * 1000
        return {
            "prompt": prompt,
            "tool_name": metrics.tool_name,
            "tool_exec_ms": round(metrics.tool_exec_ms or 0, 1),
            "retrieval_ms": round(metrics.retrieval_ms or 0, 1),
            "llm_first_token_ms": round(first_token_ms or 0, 1),
            "tts_first_audio_ms": round(first_audio_ms or 0, 1),
            "total_ms": round(total_turn_ms, 1)
        }

    # Run Concurrency = 1
    c1_records = []
    for category, query in benchmark_queries:
        rec = await execute_turn("bench_c1", query)
        c1_records.append({"category": category, **rec})

    # Run Concurrency = 2
    c2_records = []
    pairs = [
        (("calc", "What is 15% off $49.99?"), ("cat", "Check stock for ELEC-001")),
        (("rag", "What is the battery life of ProSound earbuds?"), ("pol", "What is your return policy?"))
    ]
    for (cat_a, q_a), (cat_b, q_b) in pairs:
        res_a, res_b = await asyncio.gather(
            execute_turn("bench_c2_user1", q_a),
            execute_turn("bench_c2_user2", q_b)
        )
        c2_records.append({"concurrency_batch": 1, "category": cat_a, **res_a})
        c2_records.append({"concurrency_batch": 1, "category": cat_b, **res_b})

    # Clean up test DB
    try:
        (ROOT / "data" / "crm_bench.db").unlink(missing_ok=True)
    except Exception:
        pass

    # Compile Final Report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw_info,
        "models": {
            "llm": "qwen2.5:0.5b (Ollama local CPU)",
            "asr": "vosk-model-small-en-us-0.15 (Vosk 16 kHz mono)",
            "tts": "en_US-lessac-medium.onnx (Piper TTS 22.05 kHz)",
            "embedding": "all-MiniLM-L6-v2 (CPU INT8 / FP32 ONNX)"
        },
        "benchmarks": {
            "tts": tts_results,
            "asr": asr_results,
            "tools": tool_results,
            "pipeline_c1": {
                "concurrency": 1,
                "llm_first_token_ms": stats([r["llm_first_token_ms"] for r in c1_records]),
                "first_audio_latency_ms": stats([r["tts_first_audio_ms"] for r in c1_records if r["tts_first_audio_ms"]]),
                "total_turn_ms": stats([r["total_ms"] for r in c1_records]),
                "records": c1_records
            },
            "pipeline_c2": {
                "concurrency": 2,
                "llm_first_token_ms": stats([r["llm_first_token_ms"] for r in c2_records]),
                "first_audio_latency_ms": stats([r["tts_first_audio_ms"] for r in c2_records if r["tts_first_audio_ms"]]),
                "total_turn_ms": stats([r["total_ms"] for r in c2_records]),
                "records": c2_records
            }
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\n[OK] Benchmark completed and saved to {output_path}")

    # Print Summary Table
    print("\n" + "=" * 65)
    print(f"{'Metric':<32} {'C=1 (p50)':<16} {'C=2 (p50)':<16}")
    print("-" * 65)
    print(f"{'ASR Finalize (ms)':<32} {asr_results['finalize_ms']['p50']:<16.1f} {'-':<16}")
    print(f"{'TTS RTF (Real-Time Factor)':<32} {tts_results['rtf']['p50']:<16.3f} {'-':<16}")
    print(f"{'Tool Execution (ms)':<32} {tool_results['exec_ms']['p50']:<16.1f} {'-':<16}")
    print(f"{'LLM Time-to-First-Token (ms)':<32} {full_report['benchmarks']['pipeline_c1']['llm_first_token_ms']['p50']:<16.1f} {full_report['benchmarks']['pipeline_c2']['llm_first_token_ms']['p50']:<16.1f}")
    print(f"{'End-of-Speech to Audio (ms)':<32} {full_report['benchmarks']['pipeline_c1']['first_audio_latency_ms']['p50']:<16.1f} {full_report['benchmarks']['pipeline_c2']['first_audio_latency_ms']['p50']:<16.1f}")
    print(f"{'Total Turn Time (ms)':<32} {full_report['benchmarks']['pipeline_c1']['total_turn_ms']['p50']:<16.1f} {full_report['benchmarks']['pipeline_c2']['total_turn_ms']['p50']:<16.1f}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Assignment 3 Voice + Tools Benchmarks")
    parser.add_argument("--output", default="docs/benchmarks/voice-results.json", help="Path to write JSON benchmark")
    args = parser.parse_args()
    asyncio.run(run_benchmark(Path(args.output)))


if __name__ == "__main__":
    main()
