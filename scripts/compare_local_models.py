"""Compare installed local CPU models with identical retrieved evidence."""
import asyncio
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import VECTOR_DIR
from backend.retrieval.embeddings import LocalEmbedder, load_generation_tokenizer
from backend.retrieval.store import VectorStore
from backend.retrieval.service import Retriever
from backend.llm_engine import LLMEngine
from backend.conversation_manager.manager import ConversationManager
from backend.conversation_manager.session import Session


async def main():
    r = Retriever(LocalEmbedder(), VectorStore(VECTOR_DIR, reader=True), timeout=5)
    tokenizer = load_generation_tokenizer()
    records = []
    try:
        for model in ('qwen2.5:0.5b', 'qwen2.5:1.5b'):
            llm = LLMEngine()
            llm.model = model
            await llm.warmup()
            cm = ConversationManager(llm, r, tokenizer)
            for question in ('i need cheap product', 'What is the battery life of ProSound earbuds?', 'what can I buy under $20?'):
                start = time.perf_counter()
                first = None
                parts = []
                async for token in cm.handle_turn(Session(), question):
                    first = first or time.perf_counter()
                    parts.append(token)
                row = dict(model=model, question=question, ttft_ms=(first-start)*1000,
                           total_ms=(time.perf_counter()-start)*1000, answer=''.join(parts))
                records.append(row)
                print(json.dumps(row), flush=True)
    finally:
        r.close()
    Path('docs/benchmarks/model-comparison.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(main())
