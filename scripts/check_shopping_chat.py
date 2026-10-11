"""Exercise shopping regressions and time real local WebSocket responses."""
import asyncio
import json
import time
import uuid
from pathlib import Path

import websockets


async def main():
    records = []
    questions = [
        "What is the battery life of ProSound earbuds?",
        "i need cheap product",
        "list if prodcut",
        "what can I buy under $20?",
        "do you sell telescopes?",
        "What is the capital of Pakistan?",
    ]
    for question in questions:
        started = time.perf_counter()
        first = None
        parts = []
        async with websockets.connect("ws://127.0.0.1:8000/ws/chat") as ws:
            await ws.send(json.dumps({"session_id": str(uuid.uuid4()), "message": question}))
            while True:
                frame = json.loads(await asyncio.wait_for(ws.recv(), 90))
                if frame["type"] == "token":
                    first = first or time.perf_counter()
                    parts.append(frame["content"])
                else:
                    records.append({"question": question, "terminal": frame["type"],
                                    "ttft_ms": (first-started)*1000 if first else None,
                                    "total_ms": (time.perf_counter()-started)*1000,
                                    "answer": "".join(parts)})
                    print(json.dumps(records[-1]), flush=True)
                    break
    Path("docs/benchmarks/shopping-speed-after.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    assert all(r['terminal'] == 'done' for r in records)
    assert '30' in records[0]['answer'] and 'Sources:' in records[0]['answer']
    assert '$' in records[1]['answer'] and 'Sources:' in records[1]['answer']
    assert 'CablePro' in records[2]['answer'] and 'Sources:' in records[2]['answer']
    import re
    prices = re.findall(r'\$(\d+(?:\.\d+)?)', records[3]['answer'])
    assert prices and all(float(p) <= 20 for p in prices)
    assert 'reliable information' in records[4]['answer']
    assert 'sell telescopes' not in records[4]['answer']
    assert 'Sources:' not in records[4]['answer']


if __name__ == "__main__":
    asyncio.run(main())
