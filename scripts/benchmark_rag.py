"""Real CPU retrieval measurements; generation is measured separately over WebSocket."""
import argparse
import asyncio
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def percentile(values,p):
    values=sorted(values)
    pos=(len(values)-1)*p
    lower=int(pos)
    return values[lower]+(values[min(lower+1,len(values)-1)]-values[lower])*(pos-lower)


def summarize(records):
    result={}
    for name,cached in [('uncached',False),('cached',True)]:
        values=[r['elapsed_ms'] for r in records if r['cache_hit']==cached]
        result[name]=dict(count=len(values),p50_ms=percentile(values,.5),p95_ms=percentile(values,.95)) if values else dict(count=0)
    return result


async def websocket_measure(url,concurrency):
    import websockets
    import uuid
    async def one(i):
        started=time.perf_counter(); first=None; chunks=[]
        session_id = str(uuid.uuid4())
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(dict(session_id=session_id,message="What is the battery life of ProSound earbuds?")))
            while True:
                frame=json.loads(await asyncio.wait_for(ws.recv(),120))
                if frame['type']=='token':
                    if first is None: first=time.perf_counter()
                    chunks.append(frame['content'])
                else:
                    ended=time.perf_counter()
                    return dict(session_id=session_id,terminal=frame['type'],ttft_ms=(first-started)*1000 if first else None,
                                generation_ms=(ended-first)*1000 if first else None,total_ms=(ended-started)*1000,
                                answer=''.join(chunks))
    return await asyncio.gather(*(one(i) for i in range(concurrency)))


async def run(args):
    if args.generation_only:
        if not args.websocket:
            raise ValueError("--generation-only requires --websocket")
        result = await websocket_measure(args.websocket,args.concurrency)
        output = Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        return
    from backend.config import VECTOR_DIR,MODEL_DIR,RAG_THRESHOLD
    from backend.retrieval.embeddings import LocalEmbedder
    from backend.retrieval.store import VectorStore
    from backend.retrieval.service import Retriever
    cold=time.perf_counter()
    embedder=LocalEmbedder(); store=VectorStore(VECTOR_DIR,reader=True)
    cold_ms=(time.perf_counter()-cold)*1000
    retriever=Retriever(embedder,store,threshold=args.threshold,timeout=5)
    queries=json.loads(Path(args.queries).read_text())
    records=[]; quality=[]
    async def retrieve(q):
        return await retriever.retrieve(q['query'],doc_types=tuple(q['doc_types']),order_id=q.get('order_id'))
    try:
        for q in queries:
            r=await retrieve(q)
            ids={h.chunk_id.split(':')[0] for h in r.hits}
            # Product FAQ and specification documents are both valid evidence.
            ids |= {id.removeprefix('faq-') for id in ids}
            quality.append(dict(query=q['query'],expected=q['expected'],sources=sorted(ids),status=r.status,
                                correct=bool(ids.intersection(q['expected'])) if q['expected'] else not ids))
        # Warm-up before uncached measurement; disable answer cache via unique clear per batch.
        for batch in range(0,args.iterations,args.concurrency):
            retriever.cache.clear()
            selected=[queries[(batch+i)%len(queries)] for i in range(min(args.concurrency,args.iterations-batch))]
            results=await asyncio.gather(*(retrieve(q) for q in selected))
            for r in results:
                records.append(dict(elapsed_ms=r.elapsed_ms,cache_hit=r.cache_hit,status=r.status,**r.timings))
            # Immediate identical repeats measure cache independently.
            for q in selected:
                r=await retrieve(q)
                records.append(dict(elapsed_ms=r.elapsed_ms,cache_hit=r.cache_hit,status=r.status,**r.timings))
        result=dict(hardware=dict(platform=platform.platform(),processor=platform.processor()),
                    versions={name:importlib.metadata.version(name) for name in ['torch','sentence-transformers','transformers','chromadb']},
                    revisions=json.loads((MODEL_DIR/'manifest.json').read_text()),index_revision=store.revision,
                    documents=len(store.manifest['documents']),chunks=store.collection.count(),top_k=3,
                    threshold=args.threshold,concurrency=args.concurrency,iterations=args.iterations,cold_load_ms=cold_ms,
                    retrieval=summarize(records),last_uncached_components=retriever.timings,quality=quality,
                    quality_correct=sum(q['correct'] for q in quality),quality_total=len(quality),records=records)
    finally:
        retriever.close()
    if args.websocket:
        result['generation']=await websocket_measure(args.websocket,args.concurrency)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['concurrency','cold_load_ms','retrieval','quality_correct','quality_total']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--queries',default='data/evaluation/retrieval_holdout.json')
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--concurrency',type=int,choices=[1,5],default=1)
    parser.add_argument('--threshold',type=float,default=.40)
    parser.add_argument('--output',default='docs/benchmarks/rag-results.json')
    parser.add_argument('--websocket')
    parser.add_argument('--generation-only',action='store_true')
    asyncio.run(run(parser.parse_args()))
