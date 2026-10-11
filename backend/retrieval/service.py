"""Bounded CPU retrieval, exact entity filtering, and revision-aware LRU cache."""
import asyncio
import logging
import re
import threading
import time
import queue
from collections import OrderedDict
from concurrent.futures import Future

from backend.config import RAG_THRESHOLD, RAG_TIMEOUT
from .types import RetrievalResult

log = logging.getLogger(__name__)


def extract_order_id(text):
    match = re.search(r"\b(?:ORD[-\s]?)?(\d{5})\b", text, re.I)
    return f"ORD-{match[1]}" if match else None


def is_entity_followup(text):
    from backend.conversation_manager.intent_triage import is_support_followup
    return is_support_followup(text, "PRODUCT_QUERY")


class DaemonExecutor:
    """Fixed daemon workers: stuck native work cannot prevent process shutdown."""
    def __init__(self, workers):
        self.jobs = queue.Queue()
        self.closed = False
        self.threads = [threading.Thread(target=self._worker, daemon=True, name=f"rag-{i}") for i in range(workers)]
        for thread in self.threads:
            thread.start()

    def _worker(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            future, fn, args = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as error:
                future.set_exception(error)

    def submit(self, fn, *args):
        if self.closed:
            raise RuntimeError("Executor closed")
        future = Future()
        self.jobs.put((future, fn, args))
        return future

    def shutdown(self, **kwargs):
        self.closed = True
        while True:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            if job:
                job[0].cancel()
        for _ in self.threads:
            self.jobs.put(None)
        deadline = time.monotonic() + .2
        for thread in self.threads:
            thread.join(max(0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self.threads)


def build_retrieval_query(user_message, session, intent):
    from backend.conversation_manager.intent_triage import is_catalog_request, normalize_shopping_text
    user_message = normalize_shopping_text(user_message)
    if intent == "PRODUCT_QUERY" and is_catalog_request(user_message):
        return "Store product catalog list prices affordable cheap budget shopping options " + user_message
    if intent == "ORDER_TRACKING":
        id = extract_order_id(user_message) or session.last_order_id
        return f"{id or ''} {user_message}"
    if intent == "PRODUCT_QUERY" and session.last_sku and is_entity_followup(user_message):
        return f"{session.last_sku} {user_message}"
    return user_message


class Retriever:
    def __init__(self, embedder=None, store=None, *, threshold=RAG_THRESHOLD,
                 timeout=RAG_TIMEOUT, max_jobs=8, workers=2, ttl=300, cache_size=128):
        self.embedder, self.store = embedder, store
        self.threshold, self.timeout = threshold, timeout
        self.ttl, self.cache_size = ttl, cache_size
        self.cache = OrderedDict()
        self.executor = DaemonExecutor(workers)
        self.slots = threading.BoundedSemaphore(max_jobs)
        self.closed = False
        self.timings = {}

    @property
    def ready(self):
        return self.embedder is not None and self.store is not None and not self.closed

    def _search(self, query, top_k, filters, queued):
        start = time.perf_counter()
        vectors = self.embedder.encode([query])
        encoded = time.perf_counter()
        hits = self.store.search(vectors[0],top_k=top_k,filters=filters)
        finished = time.perf_counter()
        return hits, dict(queue_ms=(start-queued)*1000,embedding_ms=(encoded-start)*1000,
                          search_ms=(finished-encoded)*1000)

    async def retrieve(self, query: str, *, doc_types: tuple[str, ...], sku: str | None = None,
                       order_id: str | None = None, top_k: int = 3) -> RetrievalResult:
        started = time.perf_counter()
        components = {}
        def result(status, hits=None, cached=False):
            elapsed = (time.perf_counter()-started)*1000
            log.info("RAG status=%s elapsed_ms=%.2f cache=%s",status,elapsed,cached)
            return RetrievalResult(status,hits or [],elapsed,cached,components)
        if not self.ready:
            return result("unavailable")
        # A missing ID searches an impossible exact filter rather than other orders.
        if "order" in doc_types and not order_id:
            order_id = "__missing_order_id__"
        clauses = [{"doc_type": {"$in": list(doc_types)}}]
        if order_id:
            clauses.append({"order_id": order_id})
        if sku:
            clauses.append({"sku": sku})
        filters = {"$and":clauses} if len(clauses)>1 else clauses[0]
        key = (query,doc_types,sku,order_id,top_k,self.threshold,self.embedder.revision,self.store.revision)
        cached = self.cache.get(key)
        if cached and time.monotonic()-cached[0] < self.ttl:
            self.cache.move_to_end(key)
            return result(cached[1],list(cached[2]),True)
        if not self.slots.acquire(blocking=False):
            return result("busy")
        try:
            future = self.executor.submit(self._search,query,top_k,filters,started)
        except RuntimeError:
            self.slots.release()
            return result("unavailable")
        future.add_done_callback(lambda _: self.slots.release())
        try:
            wrapped = asyncio.wrap_future(future)
            # Consume late errors after timeouts; the client has already received a fallback.
            wrapped.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            hits, timings = await asyncio.wait_for(asyncio.shield(wrapped),self.timeout)
            self.timings = timings
            components = timings
        except asyncio.TimeoutError:
            return result("timeout")
        except Exception:
            log.exception("Local retrieval failed")
            return result("unavailable")
        hits = list({h.chunk_id:h for h in hits if h.similarity >= self.threshold}.values())
        status = "ok" if hits else "no_match"
        self.cache[key] = (time.monotonic(),status,hits)
        self.cache.move_to_end(key)
        while len(self.cache)>self.cache_size:
            self.cache.popitem(last=False)
        return result(status,hits)

    def close(self):
        self.closed = True
        idle = self.executor.shutdown(wait=True,cancel_futures=True)
        # Keep process ownership until exit if native work still touches the store.
        if idle and self.store and hasattr(self.store,"close"):
            self.store.close()
