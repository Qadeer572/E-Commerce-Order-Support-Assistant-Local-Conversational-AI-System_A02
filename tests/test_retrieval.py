import asyncio
import time
import pytest
from backend.retrieval.service import Retriever
from backend.retrieval.types import Hit
from tests.test_indexer import Embedder


class Store:
    revision = "one"

    def search(self, embedding, *, top_k, filters):
        assert top_k == 3
        if "99999" in str(filters):
            return []
        return [Hit("chunk", "one.md", "One", "facts", .8, "order",order_id=str(filters))]


@pytest.mark.asyncio
async def test_exact_orders_and_revision_cache_are_isolated():
    store = Store()
    r = Retriever(Embedder(), store)
    try:
        missing = await r.retrieve("status", doc_types=("order",),order_id="ORD-99999")
        assert missing.status == "no_match"
        assert not missing.hits
        a = await r.retrieve("status",doc_types=("order",),order_id="ORD-10001")
        b = await r.retrieve("status",doc_types=("order",),order_id="ORD-10002")
        assert a.hits != b.hits
        assert (await r.retrieve("status",doc_types=("order",),order_id="ORD-10001")).cache_hit
        store.revision = "two"
        assert not (await r.retrieve("status",doc_types=("order",),order_id="ORD-10001")).cache_hit
    finally:
        r.close()


@pytest.mark.asyncio
async def test_timeout_retains_admission_until_worker_finishes():
    class Slow(Store):
        def search(self, *args, **kwargs):
            time.sleep(.15)
            return []
    r = Retriever(Embedder(),Slow(),timeout=.02,max_jobs=1)
    try:
        assert (await r.retrieve("one",doc_types=("product",))).status == "timeout"
        assert (await r.retrieve("two",doc_types=("product",))).status == "busy"
        await asyncio.sleep(.2)
        assert (await r.retrieve("three",doc_types=("product",))).status == "timeout"
    finally:
        r.close()


@pytest.mark.asyncio
async def test_weak_results_and_failures_do_not_become_answers():
    class Weak(Store):
        def search(self,*args,**kwargs):
            return [Hit("one","one.md","One","weak",.1,"product")]
    r = Retriever(Embedder(),Weak())
    assert (await r.retrieve("unknown",doc_types=("product",))).status == "no_match"
    r.close()
