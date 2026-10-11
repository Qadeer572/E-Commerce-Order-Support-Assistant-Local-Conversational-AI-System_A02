from scripts.benchmark_rag import summarize


def test_percentiles_and_cache_groups_are_separate():
    result = summarize([{'elapsed_ms':10,'cache_hit':False},{'elapsed_ms':30,'cache_hit':False},{'elapsed_ms':1,'cache_hit':True}])
    assert result['uncached']['p50_ms']==20
    assert result['uncached']['p95_ms']==29
    assert result['cached']['p50_ms']==1
