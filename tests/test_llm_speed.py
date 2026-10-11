import pytest
from backend.llm_engine import LLMEngine


@pytest.mark.asyncio
async def test_model_stays_loaded_and_short_replies_use_smaller_output_limit():
    engine = LLMEngine()
    calls = []
    class Client:
        async def chat(self, **kwargs):
            calls.append(kwargs)
            async def chunks():
                yield {'message': {'content': 'hello'}}
            return chunks()
    engine.client = Client()
    assert [t async for t in engine.stream([{'role': 'user', 'content': 'hi'}])] == ['hello']
    assert calls[0]['keep_alive'] == '30m'
    assert calls[0]['options']['num_predict'] <= 256
    assert calls[0]['options']['num_gpu'] == 0
