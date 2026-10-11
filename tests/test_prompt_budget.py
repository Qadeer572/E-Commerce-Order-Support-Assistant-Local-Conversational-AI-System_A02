import pytest
from backend.conversation_manager.prompt_builder import build_messages, PromptBudgetError
from backend.conversation_manager.session import Turn
from backend.retrieval.types import Hit


class Tokenizer:
    def encode(self,text,**kwargs):
        return list(text)

    def decode(self,tokens,**kwargs):
        return "".join(tokens)

    def apply_chat_template(self,messages,**kwargs):
        return list("".join(m['content']+'x'*10 for m in messages))


def test_full_prompt_budget_includes_summary_current_input_and_output():
    tokenizer=Tokenizer()
    messages=build_messages([Turn('user','old '*500),Turn('assistant','reply '*500)],'earbuds?', 'PRODUCT_QUERY',
        evidence=[Hit('one','one.md','One','facts '*1000,.9,'product')],summary='Customer Qadeer',tokenizer=tokenizer)
    assert len(tokenizer.apply_chat_template(messages))+512+256<=4096
    assert 'Customer Qadeer' in messages[0]['content']
    assert messages[-1]['content']=='earbuds?'


def test_mandatory_input_overflow_is_bounded():
    with pytest.raises(PromptBudgetError):
        build_messages([], '漢'*5000,'GREETING', evidence=[],summary=None,tokenizer=Tokenizer())
