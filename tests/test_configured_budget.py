import os
import subprocess
import sys


def test_configured_output_limit_matches_prompt_reservation():
    code = '''
from inspect import signature
from backend.config import MAX_TOKENS
from backend.conversation_manager.prompt_builder import build_messages,token_count
from tests.test_prompt_budget import Tokenizer
assert MAX_TOKENS==1024
assert signature(build_messages).parameters['response_tokens'].default==MAX_TOKENS
t=Tokenizer()
m=build_messages([],'question','PRODUCT_QUERY',evidence=[],summary=None,tokenizer=t)
assert token_count(t,m)+MAX_TOKENS+256<=4096
'''
    subprocess.run([sys.executable, '-c', code], env={**os.environ, 'SHOPBOT_MAX_TOKENS':'1024'}, check=True)
