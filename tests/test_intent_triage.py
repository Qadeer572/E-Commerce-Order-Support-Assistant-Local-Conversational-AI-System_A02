import pytest

from backend.conversation_manager.intent_triage import detect_intent


@pytest.mark.parametrize("previous", [None, "PRODUCT_QUERY", "ORDER_TRACKING", "RETURNS_POLICY"])
@pytest.mark.parametrize("message", [
    "What is the capital of Pakistan?",
    "ok what is capital of pakistan ?",
    "hi, what is the capital of Pakistan?",
    "thanks, what is the capital of Pakistan?",
    "tell me about the capital of Pakistan",
    "write me a poem about delivery drivers",
    "what is 2 + 2?",
    "What is the size of Pakistan?",
    "Hi, how much does a trip to Paris cost?",
    "What is the price of Bitcoin?",
])
def test_unrelated_questions_do_not_inherit_support_intent(message, previous):
    assert detect_intent(message, previous) == "OUT_OF_SCOPE"


@pytest.mark.parametrize("message", [
    "hi", "Hello!", "hey there", "good morning", "how are you?",
    "hi, how are you?", "my name is Qadeer Raza", "I'm Qadeer Raza",
    "hello, my name is Qadeer Raza", "what is my name?",
    "Can you tell me my name?", "Hi, I am Qadeer. How are you?",
])
def test_social_messages_have_their_own_intent(message):
    assert detect_intent(message, "ORDER_TRACKING") == "GREETING"


@pytest.mark.parametrize("message, previous, expected", [
    ("hello, track ORD-10002", None, "ORDER_TRACKING"),
    ("thanks, what is your return policy?", None, "RETURNS_POLICY"),
    ("do you sell earbuds?", None, "PRODUCT_QUERY"),
    ("tell me about ProSound", None, "PRODUCT_QUERY"),
    ("what is the price of Atomic Habits?", None, "PRODUCT_QUERY"),
    ("how much is it?", "PRODUCT_QUERY", "PRODUCT_QUERY"),
    ("what is its price?", "PRODUCT_QUERY", "PRODUCT_QUERY"),
    ("what size is it?", "PRODUCT_QUERY", "PRODUCT_QUERY"),
    ("What are its specs?", "PRODUCT_QUERY", "PRODUCT_QUERY"),
    ("What sizes are available?", "PRODUCT_QUERY", "PRODUCT_QUERY"),
    ("Does it come in pink?", "PRODUCT_QUERY", "PRODUCT_QUERY"),
    ("Hi, how much is it?", "PRODUCT_QUERY", "PRODUCT_QUERY"),
    ("Hi, how much is it? Also what is the capital of Pakistan?", "PRODUCT_QUERY", "OUT_OF_SCOPE"),
    ("does it come in black?", "PRODUCT_QUERY", "PRODUCT_QUERY"),
    ("when will it arrive?", "ORDER_TRACKING", "ORDER_TRACKING"),
    ("how long does it take?", "RETURNS_POLICY", "RETURNS_POLICY"),
    ("10002", "ORDER_TRACKING", "ORDER_TRACKING"),
    ("bye", "PRODUCT_QUERY", "CLOSING"),
    ("thanks", "PRODUCT_QUERY", "CLOSING"),
    ("capital?", "PRODUCT_QUERY", "OUT_OF_SCOPE"),
    ("how are you? what is the capital of Pakistan?", None, "OUT_OF_SCOPE"),
])
def test_support_followups_and_social_prefixes(message, previous, expected):
    assert detect_intent(message, previous) == expected
