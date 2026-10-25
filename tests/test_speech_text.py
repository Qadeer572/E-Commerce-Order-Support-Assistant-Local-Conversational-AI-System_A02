import pytest
from backend.voice.speech_text import SentenceSegmenter, clean_for_speech


def test_segmenter_basic_sentences():
    seg = SentenceSegmenter()
    tokens = ["Hello world. ", "How are you? ", "I am doing great!"]
    out = []
    for t in tokens:
        out.extend(seg.feed(t))
    out.extend(seg.flush())
    assert out == ["Hello world.", "How are you?", "I am doing great!"]


def test_segmenter_does_not_split_on_decimals():
    seg = SentenceSegmenter()
    out = seg.feed("The total price is $49.99 for two units.")
    out.extend(seg.flush())
    assert out == ["The total price is $49.99 for two units."]


def test_segmenter_does_not_split_on_abbreviations():
    seg = SentenceSegmenter()
    out = seg.feed("Please see Dr. Smith vs. Mr. Jones for order details.")
    out.extend(seg.flush())
    assert out == ["Please see Dr. Smith vs. Mr. Jones for order details."]


def test_segmenter_first_clause_flush():
    seg = SentenceSegmenter(first_flush_chars=30)
    # Once length exceeds first_flush_chars, comma or colon can flush the first clause
    out = seg.feed("Welcome to ShopBot customer support, we are happy to help you with your order.")
    assert len(out) >= 1
    assert out[0] == "Welcome to ShopBot customer support,"


def test_segmenter_max_length_flush():
    seg = SentenceSegmenter(max_chars=60)
    out = seg.feed("This is a very long sentence that keeps going without any punctuation whatsoever to test the safety flush behavior.")
    assert len(out) >= 1
    assert len(out[0]) <= 60


def test_segmenter_sources_never_spoken():
    seg = SentenceSegmenter()
    out = []
    out.extend(seg.feed("Your package is scheduled to arrive tomorrow."))
    out.extend(seg.feed("\n\nSources: `policy/shipping.md`, `orders/10002.md`"))
    out.extend(seg.flush())
    # The Sources footer should be completely dropped from spoken segments
    assert any("arrive tomorrow" in s for s in out)
    assert not any("Sources" in s for s in out)
    assert not any("policy" in s for s in out)


def test_clean_for_speech():
    text = "Your order **ORD-10002** with `ELEC-001` is $49.99 & shipping is 15% off! Visit https://shop.example.com."
    cleaned = clean_for_speech(text)
    assert "ORD-10002" not in cleaned
    assert "order one zero zero zero two" in cleaned
    assert "ELEC-001" not in cleaned
    assert "E L E C zero zero one" in cleaned
    assert "$49.99" not in cleaned
    assert "49 dollars and 99 cents" in cleaned
    assert "15 percent off" in cleaned
    assert "https://" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned


def test_clean_for_speech_markdown_bullets():
    text = "- Item one\n- Item two\n- Item three"
    cleaned = clean_for_speech(text)
    assert "-" not in cleaned
    assert "Item one Item two Item three" == cleaned
