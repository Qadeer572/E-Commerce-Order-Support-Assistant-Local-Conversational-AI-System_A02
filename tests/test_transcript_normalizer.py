import pytest
from backend.voice.transcript import normalize


@pytest.mark.parametrize("raw,expected_substr", [
    ("track order one zero zero zero two", "ORD-10002"),
    ("order number one zero zero zero two", "ORD-10002"),
    ("e l e c dash zero zero one", "ELEC-001"),
    ("forty nine dollars and ninety nine cents", "$49.99"),
    ("fifteen percent off", "15% off"),
    ("twenty five dollars", "$25"),
    ("two plus two", "2 + 2"),
    ("ten times five", "10 * 5"),
    ("twenty twenty six", "2026"),
])
def test_transcript_normalizer_table(raw, expected_substr):
    normalized = normalize(raw)
    assert expected_substr in normalized


def test_transcript_normalizer_quantities_remain_words():
    normalized = normalize("i want three of them please")
    assert "three of them" in normalized.lower()


def test_transcript_normalizer_question_punctuation():
    assert normalize("what is the price").endswith("?")
    assert normalize("how do i return this").endswith("?")
    assert not normalize("my name is sheharyar").endswith("?")


def test_transcript_normalizer_empty():
    assert normalize("") == ""
    assert normalize("   ") == ""
