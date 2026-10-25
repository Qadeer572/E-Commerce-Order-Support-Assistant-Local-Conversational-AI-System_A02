"""Rule-based inverse text normalization for Vosk output.

Vosk returns spoken-form lowercase words ("track order one zero zero zero two"). Order ids,
SKUs, prices and calculator input need digits, so we convert the cases we can do safely and
leave anything ambiguous as words rather than guess.
"""
import re
from dataclasses import dataclass
from typing import Optional

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
TEENS = {w: 10 + i for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 20 + 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
OPERATORS = {"plus": "+", "minus": "-", "times": "*", "multiplied_by": "*", "divided_by": "/", "over": "/"}
QUESTION_WORDS = {"what", "how", "where", "when", "who", "why", "which", "is", "are", "do", "does",
                  "can", "could", "will", "would"}
PHRASES = {("multiplied", "by"): "multiplied_by", ("divided", "by"): "divided_by",
           ("p", "m"): "pm", ("a", "m"): "am"}


@dataclass
class Number:
    digits: str
    words: list[str]
    kind: str  # "digits" (digit run), "single" (one digit word), "compound", "decimal"


def _under_100(tokens: list[str], i: int) -> Optional[tuple[int, int]]:
    word = tokens[i] if i < len(tokens) else ""
    if word in TEENS:
        return TEENS[word], i + 1
    if word in TENS:
        if i + 1 < len(tokens) and UNITS.get(tokens[i + 1], 0) > 0:
            return TENS[word] + UNITS[tokens[i + 1]], i + 2
        return TENS[word], i + 1
    if word in UNITS:
        return UNITS[word], i + 1
    return None


def _parse_compound(tokens: list[str], i: int) -> Optional[tuple[int, int]]:
    """Parse e.g. 'twenty five', 'one hundred', 'two thousand five hundred'."""
    piece = _under_100(tokens, i)
    if piece is None:
        return None
    total, current, j = 0, piece[0], piece[1]
    while j < len(tokens):
        word = tokens[j]
        if word == "hundred" and 0 < current < 100:
            current, j = current * 100, j + 1
        elif word == "thousand" and current > 0 and total == 0:
            total, current, j = current * 1000, 0, j + 1
        elif current % 100 == 0 and (piece := _under_100(tokens, j)) and piece[0] > 0:
            current, j = current + piece[0], piece[1]
        else:
            break
    return total + current, j


def _join_phrases(tokens: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(tokens):
        pair = tuple(tokens[i:i + 2])
        if pair in PHRASES:
            out.append(PHRASES[pair])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def _spelled_skus(tokens: list[str]) -> list[str]:
    """'e l e c dash zero zero one' -> 'ELEC-001'."""
    out, i = [], 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
            j += 1
        k = j + 1
        while k < len(tokens) and tokens[k] in UNITS:
            k += 1
        if 3 <= j - i <= 5 and j < len(tokens) and tokens[j] in ("dash", "hyphen") and k - (j + 1) >= 3:
            out.append("".join(tokens[i:j]).upper() + "-" + "".join(str(UNITS[t]) for t in tokens[j + 1:k]))
            i = k
        else:
            out.append(tokens[i])
            i += 1
    return out


def _find_numbers(tokens: list[str]) -> list:
    items: list = []
    i = 0
    while i < len(tokens):
        run = i
        while run < len(tokens) and tokens[run] in UNITS:
            run += 1
        if run - i >= 2 and not (run < len(tokens) and tokens[run] in ("hundred", "thousand")):
            number, j = Number("".join(str(UNITS[t]) for t in tokens[i:run]), tokens[i:run], "digits"), run
        else:
            parsed = _parse_compound(tokens, i)
            if parsed is None:
                items.append(tokens[i])
                i += 1
                continue
            value, j = parsed
            kind = "single" if j - i == 1 and tokens[i] in UNITS else "compound"
            number = Number(str(value), tokens[i:j], kind)
        if j < len(tokens) and tokens[j] == "point":
            end = j + 1
            while end < len(tokens) and tokens[end] in UNITS:
                end += 1
            if end > j + 1:
                fraction = "".join(str(UNITS[t]) for t in tokens[j + 1:end])
                number = Number(number.digits + "." + fraction, tokens[i:end], "decimal")
                j = end
        items.append(number)
        i = j
    return items


def _resolve_adjacent(items: list) -> list:
    """Adjacent numbers are ambiguous: keep years, otherwise fall back to the original words."""
    out, i = [], 0
    while i < len(items):
        if not isinstance(items[i], Number):
            out.append(items[i])
            i += 1
            continue
        j = i
        while j < len(items) and isinstance(items[j], Number):
            j += 1
        group = items[i:j]
        if len(group) == 1:
            out.append(group[0])
        elif len(group) == 2 and group[0].digits in ("19", "20") and len(group[1].digits) <= 2 \
                and group[1].kind != "decimal":
            out.append(Number(group[0].digits + group[1].digits.zfill(2), group[0].words + group[1].words, "compound"))
        else:
            out.extend(word for number in group for word in number.words)
        i = j
    return out


def _is_number(item) -> bool:
    return isinstance(item, Number)


def _render(items: list) -> list[str]:
    """Apply the contextual rules (percent, dollars, am/pm, operators, lone digits)."""
    out: list[str] = []
    i = 0
    while i < len(items):
        item = items[i]
        nxt = items[i + 1] if i + 1 < len(items) else None
        prev_word = items[i - 1] if i > 0 and isinstance(items[i - 1], str) else None
        if not _is_number(item):
            neighbours = i > 0 and _is_number(items[i - 1]) and _is_number(nxt)
            out.append(OPERATORS[item] if item in OPERATORS and neighbours else item.replace("_", " "))
            i += 1
            continue
        if nxt == "percent":
            out.append(item.digits + "%")
            i += 2
        elif nxt in ("dollars", "dollar", "bucks"):
            skip = 1 if items[i + 2:i + 3] == ["and"] else 0
            cents = items[i + 2 + skip:i + 4 + skip]
            if len(cents) == 2 and _is_number(cents[0]) and cents[1] in ("cents", "cent") \
                    and len(cents[0].digits) <= 2:
                out.append(f"${item.digits}.{cents[0].digits.zfill(2)}")
                i += 4 + skip
            else:
                out.append("$" + item.digits)
                i += 2
        elif nxt in ("pm", "am"):
            out.append(f"{item.digits} {nxt}")
            i += 2
        elif item.kind == "single" and not (prev_word in OPERATORS or prev_word == "order"
                                            or (isinstance(nxt, str) and nxt in OPERATORS)):
            out.extend(item.words)  # "three of them" stays words
            i += 1
        else:
            out.append(item.digits)
            i += 1
    return out


def normalize(raw: str) -> str:
    """Convert raw ASR text to display/LLM text. Returns '' for empty input."""
    tokens = _spelled_skus(_join_phrases(raw.lower().split()))
    text = " ".join(_render(_resolve_adjacent(_find_numbers(tokens))))
    text = re.sub(r"\border (?:number )?(\d{5})\b", r"ORD-\1", text)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text.split()[0].lower() in QUESTION_WORDS and not text.endswith(("?", ".", "!")):
        text += "?"
    return text
