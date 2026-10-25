"""Text preparation for speech: sentence segmentation of streamed tokens and spoken-form cleaning."""
import re

SOURCES_MARKER = re.compile(r"\n\s*Sources:")
FIRST_FLUSH_CHARS = 40   # first segment may flush at a clause break once it is this long
MAX_SEGMENT_CHARS = 220  # safety flush so a never-ending sentence cannot delay audio
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "vs", "etc", "e.g", "i.e", "no", "st", "approx"}
_DIGIT_WORDS = "zero one two three four five six seven eight nine".split()


class SentenceSegmenter:
    """Feed streamed LLM tokens; get back complete sentences ready for TTS."""

    def __init__(self, first_flush_chars: int = FIRST_FLUSH_CHARS, max_chars: int = MAX_SEGMENT_CHARS):
        self.first_flush_chars, self.max_chars = first_flush_chars, max_chars
        self._buffer = ""
        self._emitted = 0
        self._stopped = False

    def feed(self, text: str) -> list[str]:
        if self._stopped:
            return []
        self._buffer += text
        marker = SOURCES_MARKER.search(self._buffer)
        if marker:  # the server-built footer is shown in the chat but never spoken
            self._buffer = self._buffer[:marker.start()]
            self._stopped = True
            return self._drain(final=True)
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """Emit whatever is left at the end of the reply."""
        return self._drain(final=True)

    def _emit(self, out: list[str], text: str) -> None:
        text = text.strip()
        if text:
            out.append(text)
            self._emitted += 1

    def _drain(self, final: bool) -> list[str]:
        out: list[str] = []
        while True:
            cut = self._next_cut()
            if cut is None:
                break
            self._emit(out, self._buffer[:cut])
            self._buffer = self._buffer[cut:]
        if final:
            self._emit(out, self._buffer)
            self._buffer = ""
        return out

    def _next_cut(self):
        """Index where the next segment ends, or None if the buffer holds no complete segment."""
        buffer = self._buffer
        newline = buffer.find("\n")
        candidates = [newline + 1] if newline >= 0 else []
        for match in re.finditer(r"[.!?]+(?=\s)", buffer):
            word = re.findall(r"[\w.]+$", buffer[:match.start()])
            last = word[0].lower() if word else ""
            at_line_start = not buffer[:match.start() - len(last)].strip() if last else False
            if last in _ABBREVIATIONS or (last.isdigit() and at_line_start):
                continue  # "Mr. Smith" or a "1." list marker
            candidates.append(match.end())
            break
        if candidates:
            return min(candidates)
        if self._emitted == 0 and len(buffer) >= self.first_flush_chars:
            clauses = list(re.finditer(r"[,;:](?=\s)", buffer))
            if clauses:
                return clauses[-1].end()
        if len(buffer) >= self.max_chars:
            space = buffer.rfind(" ", 0, self.max_chars)
            return space + 1 if space > 0 else self.max_chars
        return None


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200d]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PATH = re.compile(r"(?<!\w)(?:[\w.-]+/)+[\w.-]+")
_PRICE = re.compile(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?")


def _spell_digits(digits: str) -> str:
    return " ".join(_DIGIT_WORDS[int(d)] for d in digits)


def _say_price(match: re.Match) -> str:
    dollars = int(match.group(1).replace(",", ""))
    text = f"{dollars} {'dollar' if dollars == 1 else 'dollars'}"
    if match.group(2):
        cents = int(match.group(2).ljust(2, "0"))
        if cents:
            text += f" and {cents} {'cent' if cents == 1 else 'cents'}"
    return text


def clean_for_speech(text: str) -> str:
    """Turn chat markdown into text that sounds natural when read aloud."""
    text = _URL.sub("", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = _PATH.sub("", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+\.)\s+", "", text)
    text = re.sub(r"[*_#>|]+", " ", text)
    text = _PRICE.sub(_say_price, text)
    text = re.sub(r"\bORD-(\d{5})\b", lambda m: "order " + _spell_digits(m.group(1)), text)
    text = re.sub(r"\b([A-Z]{3,5})-(\d{3})\b",
                  lambda m: " ".join(m.group(1)) + " " + _spell_digits(m.group(2)), text)
    text = text.replace("%", " percent").replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()
