"""
context_manager.py — Conversation memory management via summarization.

Strategy: Summarize older turns instead of dropping them.
- Recent turns are always kept verbatim (MIN_HISTORY_TURNS).
- When the estimated token budget is exceeded, older turns are summarized
  into a compact summary block using the LLM.
- The summary preserves key facts: order IDs, product names/SKUs, intent,
  and any open requests (returns, exchanges, cancellations).

The summary is stored on the Session and re-summarized only when new
material needs to be folded in (so it stays fresh).
"""

from typing import Optional

from backend.config import MAX_HISTORY_TOKENS, MIN_HISTORY_TURNS


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate: characters / 4."""
    return max(1, len(text) // 4)


def estimate_prompt_tokens(system_prompt: str, injected_facts: str, history: list) -> int:
    """Estimate total token count for a prompt assembly."""
    total = _estimate_tokens(system_prompt) + _estimate_tokens(injected_facts)
    total += sum(_estimate_tokens(t.content) for t in history)
    return total


# ── Summarization prompt ──────────────────────────────────────────────────────

SUMMARY_PROMPT = """\
Summarize the following conversation history into a single concise paragraph.
Preserve these facts exactly:
- Order IDs (format ORD-XXXXX)
- Product names and SKUs
- Any return, exchange, or cancellation requests
- The current topic/intent of the conversation
- Customer name if mentioned

Ignore filler and greetings. Output ONLY the summary, no preamble.
"""

SUMMARY_REBUILD_PROMPT = """\
You are maintaining a running summary of a customer support conversation.
Merge the NEW CONVERSATION EXCERPT into the EXISTING SUMMARY below, producing
a single updated concise paragraph. Preserve all facts (order IDs, product
names/SKUs, requests, intent). Output ONLY the updated summary.
"""


def needs_summarization(history: list, reserved_tokens: int) -> bool:
    """
    Return True when the current history (plus reserved tokens) exceeds the
    configured token budget and should be summarized.
    """
    budget = MAX_HISTORY_TOKENS - reserved_tokens
    if budget <= 0:
        return True
    history_tokens = sum(_estimate_tokens(t.content) for t in history)
    return history_tokens > budget


def split_for_summarization(history: list) -> tuple[list, list]:
    """
    Split history into (to_summarize, keep_verbatim).

    The last MIN_HISTORY_TURNS turns are always kept verbatim.
    Everything older is eligible for summarization.
    """
    if len(history) <= MIN_HISTORY_TURNS:
        return [], history[:]
    return history[:-MIN_HISTORY_TURNS], history[-MIN_HISTORY_TURNS:]


def build_summary_context(summary: Optional[str]) -> str:
    """Render the summary as a prompt-injectable context block."""
    if not summary:
        return ""
    return f"\n## CONVERSATION SUMMARY\n{summary}\n"


async def summarize_and_replace(
    llm,
    history: list,
    existing_summary: Optional[str],
) -> tuple[Optional[str], list]:
    """
    Summarize the older portion of history and return (updated_summary, kept_turns).

    Args:
        llm: LLMEngine instance with an async .stream(messages) method.
        history: Full ordered list of Turn objects.
        existing_summary: Previous summary, or None.

    Returns:
        (summary_text, trimmed_history) where trimmed_history contains only
        the most recent turns kept verbatim.
    """
    to_summarize, keep = split_for_summarization(history)
    if not to_summarize:
        return existing_summary, keep

    # Build the transcript to summarize
    transcript = "\n".join(
        f"{t.role}: {t.content}" for t in to_summarize
    )

    if existing_summary:
        prompt = (
            f"EXISTING SUMMARY:\n{existing_summary}\n\n"
            f"NEW CONVERSATION EXCERPT:\n{transcript}\n\n"
            f"{SUMMARY_REBUILD_PROMPT}"
        )
    else:
        prompt = f"{transcript}\n\n{SUMMARY_PROMPT}"

    # Stream the summary (small, fast)
    summary_text = ""
    async for token in llm.stream(
        [{"role": "system", "content": "You are a precise conversation summarizer."},
         {"role": "user", "content": prompt}]
    ):
        summary_text += token

    summary_text = summary_text.strip()
    if not summary_text:
        # Fall back to keeping everything if summarization produced nothing
        return existing_summary, history

    return summary_text, keep