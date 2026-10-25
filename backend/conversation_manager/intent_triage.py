"""
intent_triage.py — Rule-based intent detection.

Classifies support questions, social messages, and unrelated requests.
This runs entirely in Python — no LLM call is made here.

Intent labels:
  PRODUCT_QUERY   — product questions, availability, specs, pricing, comparisons
  ORDER_TRACKING  — order status, shipment stage, delivery, tracking numbers
  RETURNS_POLICY  — returns, refunds, exchanges, shipping policy, costs
  OUT_OF_SCOPE    — anything outside the e-commerce support domain
  GREETING        — greetings, introductions, and name recall
"""

import re
from typing import Optional

from backend.data_lookup import get_all_categories, get_products_by_category

# ── Keyword dictionaries ──────────────────────────────────────────────────────

_ORDER_KEYWORDS = [
    r'\border\b', r'\btrack\b', r'\btracking\b', r'\bstatus\b',
    r'\bshipment\b', r'\bshipments\b', r'\bshipped\b', r'\bshipping status\b',
    r'\bdelivery\b', r'\bdeliver\b', r'\bdelivered\b', r'\bdispatch\b',
    r'\best.*delivery\b',
    r'\bpackage\b', r'\bparcel\b', r'\bcarrier\b', r'\bfedex\b',
    r'\bups\b', r'\busps\b', r'\btracking number\b', r'\bord-\d{5}\b',
    r'\bmy order\b', r'\border id\b', r'\border number\b',
    r'\bprocessing\b', r'\bout for delivery\b',
]

_PRODUCT_KEYWORDS = [
    r'\bproduct\b', r'\bproducts\b', r'\bitem\b', r'\bitems\b',
    r'\bprice\b', r'\bpricing\b', r'\bcost\b', r'\bhow much\b',
    r'\bavailable\b', r'\bavailability\b', r'\bin stock\b', r'\bstock\b',
    r'\bspec\b', r'\bspecs\b', r'\bspecifications?\b', r'\bfeature\b', r'\bfeatures\b',
    r'\bsize\b', r'\bsizes\b', r'\bvariant\b', r'\bvariants\b', r'\bcolor\b', r'\bcolors\b',
    r'\bcompare\b', r'\bcomparison\b', r'\bvs\b', r'\bversus\b',
    r'\bbuy\b', r'\bpurchase\b', r'\border\b',
    r'\bearbuds?\b', r'\bheadphones?\b', r'\bspeaker\b', r'\bcharger\b',
    r'\bwebcam\b', r'\btablet\b', r'\blaptop\b', r'\bphone\b',
    r'\bjacket\b', r'\bshirt\b', r'\bshorts\b', r'\bclothing\b', r'\bapparel\b',
    r'\byoga\b', r'\bmat\b', r'\bdumbbells?\b', r'\bweights?\b',
    r'\bpurifier\b', r'\blamp\b', r'\bdesk lamp\b', r'\bcoffee\b',
    r'\bsmart plug\b', r'\bwifi\b', r'\bbook\b', r'\bbooks\b',
    r'\bdo you (sell|have|carry|stock)\b',
    r'\bwhat.*sell\b', r'\bwhat.*carry\b',
]

_RETURNS_KEYWORDS = [
    r'\breturn\b', r'\breturns\b', r'\brefund\b', r'\brefunds\b',
    r'\bexchange\b', r'\bexchanges\b', r'\breplace\b', r'\breplacement\b',
    r'\bpolicy\b', r'\bpolicies\b', r'\bshipping cost\b', r'\bfree shipping\b',
    r'\bshipping fee\b', r'\bshipping time\b', r'\bshipping.*policy\b',
    r'\bhow long.*ship\b', r'\bhow.*return\b', r'\bcan i return\b',
    r'\breturn window\b', r'\b30 days?\b', r'\brefund process\b',
    r'\bget.*money back\b', r'\bmoney back\b', r'\bdamaged\b', r'\bdefective\b',
    r'\bwrong item\b', r'\bwrong size\b', r'\bdoesn.t fit\b',
    r'\bchange.*size\b', r'\bswap\b',
    r'\bhow much.*ship\b', r'\bovernight\b', r'\bexpedited\b',
]

_CLOSING_KEYWORDS = [
    r'\bthank you\b', r'\bthanks\b', r'\bthank\b', r'\bbye\b',
    r'\bgoodbye\b', r'\bsee you\b', r'\bthat.s all\b', r'\bthat was all\b',
    r'\bno more\b', r'\bnothing else\b', r'\ball good\b', r'\ball set\b',
    r'\bdone\b', r'\bgreat\b',
]


_SUPPORT_INTENTS = {"PRODUCT_QUERY", "ORDER_TRACKING", "RETURNS_POLICY"}

# These requests remain unrelated even when they mention products or delivery.
_UNRELATED_REQUESTS = [
    r'\bcapital\s+(?:city\s+)?of\b',
    r'\b(?:write|compose|tell|give|create)\b.*\b(?:poem|poetry|story|joke|code|essay)\b',
    r'\b(?:weather|politics|president|prime minister|horoscope)\b',
    r'\b\d+\s*[+*/=]\s*\d+\b',
    r'\b(?:ignore|forget|override)\b.*\b(?:instructions|rules|system prompt)\b',
]

_GREETING = r'(?:hi|hello|hey|hiya|howdy|greetings|salam|assalamualaikum|good (?:morning|afternoon|evening))(?: there)?'
_INTRODUCTION = r"(?:my name is|i am|i'm|i’m|call me) [^\W\d_]+(?:[-'][^\W\d_]+)*(?: [^\W\d_]+(?:[-'][^\W\d_]+)*){0,3}"
_SOCIAL_QUESTION = r"(?:how are you(?: doing)?|how is it going|what(?: is|'s) my name|do you remember my name|can you (?:tell me|remember) my name)"
_SOCIAL_ATOM = rf'(?:{_GREETING}|{_INTRODUCTION}|{_SOCIAL_QUESTION})'
_SOCIAL_PATTERNS = [
    rf'{_SOCIAL_ATOM}(?:[,!?. ]+{_SOCIAL_ATOM})*',
]

# Generic attributes (price, size, color) need an actual shopping reference.
_PRODUCT_REFERENCES = [
    r'\b[a-z]{3,5}-\d{3}\b',
    r'\b(?:products?|items?|catalog|electronics|apparel|home goods|sports equipment)\b',
    r'\b(?:earbuds?|headphones?|speakers?|chargers?|webcam|tablet|laptop|phone)\b',
    r'\b(?:jacket|shirt|shorts|clothing|yoga|mat|dumbbells?|weights?)\b',
    r'\b(?:purifier|lamp|coffee|smart plug|wifi|books?)\b',
    r'\bdo you (?:sell|have|carry|stock)\b',
    r'\bwhat.*(?:sell|carry)\b',
    r'\b(?:buy|purchase)\b',
]


def normalize_shopping_text(text: str) -> str:
    """Correct common catalog typos without rewriting arbitrary user content."""
    text = re.sub(r'\b(?:prodcut|prouduct|produt)(s)?\b',
                  lambda m: 'products' if m[1] else 'product', text.lower())
    text = re.sub(r'\b(?:availanle|availble|avaliable|availabe)\b', 'available', text)
    return re.sub(r'\blist (?:if|of) products?\b', 'list of products', text)


def is_catalog_request(text: str) -> bool:
    text = normalize_shopping_text(text)
    return bool(re.search(
        r'\b(?:catalog(?:ue)?|cheap(?:est)?|affordable|budget|inexpensive)\b'
        r'|\b(?:list|browse|all|selection|range)\b.*\b(?:products?|items?)\b'
        r'|\b(?:show|recommend|suggest)\b.*\b(?:products?|items?|something|options)\b'
        r'|\b(?:under|below|less than)\s*\$?\d+'
        r'|\b(?:what|which)\b.*\b(?:products|items)\b.*\b(?:available|in stock|have|sell|carry|offer)\b'
        r'|\bwhat (?:do you sell|can i buy|products do you have)\b', text))


def _shopping_request(text: str) -> bool:
    return bool(re.fullmatch(
        r'(?:i (?:need|want|am looking for)|recommend|suggest|show me) (?:me )?'
        r'(?:something|an? (?:option|item)) (?:cheap|affordable|inexpensive)'
        r'|(?:any|show me|recommend|suggest) (?:some )?(?:budget|affordable|cheap) options',
        text.rstrip('!?., ')))

# Full-message matching prevents an unrelated question from inheriting context.
_FOLLOWUPS = {
    "PRODUCT_QUERY": [
        r'how much(?: (?:is it|are they|does it cost|do they cost))?',
        r"what(?: is|'s) (?:its|their|the) (?:price|size|color|stock|availability)",
        r'what (?:size|color) (?:is it|are they)',
        r'(?:is it|are they) (?:available|in stock)',
        r"what(?: are|\'re) (?:its|their|the) (?:specs|features|sizes|colors|variants)",
        r'what (?:sizes|colors|variants) (?:are available|does it come in)',
        r'(?:what|how) about (?:it|that|this|them|those|the other one)',
        r'(?:does|do) (?:it|they) (?:come in|have) (?:black|white|blue|red|green|pink|purple|orange|yellow|grey|gray|navy|other colors|other sizes)',
        r'(?:black|white|blue|red|green|pink|purple|orange|yellow|grey|gray|navy|small|medium|large|xl)',
        r'(?:is|are) (?:it|they) (?:waterproof|wireless|compatible)',
        r'(?:tell me|show me) more(?: about (?:it|them|that))?',
    ],
    "ORDER_TRACKING": [
        r'(?:when|how soon) (?:will|does) (?:it|my package|my parcel) arrive',
        r'(?:where is|what about) (?:it|that|my package|my parcel)',
        r'(?:is|has) it (?:late|arrived|on the way)',
        r'\d{5}',
    ],
    "RETURNS_POLICY": [
        r'how long (?:does|will) (?:it|that|this) take',
        r'(?:what|how) about (?:it|that|this)',
        r'(?:what|which) (?:documents|photos|steps) (?:do i need|are needed)',
    ],
}


def _fullmatch_any(text: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(pattern, text, re.IGNORECASE) for pattern in patterns)


def is_support_followup(text: str, intent: str) -> bool:
    text = normalize_shopping_text(text)
    normalized = re.sub(r'\s+', ' ', text.strip()).rstrip('!?., ').lower()
    normalized = re.sub(rf'^(?:{_GREETING}|thanks|thank you)[,!. ]+', '', normalized)
    return _fullmatch_any(normalized, _FOLLOWUPS.get(intent, []))


def _mentions_product(text: str) -> bool:
    if any(re.search(pattern, text) for pattern in _PRODUCT_REFERENCES):
        return True
    # Recognize catalog brands/titles without treating arbitrary nouns as products.
    for category in get_all_categories():
        for product in get_products_by_category(category):
            words = product["name"].split()
            reference = " ".join(words[:2] if category.lower() == "books" else words[:1])
            if re.search(r'\b' + re.escape(reference.lower()) + r'\b', text):
                return True
    return False


def detect_intent(text: str, previous_intent: Optional[str] = None) -> str:
    """
    Detect the intent of a user message.

    Args:
        text: The raw user message.
        previous_intent: The active intent from the previous turn (for disambiguation).

    Returns:
        PRODUCT_QUERY, ORDER_TRACKING, RETURNS_POLICY, GREETING, OUT_OF_SCOPE, CLOSING
    """
    if not text or not text.strip():
        return "OUT_OF_SCOPE"

    text = normalize_shopping_text(text)
    normalized = re.sub(r'\s+', ' ', text.strip()).rstrip('!?., ').lower()
    if any(re.search(pattern, normalized) for pattern in _UNRELATED_REQUESTS):
        return "OUT_OF_SCOPE"

    # Social prefixes do not bypass routing: the entire message must be social.
    if _fullmatch_any(normalized, _SOCIAL_PATTERNS):
        return "GREETING"

    if _fullmatch_any(normalized, _CLOSING_KEYWORDS):
        return "CLOSING"

    if re.search(r'\b(?:warranty|international(?:ly)?|cancel(?:lation)?|shipping address)\b|\b(?:do you|can you|where do you) ship\b', normalized):
        return "RETURNS_POLICY"

    followup = re.sub(
        rf'^(?:{_GREETING}|thanks|thank you)[,!. ]+', '', normalized
    )
    if previous_intent in _SUPPORT_INTENTS and is_support_followup(text, previous_intent):
        return previous_intent

    # Score each category
    scores = {
        "ORDER_TRACKING": sum(1 for p in _ORDER_KEYWORDS if re.search(p, text, re.IGNORECASE)),
        "PRODUCT_QUERY":  sum(1 for p in _PRODUCT_KEYWORDS if re.search(p, text, re.IGNORECASE)),
        "RETURNS_POLICY": sum(1 for p in _RETURNS_KEYWORDS if re.search(p, text, re.IGNORECASE)),
    }
    if _mentions_product(normalized) or _shopping_request(normalized):
        scores["PRODUCT_QUERY"] = max(1, scores["PRODUCT_QUERY"])
    else:
        scores["PRODUCT_QUERY"] = 0

    # Require a minimum score to classify (avoids false positives on short texts)
    best_intent = max(scores, key=lambda k: scores[k])
    best_score  = scores[best_intent]

    if best_score == 0:
        # No e-commerce keywords detected — check if it could be a follow-up
        # (e.g., user just types an order ID number alone)
        if re.search(r'\bord[-\s]?\d{5}\b', text, re.IGNORECASE):
            return "ORDER_TRACKING"
        # Unknown messages are not assumed to be greetings or support follow-ups.
        return "OUT_OF_SCOPE"

    # ORDER_TRACKING and PRODUCT_QUERY can overlap; ORDER wins if there's an Order ID
    if re.search(r'\bord[-\s]?\d{5}\b', text, re.IGNORECASE):
        return "ORDER_TRACKING"

    return best_intent


def is_topic_switch(new_intent: str, previous_intent: Optional[str]) -> bool:
    """
    Return True if the user has switched to a different intent category.
    CLOSING and OUT_OF_SCOPE do not count as topic switches from a previous
    in-domain intent.
    """
    if previous_intent is None:
        return False
    if new_intent in ("OUT_OF_SCOPE", "CLOSING", "GREETING"):
        return False
    return new_intent != previous_intent


def get_policy_keywords_for_query(text: str) -> list[str]:
    """Extract topic keywords from a policy-related query for focused policy lookup."""
    keyword_map = {
        "return":   ["return", "how to start a return"],
        "refund":   ["refund", "refund timeline"],
        "exchange": ["exchange"],
        "damage":   ["damaged", "defective"],
        "cancel":   ["cancel"],
        "ship":     ["shipping", "delivery"],
        "cost":     ["shipping cost", "free shipping"],
        "overnight":["overnight"],
        "expedite": ["expedited"],
        "international": ["international"],
        "track":    ["tracking"],
    }
    found = []
    lower = text.lower()
    for trigger, kws in keyword_map.items():
        if trigger in lower:
            found.extend(kws)
    return list(set(found)) if found else ["return", "refund", "shipping"]
