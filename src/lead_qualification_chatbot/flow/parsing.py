import re

from lead_qualification_chatbot.domain.enums import Intent, ListingCategory

_YES = re.compile(r"\b(yes|yeah|yep|yup|sure|ok|okay|absolutely|definitely|correct)\b", re.I)
_NO = re.compile(r"\b(no|nope|nah|nothing|none|that'?s all|that is all)\b", re.I)
_BUY = re.compile(r"\b(buy|buying|purchase|purchasing)\b", re.I)
_SELL = re.compile(r"\b(sell|selling|valuation|valuing)\b", re.I)
_NEW = re.compile(r"\bnew[\s-]?builds?\b|\bbrand[\s-]new\b|\boff[\s-]plan\b", re.I)
_RESALE = re.compile(r"\bre[\s-]?sale\b|\bsecond[\s-]?hand\b|\bexisting\b|\bpre[\s-]?owned\b", re.I)
_NAME_LEAD = re.compile(
    r"(?:my name is|my name's|name's|call me|this is|i am|i'm|it's)\s+"
    r"([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*)?)",
    re.I,
)
_WORD = re.compile(r"^[A-Za-z][A-Za-z'’-]*$")

_STOP = frozenset(
    [
        "and",
        "looking",
        "here",
        "interested",
        "trying",
        "wanting",
        "hoping",
        "after",
        "selling",
        "buying",
        "a",
        "an",
        "the",
        "just",
        "also",
        "not",
        "new",
        "to",
        "in",
        "from",
        "calling",
        "enquiring",
        "inquiring",
        "thinking",
        "planning",
        "keen",
        "happy",
        "sorry",
        "wondering",
        "ready",
        "at",
        "on",
        "for",
        "with",
        "but",
        "so",
    ]
)
_NOT_NAMES = frozenset(
    [
        "yes",
        "no",
        "hi",
        "hello",
        "hey",
        "buy",
        "sell",
        "thanks",
        "thank",
        "please",
        "help",
        "ok",
        "okay",
    ]
)
_ORD_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_ORD_NUM = re.compile(
    r"^(?:the\s+)?(?:option\s+|number\s+|no\.?\s*|#\s*)?(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:\s+one)?[.!]?$",
    re.I,
)
_ORD_WORD = re.compile(r"^(?:the\s+)?(first|second|third|fourth|fifth)(?:\s+one)?[.!]?$", re.I)


def parse_yes_no(text: str) -> bool | None:
    yes, no = bool(_YES.search(text)), bool(_NO.search(text))
    return None if yes == no else yes


def parse_intent(text: str) -> Intent | None:
    buy, sell = bool(_BUY.search(text)), bool(_SELL.search(text))
    if buy == sell:
        return None
    return Intent.BUY if buy else Intent.SELL


def parse_category(text: str) -> ListingCategory | None:
    new, resale = bool(_NEW.search(text)), bool(_RESALE.search(text))
    if new == resale:
        return None
    return ListingCategory.NEW_BUILD if new else ListingCategory.RESALE


def _clean_name(tokens: list[str]) -> str | None:
    if not tokens or tokens[0].lower() in _STOP | _NOT_NAMES:
        return None
    if len(tokens) > 1 and tokens[1].lower() in _STOP:
        tokens = tokens[:1]
    return " ".join(t if any(c.isupper() for c in t) else t.capitalize() for t in tokens)


def parse_name(text: str, *, strict: bool) -> str | None:
    """strict: only 'I'm X' style. lenient: a bare 1-3 word reply also counts."""
    lead = _NAME_LEAD.search(text)
    if lead:
        return _clean_name(lead.group(1).split())
    if strict:
        return None
    tokens = text.replace(",", " ").split()
    if 1 <= len(tokens) <= 3 and len(text) <= 40 and all(_WORD.match(t) for t in tokens):
        return _clean_name(tokens)
    return None


def parse_ordinal(text: str) -> int | None:
    text = text.strip()
    if m := _ORD_NUM.match(text):
        return int(m.group(1))
    if m := _ORD_WORD.match(text):
        return _ORD_WORDS[m.group(1).lower()]
    return None
