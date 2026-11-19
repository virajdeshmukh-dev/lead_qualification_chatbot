import re
from typing import Any

MAX_CHARS = 1000
_EMAIL = re.compile(r"[^\s<>,;@]+@[^\s<>,;]+")
_PHONE = re.compile(r"(?<!\w)\+?(?:\d[\s().-]?){9,14}\d(?!\w)")


def sanitize_visitor_text(text: str) -> str:
    """Bound length, mask contact details, strip angle brackets."""
    return redact_contact(text[:MAX_CHARS]).replace("<", " ").replace(">", " ")


def content_to_text(content: str | list[str | dict[str, Any]]) -> str:
    """Chat message content may be a string or a list of blocks (e.g. Gemini)."""
    if isinstance(content, str):
        return content.strip()
    parts = [p if isinstance(p, str) else str(p.get("text", "")) for p in content]
    return "".join(parts).strip()


def redact_contact(text: str) -> str:
    """Mask emails and phone numbers. Budgets and postcodes are left intact."""
    return _PHONE.sub("[phone]", _EMAIL.sub("[email]", text))
