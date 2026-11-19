import re

from lead_qualification_chatbot.exceptions import InvalidInputError

_COMPACT = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\d[ABD-HJLNP-UW-Z]{2}$")
_IN_TEXT = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[ABD-HJLNP-UW-Z]{2})\b", re.IGNORECASE)


def normalize_postcode(raw: str) -> str:
    """'gu5 7aq' / 'GU57AQ' -> 'GU5 7AQ'. Raises InvalidInputError."""
    compact = re.sub(r"\s+", "", raw).upper()
    if not _COMPACT.match(compact):
        raise InvalidInputError(
            "postcode", "That doesn't look like a full UK postcode, e.g. GU5 7AQ."
        )
    return f"{compact[:-3]} {compact[-3:]}"


def extract_postcode(text: str) -> str | None:
    """Find a postcode inside a sentence; None if absent."""
    m = _IN_TEXT.search(text)
    return normalize_postcode("".join(m.groups())) if m else None
