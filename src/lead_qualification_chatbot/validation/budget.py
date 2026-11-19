import re
from decimal import Decimal

from lead_qualification_chatbot.exceptions import InvalidInputError

_PATTERN = re.compile(
    r"(?<![\w.])£?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>million|mn|m|thousand|k)?(?!\w)",
    re.IGNORECASE,
)
_MULT = {"million": 1_000_000, "mn": 1_000_000, "m": 1_000_000, "thousand": 1_000, "k": 1_000}
MIN_BUDGET = 1_000
MAX_BUDGET = 1_000_000_000
_HINT = "Please give your budget as an amount, e.g. £1.5m, 1,200,000 or 950k."


def parse_budget(text: str) -> int:
    """Parse a free-text budget into whole pounds."""
    match = _PATTERN.search(text)
    if not match:
        raise InvalidInputError("budget", _HINT)
    unit = (match["unit"] or "").lower()
    value = Decimal(match["num"].replace(",", ""))
    if not unit and value < MIN_BUDGET:
        raise InvalidInputError("budget", _HINT)  # ambiguous: "5" or "1.2"
    amount = int(value * _MULT.get(unit, 1))
    if not MIN_BUDGET <= amount <= MAX_BUDGET:
        raise InvalidInputError("budget", _HINT)
    return amount
