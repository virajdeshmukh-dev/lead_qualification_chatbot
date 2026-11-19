import re

from lead_qualification_chatbot.domain.enums import Route, Step
from lead_qualification_chatbot.exceptions import InvalidInputError
from lead_qualification_chatbot.flow.parsing import (
    parse_category,
    parse_intent,
    parse_name,
    parse_yes_no,
)
from lead_qualification_chatbot.validation.budget import parse_budget
from lead_qualification_chatbot.validation.contact import (
    find_email_candidate,
    find_phone_candidate,
    validate_email_address,
    validate_phone,
)
from lead_qualification_chatbot.validation.postcode import extract_postcode

_BUDGET = frozenset({Step.A1_BUDGET, Step.B1_BUDGET, Step.C5_BUDGET})
_POSTCODE = frozenset({Step.A3_POSTCODE, Step.B2_POSTCODE, Step.D1_POSTCODE})
_YES_NO = frozenset({Step.SPECIFIC_PROPERTY, Step.E3_ANYTHING_ELSE})
_MAX_YES_NO_CHARS = 15
_GREETING = re.compile(
    r"^(?:hi|hello|hey|hiya|good\s+(?:morning|afternoon|evening))[\s!.,]*$", re.IGNORECASE
)
_MAX_SHORT_CHARS = 40
_QUESTION_START = re.compile(
    r"^(?:what|where|when|which|who|how|why|do|does|did|can|could|would|is|are|will|tell me)\b",
    re.IGNORECASE,
)
_ASKS_ABOUT = re.compile(
    r"\b(?:address|phone|number|hours|opening|open|fees?|cost|price|branch|office|contact|"
    r"bedrooms?|available|commission|valuation)\b",
    re.IGNORECASE,
)
_PROPERTY_STEPS = frozenset({Step.C1_PROPERTY, Step.C_NOT_FOUND, Step.C_PICK})


def has_question_signal(text: str) -> bool:
    return "?" in text or bool(_QUESTION_START.match(text.strip()) or _ASKS_ABOUT.search(text))


def override_route(text: str, route: Route) -> Route:
    """Don't let a weak LLM hijack the flow: a side-question route needs a question signal,
    and 'off_topic' needs more than a word or two."""
    if route in (Route.FAQ, Route.PROPERTY) and not has_question_signal(text):
        return Route.FLOW
    if route is Route.OFF_TOPIC and len(text.split()) <= 3 and "?" not in text:
        return Route.FLOW
    return route


def deterministic_ok(step: Step, text: str) -> bool:
    """True when `text` cleanly answers this step's question and is not a side question.
    Then the LLM call is skipped: cheaper, faster, and no LLM risk on input steps.
    """
    if "?" in text:
        return False
    if step in _PROPERTY_STEPS:
        return _QUESTION_START.match(text.strip()) is None
    if step is Step.HOME:
        if _GREETING.match(text):
            return True
        return len(text) <= _MAX_SHORT_CHARS and parse_intent(text) is not None
    if step is Step.BUY_OR_SELL:
        return len(text) <= _MAX_SHORT_CHARS and parse_intent(text) is not None
    if step is Step.NEWBUILD_OR_RESALE:
        return len(text) <= _MAX_SHORT_CHARS and parse_category(text) is not None
    if step is Step.ASK_NAME:  # an "I'm Sam" pattern or one bare word, never a phrase
        single = len(text.split()) == 1 and parse_name(text, strict=False) is not None
        return single or parse_name(text, strict=True) is not None
    try:
        if step in _BUDGET:
            parse_budget(text)
            return True
        if step in _POSTCODE:
            return extract_postcode(text) is not None
        if step is Step.E1_PHONE:
            validate_phone(find_phone_candidate(text) or text)
            return True
        if step is Step.E2_EMAIL:
            validate_email_address(find_email_candidate(text) or text)
            return True
    except InvalidInputError:
        return False
    if step in _YES_NO:
        return len(text) <= _MAX_YES_NO_CHARS and parse_yes_no(text) is not None
    return False
