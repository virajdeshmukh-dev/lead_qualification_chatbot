from enum import StrEnum


class ListingCategory(StrEnum):
    NEW_BUILD = "new_build"
    RESALE = "resale"


class ListingStatus(StrEnum):
    AVAILABLE = "available"
    UNDER_OFFER = "under_offer"
    SOLD = "sold"


class ClosingCode(StrEnum):
    M1 = "M1"  # new-build < £1M
    M2 = "M2"  # postcode not catered for
    M3 = "M3"  # no available homes


class Step(StrEnum):
    """Steps that WAIT for visitor input. Decision diamonds (A2, A4, B3...)
    are pure functions, not steps."""

    ASK_NAME = "ask_name"
    BUY_OR_SELL = "buy_or_sell"
    SPECIFIC_PROPERTY = "specific_property"
    NEWBUILD_OR_RESALE = "newbuild_or_resale"
    A1_BUDGET = "a1_budget"
    A3_POSTCODE = "a3_postcode"
    B1_BUDGET = "b1_budget"
    B2_POSTCODE = "b2_postcode"
    C1_PROPERTY = "c1_property"
    C_NOT_FOUND = "c_not_found"
    C5_BUDGET = "c5_budget"
    D1_POSTCODE = "d1_postcode"
    E1_PHONE = "e1_phone"
    E2_EMAIL = "e2_email"
    E3_ANYTHING_ELSE = "e3_anything_else"
    DONE = "done"
    HOME = "home"
    C_PICK = "c_pick"


class Intent(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Route(StrEnum):
    FLOW = "flow"
    FAQ = "faq"
    PROPERTY = "property"
    OFF_TOPIC = "off_topic"
