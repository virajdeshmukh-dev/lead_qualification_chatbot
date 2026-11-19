import pytest

from lead_qualification_chatbot.domain.enums import Route, Step
from lead_qualification_chatbot.graph.shortcuts import deterministic_ok, override_route


@pytest.mark.parametrize(
    ("step", "text", "expected"),
    [
        (Step.A1_BUDGET, "£2m", True),
        (Step.A1_BUDGET, "what's the budget range?", False),
        (Step.A1_BUDGET, "lots", False),
        (Step.D1_POSTCODE, "gu5 7aq", True),
        (Step.D1_POSTCODE, "where is GU5 7AQ?", False),
        (Step.E1_PHONE, "07911 123456", True),
        (Step.E1_PHONE, "12345", False),
        (Step.E2_EMAIL, "sam@gmail.com", True),
        (Step.E2_EMAIL, "nope", False),
        (Step.E3_ANYTHING_ELSE, "no", True),
        (Step.E3_ANYTHING_ELSE, "no, but what are your fees?", False),
        (Step.SPECIFIC_PROPERTY, "yes", True),
        (Step.ASK_NAME, "Sam", True),
        (Step.ASK_NAME, "I'm Sam", True),
        (Step.ASK_NAME, "opening hours please", False),
        (Step.HOME, "hi", True),
        (Step.HOME, "Hello!", True),
        (Step.HOME, "I want to sell", True),
        (Step.HOME, "I'd like to know about fees", False),
        (Step.BUY_OR_SELL, "buy", True),
        (Step.BUY_OR_SELL, "sell", True),
        (Step.BUY_OR_SELL, "buy or sell", False),
        (Step.NEWBUILD_OR_RESALE, "resale", True),
        (Step.NEWBUILD_OR_RESALE, "new-build or resale", False),
        (Step.C1_PROPERTY, "Apartment 1, Aldermoor Place", True),
        (Step.C1_PROPERTY, "Willow Court", True),
        (Step.C1_PROPERTY, "What is its price", False),
        (Step.C_PICK, "2", True),
    ],
)
def test_deterministic_ok(step, text, expected):
    assert deterministic_ok(step, text) is expected


@pytest.mark.parametrize(
    ("text", "route", "expected"),
    [
        ("buy", Route.FAQ, Route.FLOW),
        ("maybe", Route.PROPERTY, Route.FLOW),
        ("What are your hours", Route.FAQ, Route.FAQ),
        ("Hampstead branch address", Route.FAQ, Route.FAQ),
        ("Is it still available?", Route.PROPERTY, Route.PROPERTY),
        ("hi", Route.OFF_TOPIC, Route.FLOW),
        ("tell me a joke", Route.OFF_TOPIC, Route.OFF_TOPIC),
    ],
)
def test_override_route(text, route, expected):
    assert override_route(text, route) is expected
