import pytest

from lead_qualification_chatbot.domain.enums import Intent, ListingCategory
from lead_qualification_chatbot.flow import parsing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sam", "Sam"),
        ("my name is sam smith", "Sam Smith"),
        ("I'm Tom and I want to sell", "Tom"),
        ("I'm looking to buy", None),
        ("yes", None),
        ("hello", None),
    ],
)
def test_name_lenient(text, expected):
    assert parsing.parse_name(text, strict=False) == expected


def test_name_strict_ignores_bare_word():
    assert parsing.parse_name("Sam", strict=True) is None
    assert parsing.parse_name("I'm Sam", strict=True) == "Sam"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I want to buy a flat", Intent.BUY),
        ("thinking of selling my house", Intent.SELL),
        ("hello", None),
        ("buy or sell?", None),
    ],
)
def test_intent(text, expected):
    assert parsing.parse_intent(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("new build please", ListingCategory.NEW_BUILD),
        ("a re-sale", ListingCategory.RESALE),
        ("either", None),
        ("new-build or resale", None),
    ],
)
def test_category(text, expected):
    assert parsing.parse_category(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes please", True),
        ("No thanks", False),
        ("nothing else", False),
        ("maybe", None),
        ("yes and no", None),
    ],
)
def test_yes_no(text, expected):
    assert parsing.parse_yes_no(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2", 2),
        ("the first one", 1),
        ("option 3", 3),
        ("#4", 4),
        ("Apartment 3, Aldermoor Place", None),
    ],
)
def test_ordinal(text, expected):
    assert parsing.parse_ordinal(text) == expected
