import pytest

from lead_qualification_chatbot.exceptions import InvalidInputError
from lead_qualification_chatbot.validation.budget import parse_budget
from lead_qualification_chatbot.validation.contact import validate_email_address, validate_phone
from lead_qualification_chatbot.validation.postcode import extract_postcode, normalize_postcode


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("£1.5m", 1_500_000),
        ("1,200,000", 1_200_000),
        ("1.2 million", 1_200_000),
        ("950k", 950_000),
        ("my budget is £2m", 2_000_000),
        ("1M", 1_000_000),
    ],
)
def test_budget_ok(text, expected):
    assert parse_budget(text) == expected


@pytest.mark.parametrize("text", ["lots", "5", "", "0", "-", "£5000000000"])
def test_budget_bad(text):
    with pytest.raises(InvalidInputError):
        parse_budget(text)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gu5 7aq", "GU5 7AQ"),
        ("GU57AQ", "GU5 7AQ"),
        (" sw3  3rb ", "SW3 3RB"),
        ("W1K 6AB", "W1K 6AB"),
    ],
)
def test_postcode_ok(raw, expected):
    assert normalize_postcode(raw) == expected


@pytest.mark.parametrize("raw", ["", "12345", "GU5", "ZZZZ ZZZ", "GU5 7CI"])
def test_postcode_bad(raw):
    with pytest.raises(InvalidInputError):
        normalize_postcode(raw)


def test_extract_postcode():
    assert extract_postcode("I live near gu5 7aq, thanks") == "GU5 7AQ"
    assert extract_postcode("no code here") is None


def test_phone():
    assert validate_phone("07911 123456") == "+447911123456"
    assert validate_phone("+44 7911 123456") == "+447911123456"
    for bad in ("12345", "abc", ""):
        with pytest.raises(InvalidInputError):
            validate_phone(bad)


def test_email():
    assert validate_email_address(" jane.doe@gmail.com ") == "jane.doe@gmail.com"
    for bad in ("jane@", "no-at-sign", ""):
        with pytest.raises(InvalidInputError):
            validate_email_address(bad)
