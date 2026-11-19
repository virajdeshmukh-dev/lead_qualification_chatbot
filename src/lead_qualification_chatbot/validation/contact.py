import re

import phonenumbers
from email_validator import EmailNotValidError, validate_email

from lead_qualification_chatbot.exceptions import InvalidInputError

_PHONE_MSG = "That phone number doesn't look valid. Could you re-enter it?"
_EMAIL_MSG = "That email address doesn't look valid. Could you re-enter it?"
_PHONE = re.compile(r"\+?\d[\d\s().-]{5,}\d")
_EMAIL = re.compile(r"[^\s<>,;]+@[^\s<>,;]+")


def validate_phone(raw: str, region: str = "GB") -> str:
    """Return E.164, e.g. +447911123456."""
    try:
        parsed = phonenumbers.parse(raw.strip(), region)
    except phonenumbers.NumberParseException:
        parsed = None
    if parsed is None or not phonenumbers.is_valid_number(parsed):
        raise InvalidInputError("phone", _PHONE_MSG)
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def validate_email_address(raw: str) -> str:
    try:
        return validate_email(raw.strip(), check_deliverability=False).normalized
    except EmailNotValidError as exc:
        raise InvalidInputError("email", _EMAIL_MSG) from exc


def find_phone_candidate(text: str) -> str | None:
    match = _PHONE.search(text)
    return match.group(0) if match else None


def find_email_candidate(text: str) -> str | None:
    match = _EMAIL.search(text)
    return match.group(0) if match else None
