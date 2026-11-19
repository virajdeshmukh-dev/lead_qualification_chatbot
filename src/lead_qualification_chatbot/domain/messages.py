from collections.abc import Sequence

from lead_qualification_chatbot.domain.enums import ClosingCode, ListingStatus, Step
from lead_qualification_chatbot.domain.models import Property

GOODBYE = "Thank you for chatting with us. Goodbye."
REASSIST = "How can I help you?"
RETRY = "Sorry, I didn't quite catch that."
EMPTY = "Sorry, I didn't receive a message."
OFF_TOPIC = "I can only help with buying or selling a home."

_CLOSING = {
    ClosingCode.M1: "Sorry, we don't cater to new-build properties under £1 million.",
    ClosingCode.M2: "Sorry, we don't cater to the postcode you provided.",
    ClosingCode.M3: "Sorry, we have no available homes matching your requirements right now.",
}

PROMPTS: dict[Step, str] = {
    Step.HOME: "How may I help you?",
    Step.ASK_NAME: "Can I get your name?",
    Step.BUY_OR_SELL: "Are you looking to buy or sell a property?",
    Step.SPECIFIC_PROPERTY: "Do you have a specific property in mind?",
    Step.NEWBUILD_OR_RESALE: "Are you looking for a new-build or a resale home?",
    Step.A1_BUDGET: "What is your budget?",
    Step.B1_BUDGET: "What is your budget?",
    Step.A3_POSTCODE: "What is the postcode of your location of interest?",
    Step.B2_POSTCODE: "What is the postcode of your location of interest?",
    Step.C1_PROPERTY: "Which property is it?",
    Step.C_NOT_FOUND: (
        "I couldn't find that property. Could you give me its name or address once more?"
    ),
    Step.C5_BUDGET: (
        "Sorry, that one's gone. I can help you find a similar home. What's your budget?"
    ),
    Step.D1_POSTCODE: "What is your postcode?",
    Step.E1_PHONE: "Can I get your phone number?",
    Step.E2_EMAIL: "Can I get your email address?",
    Step.E3_ANYTHING_ELSE: (
        "Someone will get in touch within 24 hours via phone or email. "
        "Is there anything else I can help you with?"
    ),
}


def money(value: int) -> str:
    return f"£{value:,}"


def closing_message(code: ClosingCode, office_phone: str) -> str:
    return (
        f"{_CLOSING[code]} Please call our office on {office_phone}. "
        "Thank you for chatting with us. Goodbye."
    )


def ended(office_phone: str) -> str:
    return f"This chat has ended. You can reach our office on {office_phone}."


def format_matches(name: str | None, props: Sequence[Property]) -> str:
    head = f"Here are some homes that could suit you, {name}:" if name else "Here are some homes:"
    lines = [
        f"{i}. {p.property_name}, {p.bedrooms} bed, {money(p.price_gbp)}"
        for i, p in enumerate(props, start=1)
    ]
    return "\n".join([head, *lines])


def property_details(p: Property) -> str:
    return (
        f"{p.property_name} is available: {p.property_type}, "
        f"{p.bedrooms} bedrooms, {money(p.price_gbp)}."
    )


def property_gone(p: Property) -> str:
    label = "under offer" if p.status is ListingStatus.UNDER_OFFER else "sold"
    return f"{p.property_name} is {label}."


def pick_prompt(props: Sequence[Property]) -> str:
    lines = [f"{i}. {p.property_name} ({p.address})" for i, p in enumerate(props, start=1)]
    return "\n".join(
        ["I found several matches:", *lines, "Which one do you mean? Reply with its number."]
    )


def no_info(office_phone: str) -> str:
    return (
        "I'm sorry, I don't have that information to hand. "
        f"Please call our office on {office_phone}."
    )


def off_topic_close(office_phone: str) -> str:
    return f"{OFF_TOPIC} Please call our office on {office_phone} \
                                if you need anything else. {GOODBYE}"


def session_limit(office_phone: str) -> str:
    return f"We've reached the limit for this chat. \
            Please call our office on {office_phone}. {GOODBYE}"


def technical_difficulty(office_phone: str) -> str:
    return f"Sorry, something went wrong on our side. Please call our office on {office_phone}."
