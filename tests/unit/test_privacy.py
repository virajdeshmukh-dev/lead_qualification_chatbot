from lead_qualification_chatbot.llm.text import redact_contact, sanitize_visitor_text
from lead_qualification_chatbot.observability import scrub


def test_redact_contact_masks_only_contact_details():
    assert "07911" not in redact_contact("my number is 07911 123456")
    assert "gmail" not in redact_contact("mail a.b@gmail.co.uk please")
    for safe in ("£1,200,000", "budget 2000000", "GU5 7AQ", "SW1A 1AA"):
        assert redact_contact(safe) == safe


def test_sanitize_redacts_before_llm():
    assert "07911" not in sanitize_visitor_text("call 07911 123456 <b>")


def test_scrub_is_recursive_and_keeps_numbers():
    data = {
        "user_text": "call 07911 123456 or sam@gmail.com",
        "flow": {"slots": {"phone": "+447911123456", "email": "sam@gmail.com", "budget": 2000000}},
        "replies": ["Can I get your phone number?"],
    }
    out = scrub(data)
    flat = str(out)
    assert "07911" not in flat and "gmail" not in flat and "447911" not in flat
    assert out["flow"]["slots"]["budget"] == 2000000
