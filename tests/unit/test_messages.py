from lead_qualification_chatbot.domain.enums import ClosingCode
from lead_qualification_chatbot.domain.messages import closing_message


def test_closing_has_phone_and_goodbye():
    for code in ClosingCode:
        msg = closing_message(code, "020 7946 0100")
        assert "020 7946 0100" in msg and msg.endswith("Goodbye.")
