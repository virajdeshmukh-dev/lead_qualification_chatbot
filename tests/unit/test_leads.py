import json
import stat

import pytest

from lead_qualification_chatbot.domain.enums import Intent
from lead_qualification_chatbot.domain.models import Lead
from lead_qualification_chatbot.leads.sink import JsonlLeadSink
from tests.graph.test_orchestrator import make_bot, say

PC = "GU5 7AQ"  # covered by the test store


class ListSink:
    def __init__(self):
        self.leads: list[Lead] = []

    async def submit(self, lead):
        self.leads.append(lead)


async def test_jsonl_sink_appends_private_file(tmp_path):
    sink = JsonlLeadSink(tmp_path / "out" / "leads.jsonl")
    lead = Lead(
        lead_id="1",
        session_id="s",
        name="Sam",
        intent=Intent.SELL,
        phone="+447911123456",
        email="sam@gmail.com",
    )
    await sink.submit(lead)
    await sink.submit(lead.model_copy(update={"lead_id": "2"}))
    path = tmp_path / "out" / "leads.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["lead_id"] for r in rows] == ["1", "2"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_lead_is_emitted_once_when_contact_is_complete():
    sink = ListSink()
    bot, *_ = make_bot(lead_sink=sink)
    await say(bot, "s", "I want to sell", "Sam", PC, "07911 123456")
    assert sink.leads == []  # no lead before the email
    await say(bot, "s", "sam@gmail.com")
    assert len(sink.leads) == 1
    lead = sink.leads[0]
    assert lead.phone == "+447911123456" and lead.email == "sam@gmail.com"
    assert lead.intent is Intent.SELL and lead.postcode == PC
    await say(bot, "s", "no")
    assert len(sink.leads) == 1


async def test_second_journey_creates_a_second_lead():
    sink = ListSink()
    bot, *_ = make_bot(lead_sink=sink)
    await say(bot, "s", "I want to sell", "Sam", PC, "07911 123456", "sam@gmail.com", "yes")
    await say(bot, "s", "sell", PC)
    assert len(sink.leads) == 2


async def test_no_lead_for_unqualified_visitor():
    sink = ListSink()
    bot, *_ = make_bot(lead_sink=sink)
    await say(bot, "s", "buy", "Sam", "no", "new build", "900k")
    assert sink.leads == []


def test_unwritable_path_fails_fast(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        JsonlLeadSink(blocker / "leads.jsonl").ensure_writable()
