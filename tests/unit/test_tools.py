from lead_qualification_chatbot.data.repositories import PropertyRepository
from lead_qualification_chatbot.exceptions import LLMProviderError
from lead_qualification_chatbot.rag.faq_store import FaqHit
from lead_qualification_chatbot.tools.common import NO_MATCH, TOOL_ERROR
from lead_qualification_chatbot.tools.faq import make_faq_tool
from lead_qualification_chatbot.tools.properties import make_property_tool
from tests.factories import FAQ, make_property


class StubStore:
    def __init__(self, hits=None, exc=None):
        self.hits, self.exc = hits or [], exc

    async def search(self, query):
        if self.exc:
            raise self.exc
        return self.hits


async def test_property_tool_found():
    tool = make_property_tool(PropertyRepository([make_property()]))
    out = await tool.ainvoke({"query": "Apartment 1 Aldermoor Place"})
    assert "£1,500,000" in out and "Status: available" in out and "bedrooms: 2" in out


async def test_property_tool_no_match():
    tool = make_property_tool(PropertyRepository([make_property()]))
    assert await tool.ainvoke({"query": "Castle Nowhere"}) == NO_MATCH


async def test_faq_tool_formats_hits():
    tool = make_faq_tool(StubStore([FaqHit(FAQ[0], 0.1)]))
    out = await tool.ainvoke({"query": "hours"})
    assert "Q: What are your opening hours?" in out and "9am to 5pm" in out


async def test_faq_tool_no_match_and_error():
    assert await make_faq_tool(StubStore()).ainvoke({"query": "x"}) == NO_MATCH
    err = StubStore(exc=LLMProviderError("down"))
    assert await make_faq_tool(err).ainvoke({"query": "x"}) == TOOL_ERROR
