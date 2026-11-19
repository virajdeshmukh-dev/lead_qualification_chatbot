from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from tenacity import wait_none

from lead_qualification_chatbot.agents.react import SideAgent, build_react_agent
from lead_qualification_chatbot.rag.faq_store import FaqHit
from lead_qualification_chatbot.tools.faq import make_faq_tool
from tests.factories import FAQ


class ToolFake(FakeMessagesListChatModel):
    """Scripted chat model that tolerates bind_tools."""

    def bind_tools(self, tools, **kwargs):
        return self


class StubStore:
    async def search(self, query):
        return [FaqHit(FAQ[0], 0.1)]


def tool_call(name, args):
    call = {"name": name, "args": args, "id": "call_1", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


def agent(model, **kw):
    graph = build_react_agent(model, [make_faq_tool(StubStore())], "system prompt")
    return SideAgent(graph, name="faq", fallback="FALLBACK", attempts=1, wait=wait_none(), **kw)


async def test_react_loop_calls_tool_then_answers():
    model = ToolFake(
        responses=[
            tool_call("search_faq", {"query": "hours"}),
            AIMessage(content="We open 9 to 5."),
        ]
    )
    assert await agent(model).answer("When do you open?") == "We open 9 to 5."


async def test_answer_without_tool_call_is_discarded():
    model = ToolFake(responses=[AIMessage(content="Made-up answer from memory")])
    assert await agent(model).answer("When do you open?") == "FALLBACK"


class Boom(Exception):
    status_code = 503


class ExplodingGraph:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, input, config=None, /):
        self.calls += 1
        raise Boom()


async def test_provider_failure_returns_fallback_after_retries():
    graph = ExplodingGraph()
    side = SideAgent(graph, name="faq", fallback="FALLBACK", attempts=3, wait=wait_none())
    assert await side.answer("x") == "FALLBACK" and graph.calls == 3


class EmptyStore:
    async def search(self, query):
        return []


async def test_no_match_returns_fixed_fallback_not_model_wording():
    model = ToolFake(
        responses=[tool_call("search_faq", {"query": "x"}), AIMessage(content="No idea!")]
    )  # noqa: E501
    graph = build_react_agent(model, [make_faq_tool(EmptyStore())], "system prompt")
    side = SideAgent(graph, name="faq", fallback="FALLBACK", attempts=1, wait=wait_none())
    assert await side.answer("Do you sell yachts?") == "FALLBACK"
