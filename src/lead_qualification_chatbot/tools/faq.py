from typing import Protocol

from langchain_core.tools import BaseTool, tool

from lead_qualification_chatbot.exceptions import LLMProviderError
from lead_qualification_chatbot.rag.faq_store import FaqHit
from lead_qualification_chatbot.tools.common import NO_MATCH, TOOL_ERROR


class FaqSearcher(Protocol):
    async def search(self, query: str) -> list[FaqHit]: ...


def make_faq_tool(store: FaqSearcher) -> BaseTool:
    @tool
    async def search_faq(query: str) -> str:
        """Search the agency FAQ: company background, head office, branch addresses,
        phone numbers, opening hours, the buying and selling process, fees and policies.
        Input: the visitor's question in plain words.
        - Call the tool once. Search again only if the first result was clearly about something \
            else, and never more than twice."""
        try:
            hits = await store.search(query[:300])
        except LLMProviderError:
            return TOOL_ERROR
        if not hits:
            return NO_MATCH
        return "\n\n".join(f"Q: {h.entry.question}\nA: {h.entry.answer}" for h in hits)

    return search_faq
