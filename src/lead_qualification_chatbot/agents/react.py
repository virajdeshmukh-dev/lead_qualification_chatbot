import importlib
from typing import Any, Protocol, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from tenacity.wait import wait_base

from lead_qualification_chatbot.exceptions import LLMProviderError
from lead_qualification_chatbot.llm.resilience import call_with_retry
from lead_qualification_chatbot.llm.text import content_to_text, sanitize_visitor_text
from lead_qualification_chatbot.logging_config import get_logger
from lead_qualification_chatbot.tools.common import NO_DATA_PREFIXES

log = get_logger(__name__)

RECURSION_LIMIT = 8  # ~3 tool rounds, then the graph stops itself


class Graph(Protocol):
    async def ainvoke(self, input: Any, config: Any = None, /) -> Any: ...  # noqa: ANN401


def build_react_agent(model: BaseChatModel, tools: list[BaseTool], system_prompt: str) -> Graph:
    """A ReAct loop (reason -> tool call -> observe -> answer), across library versions."""
    agents = importlib.import_module("langchain.agents")
    if hasattr(agents, "create_agent"):
        return cast(
            Graph, agents.create_agent(model=model, tools=tools, system_prompt=system_prompt)
        )
    prebuilt = importlib.import_module("langgraph.prebuilt")
    return cast(Graph, prebuilt.create_react_agent(model, tools, prompt=system_prompt))


class SideAgent:
    """Answers one side question. Never raises and never answers without a tool result."""

    def __init__(
        self,
        graph: Graph,
        *,
        name: str,
        fallback: str,
        attempts: int = 3,
        wait: wait_base | None = None,
    ) -> None:
        self._graph = graph
        self._name = name
        self._fallback = fallback
        self._attempts = attempts
        self._wait = wait

    async def try_answer(self, question: str) -> str | None:
        """Grounded answer, or None if the agent failed or answered without a tool result."""
        payload = {"messages": [HumanMessage(content=sanitize_visitor_text(question))]}
        try:
            result = await call_with_retry(
                lambda: self._graph.ainvoke(payload, {"recursion_limit": RECURSION_LIMIT}),
                attempts=self._attempts,
                op=f"{self._name}_agent",
                wait=self._wait,
            )
        except LLMProviderError as exc:
            log.warning("side_agent_degraded", agent=self._name, error=str(exc))
            return None

        messages = result.get("messages", []) if isinstance(result, dict) else []
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_messages:  # answered from model memory: not allowed
            log.warning("side_agent_ungrounded", agent=self._name)
            return None
        if all(content_to_text(m.content).startswith(NO_DATA_PREFIXES) for m in tool_messages):
            log.info("side_agent_no_data", agent=self._name)
            return None  # caller uses the fixed "call our office" fallback
        final_message = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)),
            None,
        )
        if final_message is None:
            log.warning("side_agent_missing_answer", agent=self._name)
            return None
        answer = content_to_text(final_message.content)
        return answer or None

    async def answer(self, question: str) -> str:
        return await self.try_answer(question) or self._fallback
