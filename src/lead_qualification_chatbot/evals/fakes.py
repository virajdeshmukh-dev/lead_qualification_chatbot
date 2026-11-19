from lead_qualification_chatbot.data.repositories import DataStore
from lead_qualification_chatbot.domain.enums import Step
from lead_qualification_chatbot.flow.engine import FlowEngine
from lead_qualification_chatbot.flow.types import Interpretation
from lead_qualification_chatbot.graph.orchestrator import ChatOrchestrator
from lead_qualification_chatbot.leads.sink import LeadSink


class NoopInterpreter:
    async def interpret(self, text: str, *, step: Step, question: str) -> Interpretation:
        return Interpretation()


class StubAnswerer:
    async def answer(self, question: str) -> str:
        return "[stub answer]"

    async def try_answer(self, question: str) -> str | None:
        return "[stub answer]"


def build_offline_bot(
    data: DataStore, office_phone: str, lead_sink: LeadSink | None = None
) -> ChatOrchestrator:
    return ChatOrchestrator(
        engine=FlowEngine(data, office_phone),
        interpreter=NoopInterpreter(),
        faq_agent=StubAnswerer(),
        property_agent=StubAnswerer(),
        office_phone=office_phone,
        max_turns=1000,
        lead_sink=lead_sink,
    )
