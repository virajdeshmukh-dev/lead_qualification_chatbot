import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict
from uuid import uuid4

import structlog
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from lead_qualification_chatbot.domain import messages as msg
from lead_qualification_chatbot.domain.enums import Step
from lead_qualification_chatbot.domain.models import Lead
from lead_qualification_chatbot.flow.engine import FlowEngine
from lead_qualification_chatbot.flow.types import FlowState, Interpretation
from lead_qualification_chatbot.graph.shortcuts import deterministic_ok, override_route
from lead_qualification_chatbot.leads.sink import LeadSink
from lead_qualification_chatbot.llm.text import MAX_CHARS
from lead_qualification_chatbot.logging_config import get_logger

log = get_logger(__name__)

_INJECTION = re.compile(
    r"\bignore\s+(?:(?:all|any|your|the|previous|prior|above|these|those)\s+)+"
    r"(?:instructions?|rules?|prompts?)"
    r"|\bdisregard\b.{0,30}\b(?:instructions?|rules?)"
    r"|\bsystem\s+prompt\b|\bdeveloper\s+mode\b|\bjailbreak\b|\byou\s+are\s+now\b",
    re.IGNORECASE,
)
_NOOP = Interpretation().model_dump(mode="json")


class Interpreter(Protocol):
    async def interpret(self, text: str, *, step: Step, question: str) -> Interpretation: ...


class Answerer(Protocol):
    async def answer(self, question: str) -> str: ...
    async def try_answer(self, question: str) -> str | None: ...


class ChatState(TypedDict, total=False):
    # persisted across turns (checkpointed, plain JSON)
    flow: dict[str, Any]
    off_topic_strikes: int
    turns: int
    # per-turn
    user_text: str
    interp: dict[str, Any]
    route: str
    replies: list[str]
    session_id: str


@dataclass(frozen=True, slots=True)
class ChatReply:
    messages: tuple[str, ...]
    step: str
    route: str
    qualified: bool
    done: bool
    closing: str | None
    degraded: bool = False


def _dump(flow: FlowState) -> dict[str, Any]:
    return flow.model_dump(mode="json")


class ChatOrchestrator:
    def __init__(
        self,
        *,
        engine: FlowEngine,
        interpreter: Interpreter,
        faq_agent: Answerer,
        property_agent: Answerer,
        office_phone: str,
        max_turns: int = 60,
        max_off_topic: int = 3,
        checkpointer: Any = None,  # noqa: ANN401
        tracer: LangChainTracer | None = None,
        lead_sink: LeadSink | None = None,
    ) -> None:
        self._engine = engine
        self._interpreter = interpreter
        self._faq = faq_agent
        self._property = property_agent
        self._phone = office_phone
        self._max_turns = max_turns
        self._max_off_topic = max_off_topic
        self._tracer = tracer
        self._graph = self._build(checkpointer or MemorySaver())
        self._lead_sink = lead_sink

    # ---- graph wiring (node names must differ from state keys)
    def _build(self, checkpointer: Any) -> Any:  # noqa: ANN401
        g = StateGraph(ChatState)
        g.add_node("guard", self._guard)
        g.add_node("interpret", self._interpret)
        g.add_node("run_flow", self._run_flow)
        g.add_node("answer_faq", self._answer_faq)
        g.add_node("answer_property", self._answer_property)
        g.add_node("decline", self._decline)
        g.add_edge(START, "guard")
        g.add_conditional_edges(
            "guard",
            self._route_of,
            {"blocked": END, "interpret": "interpret", "flow": "run_flow", "off_topic": "decline"},
        )
        g.add_conditional_edges(
            "interpret",
            self._route_of,
            {
                "flow": "run_flow",
                "faq": "answer_faq",
                "property": "answer_property",
                "off_topic": "decline",
            },
        )
        for node in ("run_flow", "answer_faq", "answer_property", "decline"):
            g.add_edge(node, END)
        return g.compile(checkpointer=checkpointer)

    @staticmethod
    def _route_of(state: ChatState) -> str:
        return state["route"]

    # ---- public API
    def greeting(self) -> str:
        return msg.PROMPTS[Step.HOME]

    async def chat(self, session_id: str, text: str) -> ChatReply:
        structlog.contextvars.bind_contextvars(session_id=session_id)
        started = time.perf_counter()
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "run_name": "chat_turn",
            "metadata": {"session_id": session_id},
        }

        if self._tracer is not None:
            config["callbacks"] = [self._tracer]
        try:
            out = await self._graph.ainvoke({"user_text": text, "session_id": session_id}, config)
        except Exception:  # boundary: a visitor must never see a stack trace
            log.exception("turn_failed")
            reply = ChatReply(
                messages=(msg.technical_difficulty(self._phone),),
                step="",
                route="error",
                qualified=False,
                done=False,
                closing=None,
                degraded=True,
            )
        else:
            flow = FlowState.model_validate(out["flow"])
            reply = ChatReply(
                messages=tuple(out["replies"]),
                step=flow.step.value,
                route=out["route"],
                qualified=flow.qualified,
                done=flow.step is Step.DONE,
                closing=flow.closing.value if flow.closing else None,
            )
            log.info(
                "turn_done",
                route=reply.route,
                step=reply.step,
                qualified=reply.qualified,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        finally:
            structlog.contextvars.unbind_contextvars("session_id")
        return reply

    # ---- nodes
    async def _guard(self, state: ChatState) -> dict[str, Any]:
        flow = FlowState.model_validate(state["flow"]) if state.get("flow") else FlowState()
        turns = state.get("turns", 0) + 1
        text = state["user_text"].strip()[:MAX_CHARS]
        base: dict[str, Any] = {
            "turns": turns,
            "user_text": text,
            "interp": _NOOP,
            "replies": [],
            "flow": _dump(flow),
        }
        if flow.step is Step.DONE:
            return {**base, "route": "flow"}  # engine answers "this chat has ended"
        if turns > self._max_turns:
            flow.step = Step.DONE
            return {
                **base,
                "flow": _dump(flow),
                "route": "blocked",
                "replies": [msg.session_limit(self._phone)],
            }
        if not text:
            return {
                **base,
                "route": "blocked",
                "replies": [msg.EMPTY, self._engine.prompt(flow)],
            }
        if _INJECTION.search(text):
            log.warning("injection_pattern_blocked")
            return {**base, "route": "off_topic"}
        return {**base, "route": "interpret"}

    async def _interpret(self, state: ChatState) -> dict[str, Any]:
        flow = FlowState.model_validate(state["flow"])
        text = state["user_text"]
        if deterministic_ok(flow.step, text):
            return {"interp": _NOOP, "route": "flow"}

        interp = await self._interpreter.interpret(
            text, step=flow.step, question=self._engine.prompt(flow)
        )
        route = override_route(text, interp.route)
        if route is not interp.route:
            log.info("route_overridden", llm_route=interp.route.value, route=route.value)
            interp = interp.model_copy(update={"route": route})
        return {"interp": interp.model_dump(mode="json"), "route": interp.route.value}

    async def _run_flow(self, state: ChatState) -> dict[str, Any]:
        flow = FlowState.model_validate(state["flow"])
        interp = Interpretation.model_validate(state["interp"])
        result = self._engine.step(flow, state["user_text"], interp)
        new = result.state
        if new.qualified and new.step is Step.E3_ANYTHING_ELSE and not new.lead_submitted:
            new.lead_submitted = await self._submit_lead(new, state.get("session_id", ""))
        return {"flow": _dump(new), "replies": list(result.messages), "off_topic_strikes": 0}

    async def _submit_lead(self, flow: FlowState, session_id: str) -> bool:
        sl = flow.slots
        if self._lead_sink is None or sl.intent is None or not (sl.phone and sl.email):
            return False
        lead = Lead(
            lead_id=uuid4().hex,
            session_id=session_id,
            name=sl.name or "",
            intent=sl.intent,
            category=sl.category,
            budget=sl.budget,
            postcode=sl.postcode,
            property_id=sl.property_id,
            match_ids=flow.match_ids,
            phone=sl.phone,
            email=sl.email,
        )
        try:
            await self._lead_sink.submit(lead)
        except Exception:  
            log.exception("lead_handoff_failed", lead_id=lead.lead_id)
            return False
        log.info("lead_captured", lead_id=lead.lead_id, intent=lead.intent.value)
        return True

    def _resume(self, state: ChatState, answer: str) -> dict[str, Any]:
        flow = FlowState.model_validate(state["flow"])
        if flow.step is Step.HOME:  # greeting was answered; carry on with "name?"
            flow.step = Step.ASK_NAME
        return {
            "flow": _dump(flow),
            "replies": [answer, self._engine.prompt(flow)],
            "off_topic_strikes": 0,
        }

    async def _answer_faq(self, state: ChatState) -> dict[str, Any]:
        return self._resume(state, await self._faq.answer(state["user_text"]))

    async def _answer_property(self, state: ChatState) -> dict[str, Any]:
        text = state["user_text"]
        answer = await self._property.try_answer(text)
        if answer is None:  # misrouted (e.g. a branch question) or unknown: try the FAQ once
            log.info("property_to_faq_cascade")
            answer = await self._faq.answer(text)
        return self._resume(state, answer)

    async def _decline(self, state: ChatState) -> dict[str, Any]:
        flow = FlowState.model_validate(state["flow"])
        strikes = state.get("off_topic_strikes", 0) + 1
        if strikes >= self._max_off_topic:
            flow.step = Step.DONE
            replies = [msg.off_topic_close(self._phone)]
        else:
            if flow.step is Step.HOME:
                flow.step = Step.ASK_NAME
            replies = [msg.OFF_TOPIC, self._engine.prompt(flow)]
        return {"flow": _dump(flow), "replies": replies, "off_topic_strikes": strikes}
