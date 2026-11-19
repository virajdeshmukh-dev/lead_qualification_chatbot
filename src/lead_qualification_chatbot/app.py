from lead_qualification_chatbot.agents.faq_agent import build_faq_agent
from lead_qualification_chatbot.agents.interpreter import LLMInterpreter
from lead_qualification_chatbot.agents.property_agent import build_property_agent
from lead_qualification_chatbot.config import Settings, settings
from lead_qualification_chatbot.data.repositories import load_data
from lead_qualification_chatbot.exceptions import ConfigError
from lead_qualification_chatbot.flow.engine import FlowEngine
from lead_qualification_chatbot.flow.types import Interpretation
from lead_qualification_chatbot.graph.orchestrator import ChatOrchestrator
from lead_qualification_chatbot.leads.sink import JsonlLeadSink, LeadSink
from lead_qualification_chatbot.llm.embeddings import build_embeddings, embedding_id
from lead_qualification_chatbot.llm.factory import build_chat_model
from lead_qualification_chatbot.observability import build_tracer
from lead_qualification_chatbot.rag.faq_store import FaqStore


def _jsonl_sink(cfg: Settings) -> JsonlLeadSink:
    sink = JsonlLeadSink(cfg.leads_file)
    try:
        sink.ensure_writable()
    except OSError as exc:
        raise ConfigError(f"LEADS_FILE is not writable: {cfg.leads_file} ({exc.strerror})") from exc
    return sink


async def build_orchestrator(
    cfg: Settings = settings, *, lead_sink: LeadSink | None = None
) -> ChatOrchestrator:
    """Wire everything once at startup. Any config or data problem fails here, loudly."""
    data = load_data(cfg.data_dir)
    model = build_chat_model(cfg)
    faq_store = await FaqStore.create(
        data.faq,
        build_embeddings(cfg),
        embedding_id=embedding_id(cfg),
        persist_dir=cfg.chroma_dir,
        top_k=cfg.faq_top_k,
        max_distance=cfg.faq_max_distance,
        attempts=cfg.llm_max_attempts,
    )
    phone, attempts = cfg.office_phone, cfg.llm_max_attempts
    return ChatOrchestrator(
        engine=FlowEngine(data, phone),
        interpreter=LLMInterpreter(model.with_structured_output(Interpretation), attempts=attempts),
        faq_agent=build_faq_agent(model, faq_store, office_phone=phone, attempts=attempts),
        property_agent=build_property_agent(
            model, data.properties, office_phone=phone, attempts=attempts
        ),
        office_phone=phone,
        max_turns=cfg.max_turns,
        max_off_topic=cfg.max_off_topic,
        tracer=build_tracer(cfg),
        lead_sink=lead_sink or _jsonl_sink(cfg),
    )
