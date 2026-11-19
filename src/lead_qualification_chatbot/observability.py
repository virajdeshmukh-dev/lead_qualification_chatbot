import os
from typing import Any

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client

from lead_qualification_chatbot.config import Settings
from lead_qualification_chatbot.llm.text import redact_contact
from lead_qualification_chatbot.logging_config import get_logger

log = get_logger(__name__)

_SENSITIVE_KEYS = frozenset({"phone", "email"})


def scrub(value: Any) -> Any:  # noqa: ANN401
    """Recursively mask contact details before anything leaves the process."""
    if isinstance(value, str):
        return redact_contact(value)
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if k in _SENSITIVE_KEYS and v else scrub(v) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [scrub(v) for v in value]
    return value


def build_tracer(cfg: Settings) -> LangChainTracer | None:
    """Explicit LangSmith tracer with PII scrubbing; disabled once if auth fails."""
    if not cfg.langsmith_tracing:
        return None
    key = cfg.langsmith_api_key.get_secret_value()
    if not key:
        log.warning("tracing_disabled", reason="LANGSMITH_API_KEY missing")
        return None
    if cfg.langsmith_workspace_id:
        os.environ.setdefault("LANGSMITH_WORKSPACE_ID", cfg.langsmith_workspace_id)
    client = Client(
        api_url=cfg.langsmith_endpoint, api_key=key, hide_inputs=scrub, hide_outputs=scrub
    )
    try:
        next(iter(client.list_projects(limit=1)), None)  # fail fast, not on every turn
    except Exception as exc:
        log.error("tracing_disabled", reason="auth check failed", error=type(exc).__name__)
        return None
    log.info("tracing_enabled", project=cfg.langsmith_project)
    return LangChainTracer(client=client, project_name=cfg.langsmith_project)
