import logging
from typing import cast

import structlog
from structlog.types import Processor

from lead_qualification_chatbot.config import settings


def setup_logging() -> None:
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level.upper())
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
