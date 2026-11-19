import asyncio
import hmac
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, cast
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lead_qualification_chatbot.api.locks import SessionLocks
from lead_qualification_chatbot.api.middleware import RateLimiter, resolve_request_id
from lead_qualification_chatbot.api.schemas import ChatRequest, ChatResponse, SessionResponse
from lead_qualification_chatbot.app import build_orchestrator
from lead_qualification_chatbot.config import Settings, settings
from lead_qualification_chatbot.graph.orchestrator import ChatOrchestrator
from lead_qualification_chatbot.logging_config import get_logger, setup_logging

log = get_logger(__name__)

_UNLIMITED = frozenset({"/health", "/ready"})


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def create_app(
    *, orchestrator: ChatOrchestrator | None = None, cfg: Settings = settings
) -> FastAPI:
    if cfg.app_env == "prod" and not cfg.api_key.get_secret_value():
        raise ValueError("API_KEY is required when APP_ENV=prod")

    locks = SessionLocks()
    limiter = RateLimiter(cfg.rate_limit_per_minute)
    prod = cfg.app_env == "prod"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        app.state.orchestrator = orchestrator or await build_orchestrator(cfg)
        log.info("api_ready", env=cfg.app_env)
        yield
        log.info("api_shutdown")

    app = FastAPI(
        title="Lead qualification chatbot",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if prod else "/docs",
        redoc_url=None,
        openapi_url=None if prod else "/openapi.json",
    )
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = resolve_request_id(request.headers.get("x-request-id"))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            retry_after = (
                None if request.url.path in _UNLIMITED else limiter.check(_client_key(request))
            )
            if retry_after is not None:
                response: Response = JSONResponse(
                    {"detail": "Too many requests. Please slow down."},
                    status_code=429,
                    headers={"Retry-After": str(math.ceil(retry_after))},
                )
            else:
                try:
                    response = await call_next(request)
                except Exception:
                    log.exception("unhandled_error")
                    response = JSONResponse(
                        {"detail": "Internal server error", "request_id": request_id},
                        status_code=500,
                    )
            response.headers["X-Request-ID"] = request_id
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            return response
        finally:
            structlog.contextvars.unbind_contextvars("request_id")

    async def auth(x_api_key: Annotated[str | None, Header()] = None) -> None:
        expected = cfg.api_key.get_secret_value()
        if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def bot(request: Request) -> ChatOrchestrator:
        return cast(ChatOrchestrator, request.app.state.orchestrator)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}  # liveness: process is up

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        ok = getattr(request.app.state, "orchestrator", None) is not None
        return JSONResponse(
            {"status": "ready" if ok else "starting"}, status_code=200 if ok else 503
        )

    @app.post("/sessions", response_model=SessionResponse, dependencies=[Depends(auth)])
    async def start_session(request: Request) -> SessionResponse:
        return SessionResponse(session_id=uuid4().hex, messages=[bot(request).greeting()])

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(auth)])
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
        session_id = body.session_id or uuid4().hex
        try:
            async with locks.hold(session_id):
                reply = await asyncio.wait_for(
                    bot(request).chat(session_id, body.message),
                    timeout=cfg.request_timeout_s,
                )
        except TimeoutError:
            log.error("turn_timeout", timeout_s=cfg.request_timeout_s)
            raise HTTPException(
                status_code=504, detail="The assistant took too long to respond. Please try again."
            ) from None
        return ChatResponse(
            session_id=session_id,
            messages=list(reply.messages),
            step=reply.step,
            qualified=reply.qualified,
            done=reply.done,
            closing=reply.closing,
            degraded=reply.degraded,
        )

    return app
