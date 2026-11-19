import hmac
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from lead_qualification_chatbot.app import build_orchestrator
from lead_qualification_chatbot.config import Settings, settings
from lead_qualification_chatbot.graph.orchestrator import ChatOrchestrator, ChatReply
from lead_qualification_chatbot.logging_config import setup_logging

OrchestratorBuilder = Callable[[Settings], Awaitable[ChatOrchestrator]]


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    messages: tuple[str, ...]
    step: str
    route: str
    qualified: bool
    done: bool
    closing: str | None
    degraded: bool

    @classmethod
    def from_reply(cls, reply: ChatReply) -> "ChatResponse":
        return cls(
            messages=reply.messages,
            step=reply.step,
            route=reply.route,
            qualified=reply.qualified,
            done=reply.done,
            closing=reply.closing,
            degraded=reply.degraded,
        )


def create_app(
    cfg: Settings = settings, builder: OrchestratorBuilder = build_orchestrator
) -> FastAPI:
    if cfg.app_env == "prod" and not cfg.api_key.get_secret_value():
        raise ValueError("API_KEY is required when APP_ENV=prod")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        app.state.orchestrator = await builder(cfg)
        yield

    app = FastAPI(
        title="Lead Qualification Chatbot",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        request: Request,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> ChatResponse:
        expected = cfg.api_key.get_secret_value()
        if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        bot = getattr(request.app.state, "orchestrator", None)
        if bot is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return ChatResponse.from_reply(await bot.chat(payload.session_id, payload.message))

    return app


app = create_app()
