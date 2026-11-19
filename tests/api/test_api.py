import asyncio

import pytest
from fastapi.testclient import TestClient

from lead_qualification_chatbot.api.app import create_app
from lead_qualification_chatbot.config import Settings
from tests.graph.test_orchestrator import OFFICE, make_bot


def client(bot=None, **kw):
    cfg = Settings(_env_file=None, llm_provider="ollama", llm_model="x", **kw)
    return TestClient(create_app(orchestrator=bot or make_bot()[0], cfg=cfg))


def test_health_and_ready():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/ready").json() == {"status": "ready"}


def test_start_session_returns_greeting():
    with client() as c:
        body = c.post("/sessions").json()
        assert len(body["session_id"]) == 32 and body["messages"] == ["How may I help you?"]


def test_chat_conversation_keeps_state_by_session_id():
    with client() as c:
        first = c.post("/chat", json={"message": "I want to sell"}).json()
        assert first["messages"] == ["Can I get your name?"] and not first["done"]
        second = c.post("/chat", json={"session_id": first["session_id"], "message": "Sam"}).json()
        assert second["step"] == "d1_postcode" and second["session_id"] == first["session_id"]


def test_validation_errors():
    with client() as c:
        assert c.post("/chat", json={"message": "x" * 2001}).status_code == 422
        assert c.post("/chat", json={"session_id": "bad id!", "message": "hi"}).status_code == 422
        assert c.post("/chat", json={}).status_code == 422


def test_request_id_is_echoed_or_replaced():
    with client() as c:
        ok = c.get("/health", headers={"X-Request-ID": "abc-12345"})
        assert ok.headers["X-Request-ID"] == "abc-12345"
        bad = c.get("/health", headers={"X-Request-ID": "bad id with spaces"})
        assert len(bad.headers["X-Request-ID"]) == 32


def test_rate_limit_blocks_chat_but_not_health():
    with client(rate_limit_per_minute=3) as c:
        codes = [c.post("/chat", json={"message": "hi"}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        blocked = c.post("/chat", json={"message": "hi"})
        assert int(blocked.headers["Retry-After"]) >= 1
        assert c.get("/health").status_code == 200


def test_api_key_required_when_configured():
    with client(api_key="secret") as c:
        body = {"message": "hi"}
        assert c.post("/chat", json=body).status_code == 401
        assert c.post("/chat", json=body, headers={"X-API-Key": "nope"}).status_code == 401
        assert c.post("/chat", json=body, headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_prod_requires_api_key():
    cfg = Settings(
        _env_file=None,
        app_env="prod",
        llm_provider="ollama",
        llm_model="x",
    )
    with pytest.raises(ValueError, match="API_KEY is required"):
        create_app(orchestrator=make_bot()[0], cfg=cfg)


def test_internal_failure_degrades_to_office_number_not_500():
    bot, *_ = make_bot(exc=RuntimeError("boom"))
    with client(bot) as c:
        r = c.post("/chat", json={"message": "hello there, I need some help"})
        assert r.status_code == 200
        body = r.json()
        assert body["degraded"] and OFFICE in body["messages"][0]


class SlowBot:
    def greeting(self):
        return "hi"

    async def chat(self, session_id, text):
        await asyncio.sleep(1)


def test_slow_turn_times_out_with_504():
    with client(SlowBot(), request_timeout_s=0.05) as c:
        assert c.post("/chat", json={"message": "x"}).status_code == 504
