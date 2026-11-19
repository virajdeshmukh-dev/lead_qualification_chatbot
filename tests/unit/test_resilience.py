import pytest
from tenacity import wait_none

from lead_qualification_chatbot.exceptions import LLMProviderError
from lead_qualification_chatbot.llm.resilience import call_with_retry, is_transient


class Boom(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("boom")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (Boom(429), True),
        (Boom(503), True),
        (Boom(401), False),
        (Boom(400), False),
        (TimeoutError(), True),
        (ConnectionError(), True),
        (ValueError(), False),
    ],
)
def test_is_transient(exc, expected):
    assert is_transient(exc) is expected


async def test_retries_then_succeeds():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise Boom(429)
        return "ok"

    assert await call_with_retry(fn, attempts=3, op="t", wait=wait_none()) == "ok"
    assert calls == 3


async def test_gives_up_after_attempts():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise Boom(503)

    with pytest.raises(LLMProviderError):
        await call_with_retry(fn, attempts=3, op="t", wait=wait_none())
    assert calls == 3


async def test_non_transient_fails_fast():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise Boom(401)

    with pytest.raises(LLMProviderError):
        await call_with_retry(fn, attempts=3, op="t", wait=wait_none())
    assert calls == 1


async def test_accepts_lambda_returning_coroutine():
    calls = 0

    async def work():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise Boom(503)
        return "ok"

    assert await call_with_retry(lambda: work(), attempts=3, op="t", wait=wait_none()) == "ok"
    assert calls == 2
