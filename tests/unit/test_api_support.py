import asyncio

from lead_qualification_chatbot.api.locks import SessionLocks
from lead_qualification_chatbot.api.middleware import RateLimiter, resolve_request_id


def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(2, 60, clock=lambda: now[0])
    assert limiter.check("a") is None and limiter.check("a") is None
    retry = limiter.check("a")
    assert retry is not None and 0 < retry <= 60
    assert limiter.check("b") is None  # other clients unaffected
    now[0] = 61
    assert limiter.check("a") is None  # window slid


def test_rate_limiter_disabled_with_zero():
    limiter = RateLimiter(0)
    assert all(limiter.check("a") is None for _ in range(100))


def test_request_id_validation():
    assert resolve_request_id("abc-12345") == "abc-12345"
    for bad in (None, "", "short", "has spaces in it!", "x" * 65):
        out = resolve_request_id(bad)
        assert out != bad and len(out) == 32


async def test_same_session_serialised_different_sessions_parallel():
    locks = SessionLocks()
    active = peak = 0

    async def work(key):
        nonlocal active, peak
        async with locks.hold(key):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(work("a"), work("a"), work("a"))
    assert peak == 1
    peak = 0
    await asyncio.gather(work("a"), work("b"))
    assert peak == 2
