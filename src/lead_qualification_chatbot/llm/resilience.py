from collections.abc import Awaitable, Callable

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.wait import wait_base

from lead_qualification_chatbot.exceptions import LLMProviderError
from lead_qualification_chatbot.logging_config import get_logger

log = get_logger(__name__)

_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TRANSIENT_NAMES = frozenset(
    {"APIConnectionError", "APITimeoutError", "ReadTimeout", "ConnectTimeout", "ResourceExhausted"}
)


def is_transient(exc: BaseException) -> bool:
    """Provider-agnostic: timeouts, connection drops, 429 and 5xx are retried."""
    if isinstance(exc, TimeoutError | ConnectionError | httpx.TransportError):
        return True
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value in _TRANSIENT_STATUS:
            return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in _TRANSIENT_STATUS


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "llm_retry",
        attempt=state.attempt_number,
        error=type(exc).__name__ if exc else None,
    )


async def call_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    op: str,
    wait: wait_base | None = None,
) -> T:
    """Run `fn`, retrying transient failures with jittered backoff.

    Any final failure becomes LLMProviderError (message has no secrets).
    """
    retrying = AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait or wait_retry_after(wait_random_exponential(multiplier=0.5, max=8)),
        retry=retry_if_exception(is_transient),
        before_sleep=_log_retry,
        reraise=True,
    )

    async def attempt() -> T:
        return await fn()

    try:
        return await retrying(attempt)
    except Exception as exc:
        raise LLMProviderError(f"{op} failed: {type(exc).__name__}") from exc


class wait_retry_after(wait_base):
    """Honour the provider's Retry-After header (429/503); else use the fallback backoff."""

    def __init__(self, fallback: wait_base, cap_s: float = 20.0) -> None:
        self._fallback = fallback
        self._cap = cap_s

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        headers = getattr(getattr(exc, "response", None), "headers", None)
        raw = headers.get("retry-after") if headers is not None else None
        if raw is not None:
            try:
                return min(float(raw), self._cap)
            except ValueError:  # HTTP-date form: use the fallback
                pass
        return self._fallback(retry_state)
