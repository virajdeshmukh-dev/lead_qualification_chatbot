import re
import time
from collections import deque
from collections.abc import Callable
from uuid import uuid4

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def resolve_request_id(header: str | None) -> str:
    """Accept a well-formed caller id, otherwise mint one (never trust arbitrary input)."""
    return header if header and _REQUEST_ID.match(header) else uuid4().hex


class RateLimiter:
    """Sliding-window limiter per key. In-memory: one instance per process."""

    def __init__(
        self,
        limit: int,
        window_s: float = 60.0,
        *,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_s
        self._max_keys = max_keys
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> float | None:
        """Return None if allowed, else seconds until the caller may retry."""
        if self._limit <= 0:
            return None
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return self._window - (now - hits[0])
        hits.append(now)
        if len(self._hits) > self._max_keys:
            self._prune(now)
        return None

    def _prune(self, now: float) -> None:
        stale = [k for k, q in self._hits.items() if not q or now - q[-1] >= self._window]
        for key in stale:
            del self._hits[key]
