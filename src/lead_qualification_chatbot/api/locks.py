import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from weakref import WeakValueDictionary


class SessionLocks:
    """One lock per session id. Locks vanish when unused (weak refs), so memory stays bounded.

    In-process only: multi-worker deployments need a distributed lock.
    """

    def __init__(self) -> None:
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        async with lock:
            yield
