"""Tiny in-process LRU cache with a TTL. No external dependencies.

Only ever touched from the single asyncio event-loop thread, so no locking.
"""
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, maxsize: int = 512, ttl: float = 3600.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            del self._items[key]
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (time.monotonic() + self.ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()
        self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    def stats(self) -> dict:
        return {"size": len(self._items), "hits": self.hits, "misses": self.misses}
