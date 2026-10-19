"""Stage 1 — session stickiness (TTL LRU, in-process).

Switching models mid-session kills the prompt cache (cache is per model), so once
a session lands on a tier we reuse it and only ever ESCALATE, never downgrade.
In-proc is fine for a single demo task; swap in ElastiCache for multi-instance.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict


class StickyTiers:
    def __init__(self, maxsize: int = 4096, ttl_seconds: float = 1800.0):
        self._d: "OrderedDict[str, tuple[int, float]]" = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key: str) -> int | None:
        now = time.monotonic()
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            tier, ts = item
            if now - ts > self._ttl:
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return tier

    def escalate(self, key: str, tier: int) -> int:
        """Record a tier for a session, keeping the MAX seen. Returns stored tier."""
        now = time.monotonic()
        with self._lock:
            prev = self._d.get(key)
            new_tier = tier if prev is None else max(prev[0], tier)
            self._d[key] = (new_tier, now)
            self._d.move_to_end(key)
            while len(self._d) > self._maxsize:
                self._d.popitem(last=False)
            return new_tier
