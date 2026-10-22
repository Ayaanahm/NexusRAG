"""Redis-backed search cache and per-client rate limit.

Both degrade to no-ops when Redis is unset or unreachable: a cache outage
should make the service slower, not take it down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

log = logging.getLogger(__name__)


class Cache:
    def __init__(self, url: str | None, ttl: int = 3600, rate_limit_per_minute: int = 30) -> None:
        self.ttl = ttl
        self.rate_limit = rate_limit_per_minute
        self._r = None
        if url:
            import redis.asyncio as redis

            self._r = redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)

    @property
    def enabled(self) -> bool:
        return self._r is not None

    @staticmethod
    def key(*parts) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return "nexus:" + hashlib.sha256(raw.encode()).hexdigest()[:32]

    async def get_json(self, key: str):
        if not self._r:
            return None
        try:
            v = await self._r.get(key)
            return json.loads(v) if v else None
        except Exception as e:  # noqa: BLE001 - cache errors must never fail a request
            log.warning("cache get failed: %s", e)
            return None

    async def set_json(self, key: str, value) -> None:
        if not self._r:
            return
        try:
            await self._r.set(key, json.dumps(value), ex=self.ttl)
        except Exception as e:  # noqa: BLE001
            log.warning("cache set failed: %s", e)

    async def allow(self, client_id: str, cost: int = 1) -> bool:
        """Fixed one-minute window per client."""
        if not self._r or self.rate_limit <= 0:
            return True
        window = int(time.time() // 60)
        key = f"nexus:rl:{client_id}:{window}"
        try:
            n = await self._r.incrby(key, cost)
            if n == cost:
                await self._r.expire(key, 61)
            return n <= self.rate_limit
        except Exception as e:  # noqa: BLE001
            log.warning("rate limit check failed, allowing: %s", e)
            return True

    async def ping(self) -> bool:
        if not self._r:
            return False
        try:
            return bool(await self._r.ping())
        except Exception:  # noqa: BLE001
            return False
