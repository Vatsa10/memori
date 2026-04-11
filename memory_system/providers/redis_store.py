"""Redis-backed session store and cache."""

import json
from typing import Any, Optional

from memory_system.core.models import ConversationTurn


class RedisSessionStore:
    """Persistent session store backed by Redis."""

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "memory_system:session:",
        ttl: int = 86400,  # 24 hours
    ):
        self._url = url
        self._prefix = prefix
        self._ttl = ttl
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis
            self._client = redis.from_url(self._url, decode_responses=True)
        return self._client

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        client = self._get_client()
        key = f"{self._prefix}{session_id}"
        data = client.get(key)
        if not data:
            return []
        turns = json.loads(data)
        return [ConversationTurn(**t) for t in turns]

    def add_turn(self, session_id: str, turn: ConversationTurn):
        client = self._get_client()
        key = f"{self._prefix}{session_id}"
        history = self.get_history(session_id)
        history.append(turn)
        client.setex(key, self._ttl, json.dumps([t.model_dump() for t in history]))

    def clear(self, session_id: str):
        client = self._get_client()
        client.delete(f"{self._prefix}{session_id}")

    def list_sessions(self) -> list[str]:
        client = self._get_client()
        keys = client.keys(f"{self._prefix}*")
        return [k.replace(self._prefix, "") for k in keys]


class RedisCacheStore:
    """Redis-backed shared cache for intent predictions."""

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "memory_system:cache:",
        default_ttl: int = 3600,
    ):
        self._url = url
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis
            self._client = redis.from_url(self._url, decode_responses=True)
        return self._client

    def get(self, key: str) -> Optional[Any]:
        client = self._get_client()
        data = client.get(f"{self._prefix}{key}")
        if data:
            return json.loads(data)
        return None

    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        client = self._get_client()
        client.setex(
            f"{self._prefix}{key}",
            ttl or self._default_ttl,
            json.dumps(value),
        )

    def delete(self, key: str):
        client = self._get_client()
        client.delete(f"{self._prefix}{key}")
