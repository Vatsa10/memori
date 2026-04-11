from hashlib import md5
from typing import Optional

from memory_system.core.models import IntentPrediction


class IntentCache:
    def __init__(self, maxsize: int = 256):
        self._cache: dict[str, IntentPrediction] = {}
        self._order: list[str] = []
        self._maxsize = maxsize
        self.hits: int = 0
        self.misses: int = 0

    def _make_key(self, bot_id: str, message: str) -> str:
        normalized = message.lower().strip()
        return md5(f"{bot_id}:{normalized}".encode()).hexdigest()

    def get(self, bot_id: str, message: str) -> Optional[IntentPrediction]:
        key = self._make_key(bot_id, message)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, bot_id: str, message: str, prediction: IntentPrediction):
        key = self._make_key(bot_id, message)
        if len(self._order) >= self._maxsize and key not in self._cache:
            evict = self._order.pop(0)
            del self._cache[evict]
        self._cache[key] = prediction
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def clear(self):
        self._cache.clear()
        self._order.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)
