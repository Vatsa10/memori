"""BM25 retriever with per-user lazy indexes."""

import re
from typing import Any, Optional

from memory_system.core.memory_models import Memory, MemorySearchResult


_TOKEN_RE = re.compile(r"[A-Za-z0-9-]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Retriever:
    """BM25Okapi-backed retriever with a per-user document cache.

    The cache is invalidated by calling ``bump(user_id)`` whenever the
    underlying memory_store is mutated. The MemorySystem wires this to
    HookManager memory-mutation events.
    """

    def __init__(
        self,
        memory_store: Any,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        corpus_limit: int = 10_000,
    ):
        self.memory_store = memory_store
        self.k1 = k1
        self.b = b
        self.corpus_limit = corpus_limit
        # user_id -> (version_seen_at_build, index, list[Memory])
        self._indexes: dict[str, tuple[int, Any, list[Memory]]] = {}
        # user_id -> current monotonic version
        self._versions: dict[str, int] = {}

    def bump(self, user_id: str) -> None:
        self._versions[user_id] = self._versions.get(user_id, 0) + 1

    def _current_version(self, user_id: str) -> int:
        return self._versions.get(user_id, 0)

    async def _build_for(self, user_id: str) -> tuple[Any, list[Memory]]:
        from rank_bm25 import BM25Okapi

        results = await self.memory_store.get_all(user_id, k=self.corpus_limit)
        docs = [r.memory for r in results]
        tokenized = [_tokenize(m.text) for m in docs]
        if not tokenized:
            return None, []
        index = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        return index, docs

    async def search(
        self,
        query: str,
        user_id: str,
        k: int = 20,
    ) -> list[MemorySearchResult]:
        version = self._current_version(user_id)
        cached = self._indexes.get(user_id)
        if cached is None or cached[0] != version:
            index, docs = await self._build_for(user_id)
            self._indexes[user_id] = (version, index, docs)
        else:
            _, index, docs = cached

        if index is None or not docs:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = index.get_scores(tokens)
        max_score = max(scores) if len(scores) else 0.0
        if max_score <= 0:
            return []

        ranked = sorted(
            zip(docs, scores), key=lambda t: t[1], reverse=True
        )[:k]
        return [
            MemorySearchResult(
                memory=doc, score=float(score / (max_score + 1e-9)), source="bm25"
            )
            for doc, score in ranked
            if score > 0
        ]
