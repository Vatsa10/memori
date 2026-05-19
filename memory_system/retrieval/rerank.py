"""Cross-encoder + Cohere rerankers."""

import asyncio
from typing import Any, Optional, Protocol, runtime_checkable

from memory_system.core.memory_models import MemorySearchResult


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: list[MemorySearchResult],
        top_k: int = 5,
    ) -> list[MemorySearchResult]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder rerank, runs in a worker thread."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-base",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[MemorySearchResult],
        top_k: int = 5,
    ) -> list[MemorySearchResult]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.memory.text) for c in candidates]
        scores = await asyncio.to_thread(
            model.predict, pairs, batch_size=self.batch_size
        )
        ranked = sorted(
            zip(candidates, scores), key=lambda t: float(t[1]), reverse=True
        )
        return [
            MemorySearchResult(
                memory=c.memory, score=float(s), source="rerank"
            )
            for c, s in ranked[:top_k]
        ]


class CohereReranker:
    """Cohere rerank API adapter."""

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import cohere

            self._client = cohere.AsyncClient(api_key=self.api_key)
        return self._client

    async def rerank(
        self,
        query: str,
        candidates: list[MemorySearchResult],
        top_k: int = 5,
    ) -> list[MemorySearchResult]:
        if not candidates:
            return []
        client = self._get_client()
        documents = [c.memory.text for c in candidates]
        response = await client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
        )
        out = []
        for r in response.results:
            cand = candidates[r.index]
            out.append(
                MemorySearchResult(
                    memory=cand.memory,
                    score=float(r.relevance_score),
                    source="rerank",
                )
            )
        return out
