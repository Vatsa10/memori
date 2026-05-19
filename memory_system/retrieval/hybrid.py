"""HybridRetriever: dense + BM25 + graph paths, fused via RRF."""

import asyncio
from typing import Any, Optional

from memory_system.core.memory_models import (
    Memory,
    MemorySearchResult,
    MemoryType,
)
from memory_system.retrieval.fusion import reciprocal_rank_fusion


DEFAULT_WEIGHTS = {"dense": 1.0, "bm25": 1.0, "graph": 0.5}


class HybridRetriever:
    """Run dense + BM25 + graph in parallel, then fuse via RRF.

    Bi-temporal filtering is inherited from each source (memory_store.search,
    BM25 over get_all, graph.traverse) — HybridRetriever adds no filtering of
    its own.
    """

    def __init__(
        self,
        memory_store: Any,
        graph_store: Optional[Any] = None,
        bm25: Optional[Any] = None,
        *,
        weights: Optional[dict[str, float]] = None,
        top_n_per_source: int = 20,
        graph_max_hops: int = 2,
        graph_entity_k: int = 3,
    ):
        self.memory_store = memory_store
        self.graph_store = graph_store
        self.bm25 = bm25
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.top_n_per_source = top_n_per_source
        self.graph_max_hops = graph_max_hops
        self.graph_entity_k = graph_entity_k

    async def _dense(self, query: str, user_id: str) -> list[MemorySearchResult]:
        return await self.memory_store.search(
            query, user_id=user_id, k=self.top_n_per_source
        )

    async def _bm25_search(self, query: str, user_id: str) -> list[MemorySearchResult]:
        if self.bm25 is None:
            return []
        return await self.bm25.search(query, user_id=user_id, k=self.top_n_per_source)

    async def _graph_paths(self, query: str, user_id: str) -> list[MemorySearchResult]:
        if self.graph_store is None:
            return []
        entities = await self.graph_store.search_entities(
            query, user_id=user_id, k=self.graph_entity_k
        )
        traverse = getattr(self.graph_store, "traverse", None)
        results: list[MemorySearchResult] = []
        for entity in entities:
            if traverse is not None:
                paths = await traverse(
                    entity.name, user_id=user_id, max_hops=self.graph_max_hops
                )
                for path in paths:
                    text = " -> ".join(
                        f"{r.source_entity} [{r.relation_type}] {r.target_entity}"
                        for r in path
                    )
                    score = 0.7 * (0.8 ** (len(path) - 1))
                    results.append(
                        MemorySearchResult(
                            memory=Memory(
                                text=text,
                                memory_type=MemoryType.SEMANTIC,
                                user_id=user_id,
                                source="graph",
                            ),
                            score=score,
                            source="graph",
                        )
                    )
            else:
                rels = await self.graph_store.get_related(entity.name, user_id=user_id)
                for rel in rels:
                    text = f"{rel.source_entity} {rel.relation_type} {rel.target_entity}"
                    results.append(
                        MemorySearchResult(
                            memory=Memory(
                                text=text,
                                memory_type=MemoryType.SEMANTIC,
                                user_id=user_id,
                                source="graph",
                            ),
                            score=0.7,
                            source="graph",
                        )
                    )
        return results[: self.top_n_per_source]

    async def search(
        self, query: str, user_id: str, k: int = 5
    ) -> list[MemorySearchResult]:
        dense, bm, graph = await asyncio.gather(
            self._dense(query, user_id),
            self._bm25_search(query, user_id),
            self._graph_paths(query, user_id),
        )
        fused = reciprocal_rank_fusion(
            dense,
            bm,
            graph,
            weights=[self.weights["dense"], self.weights["bm25"], self.weights["graph"]],
        )
        return fused[:k]
