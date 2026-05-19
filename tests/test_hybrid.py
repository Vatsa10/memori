"""HybridRetriever: parallel sources + RRF fusion."""

import pytest

pytest.importorskip("rank_bm25")

from memory_system.core.memory_models import Entity, Memory, Relationship
from memory_system.providers.in_memory_stores import (
    InMemoryGraphStore,
    InMemoryMemoryStore,
)
from memory_system.retrieval.bm25 import BM25Retriever
from memory_system.retrieval.hybrid import HybridRetriever


@pytest.mark.asyncio
async def test_hybrid_fuses_dense_and_bm25():
    store = InMemoryMemoryStore()
    await store.add(Memory(text="user prefers green tea", user_id="u1"))
    await store.add(Memory(text="ordering matcha is fine", user_id="u1"))

    bm = BM25Retriever(store)
    h = HybridRetriever(store, bm25=bm)
    results = await h.search("green tea", "u1", k=5)
    assert len(results) >= 1
    assert all(r.source == "hybrid" for r in results)


@pytest.mark.asyncio
async def test_hybrid_works_without_bm25():
    store = InMemoryMemoryStore()
    await store.add(Memory(text="abc def", user_id="u1"))
    h = HybridRetriever(store)
    results = await h.search("abc", "u1", k=5)
    assert all(r.source == "hybrid" for r in results)


@pytest.mark.asyncio
async def test_hybrid_includes_graph_paths():
    store = InMemoryMemoryStore()
    graph = InMemoryGraphStore()
    await graph.add_entity(Entity(name="Alice", entity_type="person", user_id="u1"))
    await graph.add_entity(Entity(name="Bob", entity_type="person", user_id="u1"))
    await graph.add_relationship(Relationship(
        source_entity="Alice", target_entity="Bob",
        relation_type="manages", user_id="u1",
    ))

    bm = BM25Retriever(store)
    h = HybridRetriever(store, graph_store=graph, bm25=bm)
    results = await h.search("Alice", "u1", k=10)
    # graph contributes a path; survives fusion
    texts = [r.memory.text for r in results]
    assert any("manages" in t for t in texts)


@pytest.mark.asyncio
async def test_hybrid_excludes_invalidated():
    from datetime import datetime, timezone

    store = InMemoryMemoryStore()
    await store.add(Memory(text="current fact xyz", user_id="u1"))
    await store.add(Memory(
        text="stale fact xyz", user_id="u1",
        valid_to=datetime.now(timezone.utc),
    ))
    bm = BM25Retriever(store)
    h = HybridRetriever(store, bm25=bm)
    results = await h.search("xyz", "u1", k=10)
    assert not any("stale" in r.memory.text for r in results)
