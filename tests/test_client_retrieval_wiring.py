"""MemorySystem honors injected retriever + reranker."""

from unittest.mock import AsyncMock

import pytest

from memory_system import MemorySystem
from memory_system.core.memory_models import Memory, MemorySearchResult
from memory_system.providers.in_memory_stores import InMemoryMemoryStore


def _mk_results(ids):
    return [
        MemorySearchResult(memory=Memory(id=i, text=f"t-{i}", user_id="u1"), score=0.5)
        for i in ids
    ]


@pytest.mark.asyncio
async def test_chat_uses_injected_retriever():
    store = InMemoryMemoryStore()
    await store.add(Memory(text="seed", user_id="u1"))

    fake_retriever = AsyncMock()
    fake_retriever.search = AsyncMock(return_value=_mk_results(["a", "b", "c"]))
    fake_retriever.bm25 = None

    async def fake_llm(model, messages):
        return "response"

    ms = MemorySystem(
        instructions="x",
        llm_fn=fake_llm,
        memory_store=store,
        retriever=fake_retriever,
        hybrid_top_n=10,
        memory_top_k=2,
    )
    result = await ms.chat("hello", user_id="u1")
    fake_retriever.search.assert_awaited_once()
    # Without reranker, we truncate to memory_top_k=2
    assert result.memories_recalled == 2
    assert "hybrid_retrieval_ms" in result.latency_ms


@pytest.mark.asyncio
async def test_chat_uses_injected_reranker():
    store = InMemoryMemoryStore()
    fake_retriever = AsyncMock()
    fake_retriever.search = AsyncMock(return_value=_mk_results(["a", "b", "c", "d"]))
    fake_retriever.bm25 = None

    fake_reranker = AsyncMock()
    fake_reranker.rerank = AsyncMock(return_value=_mk_results(["b", "a"]))

    async def fake_llm(model, messages):
        return "ok"

    ms = MemorySystem(
        instructions="x",
        llm_fn=fake_llm,
        memory_store=store,
        retriever=fake_retriever,
        reranker=fake_reranker,
        hybrid_top_n=4,
        rerank_top_k=2,
    )
    # Need a memory to make _memory.store truthy + remember stage runs
    await store.add(Memory(text="seed", user_id="u1"))

    result = await ms.chat("hello", user_id="u1")
    fake_reranker.rerank.assert_awaited_once()
    assert result.memories_recalled == 2
    assert "rerank_ms" in result.latency_ms


@pytest.mark.asyncio
async def test_default_behavior_without_retriever_unchanged():
    """No retriever/reranker injected → falls back to memory.recall as before."""
    store = InMemoryMemoryStore()
    await store.add(Memory(text="user prefers tea", user_id="u1"))

    async def fake_llm(model, messages):
        return "ok"

    ms = MemorySystem(
        instructions="x",
        llm_fn=fake_llm,
        memory_store=store,
    )
    result = await ms.chat("tea preference", user_id="u1")
    assert "memory_recall_ms" in result.latency_ms
    assert "hybrid_retrieval_ms" not in result.latency_ms


@pytest.mark.asyncio
async def test_bm25_bump_wired_to_memory_stored_event():
    from memory_system.retrieval.bm25 import BM25Retriever
    from memory_system.retrieval.hybrid import HybridRetriever

    store = InMemoryMemoryStore()
    bm = BM25Retriever(store)
    retriever = HybridRetriever(store, bm25=bm)

    async def fake_llm(model, messages):
        return "ok"

    ms = MemorySystem(
        instructions="x",
        llm_fn=fake_llm,
        memory_store=store,
        retriever=retriever,
    )

    # Emit MEMORIES_STORED manually and verify version bumped
    from memory_system.hooks import Event, EventType
    before = bm._versions.get("u1", 0)
    await ms._hooks.emit(
        Event(type=EventType.MEMORIES_STORED, data={"count": 1, "user_id": "u1"})
    )
    after = bm._versions.get("u1", 0)
    assert after == before + 1
