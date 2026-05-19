"""Streaming ingest: async iterator that yields persisted Memories."""

from unittest.mock import AsyncMock

import pytest

from memory_system import MemorySystem
from memory_system.ingestion.chunker import Chunk
from memory_system.providers.in_memory_stores import InMemoryMemoryStore


def _mk_chunks(n: int):
    return [
        Chunk(text=f"chunk {i}", index=i, token_count=2, metadata={"source": "stream"})
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_ingest_stream_yields_each_memory():
    store = InMemoryMemoryStore()
    ms = MemorySystem(instructions="x", llm_fn=AsyncMock(), knowledge_store=store)

    collected = []
    async for mem in ms.ingest_stream(_mk_chunks(5), target="knowledge", batch_size=2):
        collected.append(mem)

    assert len(collected) == 5
    assert all("chunk" in m.text for m in collected)


@pytest.mark.asyncio
async def test_ingest_stream_accepts_async_iterable():
    store = InMemoryMemoryStore()
    ms = MemorySystem(instructions="x", llm_fn=AsyncMock(), knowledge_store=store)

    async def chunk_source():
        for c in _mk_chunks(3):
            yield c

    out = [m async for m in ms.ingest_stream(chunk_source(), target="knowledge")]
    assert len(out) == 3


@pytest.mark.asyncio
async def test_ingest_stream_target_memory_requires_user_id():
    ms = MemorySystem(
        instructions="x", llm_fn=AsyncMock(), memory_store=InMemoryMemoryStore()
    )
    with pytest.raises(ValueError, match="user_id"):
        async for _ in ms.ingest_stream(_mk_chunks(1), target="memory"):
            pass


@pytest.mark.asyncio
async def test_ingest_document_stream_returns_async_iterator(monkeypatch):
    store = InMemoryMemoryStore()
    ms = MemorySystem(instructions="x", llm_fn=AsyncMock(), knowledge_store=store)

    async def fake_pdf(*args, **kwargs):
        return _mk_chunks(4)

    monkeypatch.setattr("memory_system.ingestion.pdf.ingest_pdf", fake_pdf)

    result = await ms.ingest_document("x.pdf", target="knowledge", stream=True)
    assert hasattr(result, "__aiter__")
    items = [m async for m in result]
    assert len(items) == 4


@pytest.mark.asyncio
async def test_ingest_stream_composes_with_smart_ops_when_target_is_memory(monkeypatch):
    """target='memory' goes through memory_store.add directly (smart_ops still
    governs only the remember() path — ingestion intentionally bypasses it
    so document ingest is deterministic, not LLM-gated)."""
    store = InMemoryMemoryStore()
    ms = MemorySystem(
        instructions="x",
        llm_fn=AsyncMock(),
        memory_store=store,
        enable_smart_ops=True,
    )

    out = [m async for m in ms.ingest_stream(
        _mk_chunks(3), target="memory", user_id="u1"
    )]
    assert len(out) == 3
    persisted = await store.get_all("u1", k=100)
    assert len(persisted) == 3
