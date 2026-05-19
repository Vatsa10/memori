"""MemorySystem.ingest_document end-to-end with mocked extractors."""

from unittest.mock import AsyncMock

import pytest

from memory_system import MemorySystem
from memory_system.ingestion.chunker import Chunk, SemanticChunker
from memory_system.providers.in_memory_stores import InMemoryMemoryStore


def _mk_chunks(texts: list[str], source: str = "pdf"):
    return [
        Chunk(
            text=t,
            index=i,
            token_count=len(t.split()),
            metadata={"source": source, "filename": "fake.pdf"},
        )
        for i, t in enumerate(texts)
    ]


@pytest.mark.asyncio
async def test_ingest_pdf_writes_to_knowledge_store(monkeypatch):
    store = InMemoryMemoryStore()
    ms = MemorySystem(
        instructions="x", llm_fn=AsyncMock(return_value="ok"), knowledge_store=store
    )

    async def fake_ingest_pdf(*args, **kwargs):
        return _mk_chunks(["chunk one text", "chunk two text"], source="pdf")

    monkeypatch.setattr(
        "memory_system.ingestion.pdf.ingest_pdf", fake_ingest_pdf
    )

    result = await ms.ingest_document("doc.pdf", target="knowledge")
    assert len(result) == 2
    # Stored under KNOWLEDGE_USER_ID
    from memory_system._client import KNOWLEDGE_USER_ID
    all_stored = await store.get_all(KNOWLEDGE_USER_ID, k=100)
    assert len(all_stored) == 2
    assert all_stored[0].memory.metadata["source"] == "pdf"


@pytest.mark.asyncio
async def test_ingest_url_writes_to_user_memory(monkeypatch):
    store = InMemoryMemoryStore()
    ms = MemorySystem(
        instructions="x", llm_fn=AsyncMock(return_value="ok"), memory_store=store
    )

    async def fake_ingest_url(*args, **kwargs):
        return _mk_chunks(["page text"], source="url")

    monkeypatch.setattr(
        "memory_system.ingestion.url.ingest_url", fake_ingest_url
    )

    result = await ms.ingest_document(
        "https://example.com", target="memory", user_id="u1"
    )
    assert len(result) == 1
    assert result[0].user_id == "u1"
    assert result[0].metadata["source"] == "url"


@pytest.mark.asyncio
async def test_target_memory_without_user_id_raises():
    ms = MemorySystem(
        instructions="x", llm_fn=AsyncMock(), memory_store=InMemoryMemoryStore()
    )
    with pytest.raises(ValueError, match="user_id"):
        await ms.ingest_document("doc.pdf", target="memory")


@pytest.mark.asyncio
async def test_target_knowledge_without_store_raises():
    ms = MemorySystem(instructions="x", llm_fn=AsyncMock())
    with pytest.raises(RuntimeError, match="knowledge_store"):
        await ms.ingest_document("doc.pdf", target="knowledge")


@pytest.mark.asyncio
async def test_plain_text_falls_back_to_chunker():
    store = InMemoryMemoryStore()
    ms = MemorySystem(
        instructions="x", llm_fn=AsyncMock(), knowledge_store=store
    )

    chunker = SemanticChunker(max_tokens=200, overlap_tokens=20)
    result = await ms.ingest_document(
        "Just some plain text content here.", target="knowledge", chunker=chunker
    )
    assert len(result) >= 1
    assert "plain text" in result[0].text


@pytest.mark.asyncio
async def test_chunk_index_stamped_into_metadata(monkeypatch):
    store = InMemoryMemoryStore()
    ms = MemorySystem(instructions="x", llm_fn=AsyncMock(), knowledge_store=store)

    async def fake_ingest_pdf(*args, **kwargs):
        return _mk_chunks(["a", "b", "c"], source="pdf")

    monkeypatch.setattr(
        "memory_system.ingestion.pdf.ingest_pdf", fake_ingest_pdf
    )

    result = await ms.ingest_document("doc.pdf", target="knowledge")
    indexes = sorted(m.metadata["chunk_index"] for m in result)
    assert indexes == [0, 1, 2]
