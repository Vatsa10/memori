"""BM25Retriever: tokenization, lazy build, cache invalidation."""

import pytest

pytest.importorskip("rank_bm25")

from memory_system.core.memory_models import Memory
from memory_system.providers.in_memory_stores import InMemoryMemoryStore
from memory_system.retrieval.bm25 import BM25Retriever, _tokenize


def test_tokenizer_keeps_alphanumerics_and_hyphens():
    tokens = _tokenize("The user-id ABC-123 is for foo_bar!")
    assert "user-id" in tokens
    assert "abc-123" in tokens
    # underscores are not in the regex class → split
    assert "foo" in tokens and "bar" in tokens


def test_tokenizer_lowercases():
    assert _tokenize("HELLO World") == ["hello", "world"]


@pytest.mark.asyncio
async def test_search_returns_normalized_scores():
    store = InMemoryMemoryStore()
    await store.add(Memory(text="user prefers green tea every morning", user_id="u1"))
    await store.add(Memory(text="user dislikes coffee on weekends", user_id="u1"))
    await store.add(Memory(text="bob lives in seattle", user_id="u1"))

    bm = BM25Retriever(store)
    results = await bm.search("green tea morning", "u1", k=10)
    assert len(results) >= 1
    assert all(0.0 <= r.score <= 1.0 for r in results)
    # Top result should be the green tea memory
    assert "green tea" in results[0].memory.text


@pytest.mark.asyncio
async def test_search_empty_corpus_returns_empty():
    bm = BM25Retriever(InMemoryMemoryStore())
    assert await bm.search("anything", "u1", k=5) == []


@pytest.mark.asyncio
async def test_search_isolates_by_user():
    store = InMemoryMemoryStore()
    await store.add(Memory(text="alice loves jazz", user_id="u1"))
    await store.add(Memory(text="alice loves jazz", user_id="u2"))
    bm = BM25Retriever(store)

    r1 = await bm.search("jazz", "u1", k=10)
    assert all(r.memory.user_id == "u1" for r in r1)


@pytest.mark.asyncio
async def test_cache_rebuilds_after_bump():
    store = InMemoryMemoryStore()
    # Seed with multiple docs so BM25 IDF gives positive scores
    await store.add(Memory(text="initial fact about apples", user_id="u1"))
    await store.add(Memory(text="weather is sunny", user_id="u1"))
    await store.add(Memory(text="meeting at three pm", user_id="u1"))
    bm = BM25Retriever(store)

    before = await bm.search("apples", "u1", k=5)
    assert any("apple" in r.memory.text for r in before)

    # Add a new doc without bumping; cache is stale → new doc invisible
    await store.add(Memory(text="bananas are great", user_id="u1"))
    stale = await bm.search("bananas", "u1", k=5)
    assert not any("banana" in r.memory.text for r in stale)

    # Bump and re-query — new doc now indexed
    bm.bump("u1")
    fresh = await bm.search("bananas", "u1", k=5)
    assert any("banana" in r.memory.text for r in fresh)


@pytest.mark.asyncio
async def test_search_source_is_bm25():
    store = InMemoryMemoryStore()
    await store.add(Memory(text="fact x", user_id="u1"))
    bm = BM25Retriever(store)
    results = await bm.search("fact", "u1", k=5)
    assert all(r.source == "bm25" for r in results)


@pytest.mark.asyncio
async def test_invalidated_memories_excluded_from_bm25_corpus():
    from datetime import datetime, timezone

    store = InMemoryMemoryStore()
    await store.add(Memory(text="kept fact", user_id="u1"))
    await store.add(Memory(
        text="invalidated fact", user_id="u1",
        valid_to=datetime.now(timezone.utc),
    ))
    bm = BM25Retriever(store)
    results = await bm.search("invalidated", "u1", k=10)
    # get_all defaults to include_invalidated=False, so invalidated is excluded
    assert not any("invalidated" in r.memory.text for r in results)
