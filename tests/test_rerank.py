"""Cross-encoder + Cohere rerankers (mocked — no actual model download)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_system.core.memory_models import Memory, MemorySearchResult
from memory_system.retrieval.rerank import CohereReranker, CrossEncoderReranker


def _cands():
    return [
        MemorySearchResult(memory=Memory(id="a", text="apple fruit", user_id="u1"), score=0.3),
        MemorySearchResult(memory=Memory(id="b", text="banana fruit", user_id="u1"), score=0.5),
        MemorySearchResult(memory=Memory(id="c", text="random noise", user_id="u1"), score=0.7),
    ]


@pytest.mark.asyncio
async def test_cross_encoder_reorders_by_predicted_score():
    rr = CrossEncoderReranker(model="fake-model")
    fake_model = MagicMock()
    # Higher score for "apple" candidate when query is "apple"
    fake_model.predict = MagicMock(return_value=[0.9, 0.2, 0.05])
    rr._model = fake_model

    results = await rr.rerank("apple", _cands(), top_k=2)
    assert [r.memory.id for r in results] == ["a", "b"]
    assert all(r.source == "rerank" for r in results)


@pytest.mark.asyncio
async def test_cross_encoder_top_k_truncation():
    rr = CrossEncoderReranker(model="fake-model")
    fake_model = MagicMock()
    fake_model.predict = MagicMock(return_value=[0.1, 0.5, 0.9])
    rr._model = fake_model

    results = await rr.rerank("q", _cands(), top_k=1)
    assert len(results) == 1
    assert results[0].memory.id == "c"


@pytest.mark.asyncio
async def test_cross_encoder_empty_candidates_returns_empty():
    rr = CrossEncoderReranker(model="fake-model")
    rr._model = MagicMock()
    results = await rr.rerank("q", [], top_k=5)
    assert results == []
    rr._model.predict.assert_not_called()


@pytest.mark.asyncio
async def test_cohere_reranker_maps_indices_to_candidates():
    rr = CohereReranker(api_key="fake")
    # Mock cohere response: results have .index and .relevance_score
    res = MagicMock()
    res.results = [
        MagicMock(index=2, relevance_score=0.95),
        MagicMock(index=0, relevance_score=0.8),
    ]
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=res)
    rr._client = fake_client

    out = await rr.rerank("q", _cands(), top_k=2)
    assert [r.memory.id for r in out] == ["c", "a"]
    assert out[0].score == pytest.approx(0.95)
    assert all(r.source == "rerank" for r in out)
