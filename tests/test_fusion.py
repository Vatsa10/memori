"""Reciprocal Rank Fusion correctness."""

import pytest

from memory_system.core.memory_models import Memory, MemorySearchResult
from memory_system.retrieval.fusion import reciprocal_rank_fusion


def _mk(id_, text="x", score=0.0):
    return MemorySearchResult(memory=Memory(id=id_, text=text, user_id="u1"), score=score)


def test_empty_inputs_return_empty():
    assert reciprocal_rank_fusion() == []
    assert reciprocal_rank_fusion([], [], []) == []


def test_single_source_passthrough_preserves_order():
    ranked = [_mk("a"), _mk("b"), _mk("c")]
    out = reciprocal_rank_fusion(ranked)
    assert [r.memory.id for r in out] == ["a", "b", "c"]
    # Scores monotonically decreasing
    assert out[0].score > out[1].score > out[2].score


def test_dedup_by_id_sums_scores():
    list1 = [_mk("a"), _mk("b")]
    list2 = [_mk("a"), _mk("c")]
    out = reciprocal_rank_fusion(list1, list2, k=60)
    ids = [r.memory.id for r in out]
    assert ids.count("a") == 1
    # "a" appears in both at rank 1 → score is 2 * 1/(60+1) = ~0.0328
    a = next(r for r in out if r.memory.id == "a")
    assert pytest.approx(a.score, rel=1e-6) == 2 * (1 / 61)
    # "a" beats "b" and "c"
    assert ids[0] == "a"


def test_weights_scale_contributions():
    list1 = [_mk("only_in_1")]
    list2 = [_mk("only_in_2")]
    out = reciprocal_rank_fusion(list1, list2, k=60, weights=[3.0, 1.0])
    # only_in_1 gets 3 * 1/61, only_in_2 gets 1/61 → only_in_1 wins
    assert out[0].memory.id == "only_in_1"


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([_mk("a")], [_mk("b")], weights=[1.0])


def test_source_marked_hybrid():
    out = reciprocal_rank_fusion([_mk("a")])
    assert out[0].source == "hybrid"
