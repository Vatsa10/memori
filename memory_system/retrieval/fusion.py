"""Reciprocal Rank Fusion (RRF) for merging ranked retrieval results."""

from typing import Optional

from memory_system.core.memory_models import MemorySearchResult


def reciprocal_rank_fusion(
    *ranked_lists: list[MemorySearchResult],
    k: int = 60,
    weights: Optional[list[float]] = None,
) -> list[MemorySearchResult]:
    """Fuse multiple ranked lists via RRF.

    score(doc) = sum_i weight_i * 1 / (k + rank_i)
    Results are deduplicated by memory.id; first MemorySearchResult wins.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match number of ranked_lists")

    fused: dict[str, tuple[MemorySearchResult, float]] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, result in enumerate(ranked, start=1):
            mid = result.memory.id
            contribution = weight * (1.0 / (k + rank))
            if mid in fused:
                existing, score = fused[mid]
                fused[mid] = (existing, score + contribution)
            else:
                fused[mid] = (result, contribution)

    merged = [
        MemorySearchResult(memory=r.memory, score=s, source="hybrid")
        for r, s in fused.values()
    ]
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged
