"""Memory lifecycle operations: decay, consolidate, cleanup."""

import math
from datetime import datetime, timezone
from typing import Optional

from memory_system.core.protocols import MemoryStore


async def decay_memories(
    store: MemoryStore,
    user_id: Optional[str] = None,
    half_life_days: float = 30.0,
    min_importance: float = 0.01,
) -> int:
    """
    Reduce importance of memories over time using exponential decay.
    Memories accessed recently decay slower (access_count acts as a boost).
    """
    if not user_id:
        return 0

    all_mems = await store.get_all(user_id, k=10000)
    now = datetime.now(timezone.utc)
    updated = 0

    for r in all_mems:
        mem = r.memory
        age_days = (now - mem.created_at).total_seconds() / 86400
        # Exponential decay: importance * 2^(-age/half_life)
        # Access count slows decay: effective half_life = half_life * (1 + access_count * 0.1)
        effective_half_life = half_life_days * (1 + mem.access_count * 0.1)
        decay_factor = math.pow(2, -age_days / effective_half_life)
        new_importance = max(mem.importance * decay_factor, min_importance)

        if abs(new_importance - mem.importance) > 0.01:
            mem.importance = round(new_importance, 3)
            mem.updated_at = now
            await store.update(mem.id, mem.text)
            updated += 1

    return updated


async def consolidate_memories(
    store: MemoryStore,
    user_id: str,
    similarity_threshold: float = 0.85,
) -> int:
    """
    Find similar memories and merge them into one stronger memory.
    The merged memory gets higher importance and combined metadata.
    """
    all_mems = await store.get_all(user_id, k=10000)
    if len(all_mems) < 2:
        return 0

    merged = 0
    deleted_ids = set()

    for i, r1 in enumerate(all_mems):
        if r1.memory.id in deleted_ids:
            continue

        for r2 in all_mems[i + 1:]:
            if r2.memory.id in deleted_ids:
                continue

            # Check similarity by searching for r1's text
            similar = await store.search(r1.memory.text, user_id=user_id, k=2)
            is_similar = any(
                s.memory.id == r2.memory.id and s.score >= similarity_threshold
                for s in similar
            )

            if is_similar:
                # Keep the one with higher importance, boost it
                if r1.memory.importance >= r2.memory.importance:
                    keep, remove = r1, r2
                else:
                    keep, remove = r2, r1

                # Boost importance of kept memory
                keep.memory.importance = min(1.0, keep.memory.importance + 0.1)
                keep.memory.access_count += remove.memory.access_count
                keep.memory.updated_at = datetime.now(timezone.utc)
                await store.update(keep.memory.id, keep.memory.text)

                # Delete the duplicate
                await store.delete(remove.memory.id)
                deleted_ids.add(remove.memory.id)
                merged += 1

    return merged


async def cleanup_expired(
    store: MemoryStore,
    user_id: Optional[str] = None,
    min_importance: float = 0.01,
) -> int:
    """Remove expired memories and memories below minimum importance."""
    if not user_id:
        return 0

    all_mems = await store.get_all(user_id, k=10000)
    removed = 0

    for r in all_mems:
        mem = r.memory
        should_remove = mem.is_expired or mem.importance < min_importance

        if should_remove:
            await store.delete(mem.id)
            removed += 1

    return removed
