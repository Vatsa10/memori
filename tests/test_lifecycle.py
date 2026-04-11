import pytest
from datetime import datetime, timedelta, timezone

from memory_system.core.memory_models import Memory, MemoryType
from memory_system.providers.in_memory_stores import InMemoryMemoryStore
from memory_system.memory.lifecycle import decay_memories, cleanup_expired


@pytest.fixture
def store():
    return InMemoryMemoryStore()


class TestDecay:
    @pytest.mark.asyncio
    async def test_old_memories_decay(self, store):
        old_mem = Memory(
            text="Old fact",
            user_id="u1",
            importance=0.8,
            created_at=datetime.now(timezone.utc) - timedelta(days=60),
        )
        await store.add(old_mem)

        updated = await decay_memories(store, "u1", half_life_days=30)
        assert updated >= 1

        results = await store.get_all("u1")
        # Importance should have decreased
        assert results[0].memory.importance < 0.8

    @pytest.mark.asyncio
    async def test_recent_memories_dont_decay_much(self, store):
        recent = Memory(
            text="Recent fact",
            user_id="u1",
            importance=0.8,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await store.add(recent)

        await decay_memories(store, "u1", half_life_days=30)
        results = await store.get_all("u1")
        # Should barely change
        assert results[0].memory.importance > 0.7

    @pytest.mark.asyncio
    async def test_no_user_returns_zero(self, store):
        result = await decay_memories(store, None)
        assert result == 0


class TestCleanup:
    @pytest.mark.asyncio
    async def test_removes_expired(self, store):
        expired = Memory(
            text="Expired",
            user_id="u1",
            ttl=1,  # 1 second TTL
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await store.add(expired)

        removed = await cleanup_expired(store, "u1")
        assert removed == 1

    @pytest.mark.asyncio
    async def test_removes_zero_importance(self, store):
        low = Memory(text="Unimportant", user_id="u1", importance=0.005)
        await store.add(low)

        removed = await cleanup_expired(store, "u1", min_importance=0.01)
        assert removed == 1

    @pytest.mark.asyncio
    async def test_keeps_valid_memories(self, store):
        valid = Memory(text="Valid", user_id="u1", importance=0.5)
        await store.add(valid)

        removed = await cleanup_expired(store, "u1")
        assert removed == 0
        results = await store.get_all("u1")
        assert len(results) == 1
