from datetime import datetime, timedelta, timezone

import pytest

from memory_system.core.memory_models import Memory
from memory_system.providers.in_memory_stores import InMemoryMemoryStore


def _t(offset_seconds: float = 0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


class TestMemoryBitemporalProperties:
    def test_defaults_make_memory_current(self):
        mem = Memory(text="fact", user_id="u1")
        assert mem.is_current
        assert mem.valid_to is None
        assert mem.superseded_by is None
        assert mem.valid_from is not None
        assert mem.recorded_at is not None

    def test_is_valid_at_within_window(self):
        now = _t()
        mem = Memory(
            text="fact",
            user_id="u1",
            valid_from=now - timedelta(seconds=60),
            valid_to=now + timedelta(seconds=60),
        )
        assert mem.is_valid_at(now)

    def test_is_valid_at_before_valid_from(self):
        now = _t()
        mem = Memory(text="fact", user_id="u1", valid_from=now)
        assert not mem.is_valid_at(now - timedelta(seconds=10))

    def test_is_valid_at_after_valid_to(self):
        now = _t()
        mem = Memory(
            text="fact",
            user_id="u1",
            valid_from=now - timedelta(seconds=60),
            valid_to=now,
        )
        assert not mem.is_valid_at(now + timedelta(seconds=10))


class TestInMemoryStoreBitemporal:
    @pytest.mark.asyncio
    async def test_default_search_excludes_invalidated(self):
        store = InMemoryMemoryStore()
        keep = Memory(text="user lives in SF", user_id="u1")
        gone = Memory(text="user lives in NYC", user_id="u1", valid_to=_t())
        await store.add(keep)
        await store.add(gone)

        results = await store.search("lives", user_id="u1")
        ids = [r.memory.id for r in results]
        assert keep.id in ids
        assert gone.id not in ids

    @pytest.mark.asyncio
    async def test_include_invalidated_returns_all(self):
        store = InMemoryMemoryStore()
        keep = Memory(text="user lives in SF", user_id="u1")
        gone = Memory(text="user lives in NYC", user_id="u1", valid_to=_t())
        await store.add(keep)
        await store.add(gone)

        results = await store.search(
            "lives", user_id="u1", include_invalidated=True
        )
        ids = {r.memory.id for r in results}
        assert keep.id in ids
        assert gone.id in ids

    @pytest.mark.asyncio
    async def test_get_all_default_excludes_invalidated(self):
        store = InMemoryMemoryStore()
        keep = Memory(text="A", user_id="u1")
        gone = Memory(text="B", user_id="u1", valid_to=_t())
        await store.add(keep)
        await store.add(gone)

        results = await store.get_all("u1")
        ids = {r.memory.id for r in results}
        assert ids == {keep.id}

    @pytest.mark.asyncio
    async def test_invalidate_sets_valid_to_and_superseded_by(self):
        store = InMemoryMemoryStore()
        mem = Memory(text="user lives in NYC", user_id="u1")
        await store.add(mem)

        cutoff = _t()
        await store.invalidate(mem.id, valid_to=cutoff, superseded_by="new-id")

        all_results = await store.get_all("u1", include_invalidated=True)
        invalidated = next(r.memory for r in all_results if r.memory.id == mem.id)
        assert invalidated.valid_to == cutoff
        assert invalidated.superseded_by == "new-id"

    @pytest.mark.asyncio
    async def test_search_at_returns_state_at_point_in_time(self):
        store = InMemoryMemoryStore()
        t_old = _t(-3600)
        t_mid = _t(-1800)
        t_now = _t()

        old_fact = Memory(
            text="user lives in NYC", user_id="u1", valid_from=t_old, valid_to=t_mid
        )
        new_fact = Memory(
            text="user lives in SF", user_id="u1", valid_from=t_mid
        )
        await store.add(old_fact)
        await store.add(new_fact)

        before = await store.search_at("lives", "u1", as_of=t_old + timedelta(seconds=60))
        assert any("NYC" in r.memory.text for r in before)
        assert not any("SF" in r.memory.text for r in before)

        after = await store.search_at("lives", "u1", as_of=t_now)
        assert any("SF" in r.memory.text for r in after)
        assert not any("NYC" in r.memory.text for r in after)
