import pytest
from memory_system.providers.in_memory_stores import InMemoryMemoryStore, InMemoryGraphStore
from memory_system.core.memory_models import Memory, MemoryType, Entity, Relationship


class TestInMemoryMemoryStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        store = InMemoryMemoryStore()
        mem = Memory(text="User prefers tea", user_id="u1", memory_type=MemoryType.SEMANTIC)
        await store.add(mem)

        results = await store.search("tea preference", user_id="u1")
        assert len(results) == 1
        assert "tea" in results[0].memory.text

    @pytest.mark.asyncio
    async def test_search_filters_by_user(self):
        store = InMemoryMemoryStore()
        await store.add(Memory(text="tea", user_id="u1"))
        await store.add(Memory(text="coffee", user_id="u2"))

        results = await store.search("beverage", user_id="u1")
        assert all(r.memory.user_id == "u1" for r in results)

    @pytest.mark.asyncio
    async def test_update(self):
        store = InMemoryMemoryStore()
        mem = Memory(text="old text", user_id="u1")
        await store.add(mem)
        await store.update(mem.id, "new text")

        all_mems = await store.get_all("u1")
        assert all_mems[0].memory.text == "new text"

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryMemoryStore()
        mem = Memory(text="delete me", user_id="u1")
        await store.add(mem)
        await store.delete(mem.id)

        all_mems = await store.get_all("u1")
        assert len(all_mems) == 0

    @pytest.mark.asyncio
    async def test_get_all(self):
        store = InMemoryMemoryStore()
        await store.add(Memory(text="a", user_id="u1"))
        await store.add(Memory(text="b", user_id="u1"))
        await store.add(Memory(text="c", user_id="u2"))

        results = await store.get_all("u1")
        assert len(results) == 2


class TestInMemoryGraphStore:
    @pytest.mark.asyncio
    async def test_add_entity(self):
        store = InMemoryGraphStore()
        await store.add_entity(Entity(name="pizza", entity_type="food", user_id="u1"))

        results = await store.search_entities("pizza", user_id="u1")
        assert len(results) == 1
        assert results[0].name == "pizza"

    @pytest.mark.asyncio
    async def test_add_relationship(self):
        store = InMemoryGraphStore()
        await store.add_relationship(Relationship(
            source_entity="user", target_entity="pizza",
            relation_type="likes", user_id="u1",
        ))

        results = await store.get_related("user", user_id="u1")
        assert len(results) == 1
        assert results[0].relation_type == "likes"

    @pytest.mark.asyncio
    async def test_entity_dedup(self):
        store = InMemoryGraphStore()
        await store.add_entity(Entity(name="pizza", entity_type="food", user_id="u1"))
        await store.add_entity(Entity(name="pizza", entity_type="food", user_id="u1", properties={"spicy": True}))

        results = await store.search_entities("pizza", user_id="u1")
        assert len(results) == 1
        assert results[0].properties.get("spicy") is True

    @pytest.mark.asyncio
    async def test_filter_by_relation_type(self):
        store = InMemoryGraphStore()
        await store.add_relationship(Relationship(
            source_entity="user", target_entity="pizza", relation_type="likes", user_id="u1",
        ))
        await store.add_relationship(Relationship(
            source_entity="user", target_entity="NYC", relation_type="lives_in", user_id="u1",
        ))

        likes = await store.get_related("user", user_id="u1", relation_type="likes")
        assert len(likes) == 1
        assert likes[0].target_entity == "pizza"

    @pytest.mark.asyncio
    async def test_user_isolation(self):
        store = InMemoryGraphStore()
        await store.add_entity(Entity(name="pizza", entity_type="food", user_id="u1"))
        await store.add_entity(Entity(name="sushi", entity_type="food", user_id="u2"))

        results = await store.search_entities("food", user_id="u1")
        assert all(r.user_id == "u1" for r in results)
