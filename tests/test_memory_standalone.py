import pytest

from memory_system.memory.memory import Memory
from memory_system.core.memory_models import MemoryType
from memory_system.providers.in_memory_stores import InMemoryMemoryStore, InMemoryGraphStore


@pytest.fixture
def mem():
    return Memory(
        store=InMemoryMemoryStore(),
        graph=InMemoryGraphStore(),
    )


class TestStandaloneMemory:
    @pytest.mark.asyncio
    async def test_add_and_search(self, mem):
        await mem.add("User prefers morning deliveries", user_id="u1")
        results = await mem.search("morning delivery", user_id="u1")
        assert len(results) > 0
        assert "morning" in results[0].memory.text

    @pytest.mark.asyncio
    async def test_add_with_importance(self, mem):
        m = await mem.add("Critical fact", user_id="u1", importance=0.9)
        assert m.importance == 0.9

    @pytest.mark.asyncio
    async def test_add_with_ttl(self, mem):
        m = await mem.add("Temporary fact", user_id="u1", ttl=3600)
        assert m.ttl == 3600
        assert m.expires_at is not None
        assert not m.is_expired

    @pytest.mark.asyncio
    async def test_search_filters_by_user(self, mem):
        await mem.add("Fact for u1", user_id="u1")
        await mem.add("Fact for u2", user_id="u2")
        results = await mem.search("fact", user_id="u1")
        assert all(r.memory.user_id == "u1" for r in results)

    @pytest.mark.asyncio
    async def test_update(self, mem):
        m = await mem.add("Old text", user_id="u1")
        await mem.update(m.id, "New text")
        results = await mem.get_all("u1")
        assert any("New text" in r.memory.text for r in results)

    @pytest.mark.asyncio
    async def test_delete(self, mem):
        m = await mem.add("Delete me", user_id="u1")
        await mem.delete(m.id)
        results = await mem.get_all("u1")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_all_by_type(self, mem):
        await mem.add("A fact", user_id="u1", memory_type=MemoryType.SEMANTIC)
        await mem.add("An event", user_id="u1", memory_type=MemoryType.EPISODIC)
        results = await mem.get_all("u1", memory_type=MemoryType.SEMANTIC)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_recall_includes_graph(self, mem):
        from memory_system.core.memory_models import Entity, Relationship
        await mem.graph.add_entity(Entity(name="pizza", entity_type="food", user_id="u1"))
        await mem.graph.add_relationship(Relationship(
            source_entity="user", target_entity="pizza",
            relation_type="likes", user_id="u1",
        ))
        results = await mem.recall("pizza", user_id="u1")
        graph_results = [r for r in results if r.source == "graph"]
        assert len(graph_results) > 0

    @pytest.mark.asyncio
    async def test_forget(self, mem):
        await mem.add("Fact 1", user_id="u1")
        await mem.add("Fact 2", user_id="u1")
        count = await mem.forget("u1")
        assert count == 2
        results = await mem.get_all("u1")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_remember_without_llm(self, mem):
        """Without LLM, remember extracts nothing."""
        result = await mem.remember(
            messages=[
                {"role": "user", "content": "I like pizza"},
                {"role": "assistant", "content": "Noted!"},
            ],
            user_id="u1",
        )
        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_format_memories(self, mem):
        await mem.add("Likes coffee", user_id="u1")
        results = await mem.search("coffee", user_id="u1")
        formatted = mem.format_memories(results, format="bullet")
        assert formatted.startswith("- ")

    @pytest.mark.asyncio
    async def test_format_numbered(self, mem):
        await mem.add("Likes coffee", user_id="u1")
        results = await mem.search("coffee", user_id="u1")
        formatted = mem.format_memories(results, format="numbered")
        assert formatted.startswith("1. ")

    @pytest.mark.asyncio
    async def test_stats(self, mem):
        await mem.add("Fact 1", user_id="u1", memory_type=MemoryType.SEMANTIC)
        await mem.add("Event 1", user_id="u1", memory_type=MemoryType.EPISODIC)
        stats = await mem.stats("u1")
        assert stats.total_memories == 2
        assert stats.by_type["semantic"] == 1
        assert stats.by_type["episodic"] == 1

    @pytest.mark.asyncio
    async def test_user_profile(self, mem):
        await mem.add("User lives in NYC", user_id="u1")
        await mem.add("User prefers morning deliveries", user_id="u1")
        profile = await mem.get_user_profile("u1")
        assert profile.user_id == "u1"
        assert profile.memory_count == 2

    @pytest.mark.asyncio
    async def test_context_window(self, mem):
        await mem.add("User likes coffee", user_id="u1", importance=0.8)
        await mem.add("User lives in NYC", user_id="u1", importance=0.9)
        context = await mem.get_context_window("u1", "preferences", token_budget=500)
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_no_store(self):
        """Memory works without a store (returns empty results)."""
        mem = Memory()
        results = await mem.search("anything", user_id="u1")
        assert results == []
