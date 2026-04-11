import pytest
from memory_system.memory.manager import MemoryManager
from memory_system.core.memory_models import Memory, MemoryType, MemoryExtractionResult
from memory_system.providers.in_memory_stores import InMemoryMemoryStore, InMemoryGraphStore


@pytest.fixture
def memory_store():
    return InMemoryMemoryStore()


@pytest.fixture
def graph_store():
    return InMemoryGraphStore()


@pytest.fixture
def manager(memory_store, graph_store):
    return MemoryManager(
        memory_store=memory_store,
        graph_store=graph_store,
        extraction_llm_fn=None,  # No LLM for unit tests
    )


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_add_memory(self, manager):
        mem = await manager.add_memory(
            text="User prefers morning deliveries",
            user_id="user1",
            memory_type=MemoryType.SEMANTIC,
        )
        assert mem.text == "User prefers morning deliveries"
        assert mem.user_id == "user1"
        assert mem.memory_type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_recall_after_add(self, manager):
        await manager.add_memory("User likes coffee", user_id="user1")
        await manager.add_memory("User lives in NYC", user_id="user1")

        results = await manager.recall(query="coffee preference", user_id="user1")
        assert len(results) > 0
        assert any("coffee" in r.memory.text for r in results)

    @pytest.mark.asyncio
    async def test_recall_filters_by_user(self, manager):
        await manager.add_memory("User likes tea", user_id="user1")
        await manager.add_memory("User likes coffee", user_id="user2")

        results = await manager.recall(query="likes", user_id="user1")
        assert all(r.memory.user_id == "user1" for r in results)

    @pytest.mark.asyncio
    async def test_delete_memory(self, manager, memory_store):
        mem = await manager.add_memory("Delete me", user_id="user1")
        await manager.delete_memory(mem.id)

        results = await manager.get_user_memories("user1")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_user_memories(self, manager):
        await manager.add_memory("Fact 1", user_id="user1")
        await manager.add_memory("Fact 2", user_id="user1")
        await manager.add_memory("Fact 3", user_id="user2")

        results = await manager.get_user_memories("user1")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_dedup_prevents_duplicate(self, manager):
        await manager.add_memory("User prefers morning deliveries", user_id="user1")

        # Remember should detect duplicate
        extraction = MemoryExtractionResult(
            memories=[Memory(
                text="User prefers morning deliveries",
                user_id="user1",
            )]
        )
        # Manually test dedup
        is_dup = await manager._is_duplicate(extraction.memories[0])
        assert is_dup is True

    @pytest.mark.asyncio
    async def test_remember_without_llm(self, manager):
        """Without extraction LLM, remember should return empty result."""
        result = await manager.remember(
            user_message="I like pizza",
            assistant_response="Noted!",
            user_id="user1",
        )
        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_recall_with_intent(self, manager):
        await manager.add_memory("Order ORD-123 shipped", user_id="user1")
        await manager.add_memory("User prefers blue color", user_id="user1")

        results = await manager.recall(
            query="Where is my order?",
            user_id="user1",
            intent_name="check_order",
        )
        # Should find order-related memory
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_recall_with_graph(self, manager, graph_store):
        from memory_system.core.memory_models import Entity, Relationship

        await graph_store.add_entity(Entity(name="pizza", entity_type="food", user_id="user1"))
        await graph_store.add_relationship(Relationship(
            source_entity="user1", target_entity="pizza",
            relation_type="prefers", user_id="user1",
        ))

        results = await manager.recall(query="pizza", user_id="user1")
        # Should include graph-sourced results
        graph_results = [r for r in results if r.source == "graph"]
        assert len(graph_results) > 0
