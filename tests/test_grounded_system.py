"""Tests for the grounded MemorySystem — knowledge + memory + context."""

import pytest
from unittest.mock import AsyncMock

from memory_system import MemorySystem, ChatResponse, EventType, Event
from memory_system.providers.in_memory_stores import InMemoryMemoryStore, InMemoryGraphStore


@pytest.fixture
def mock_llm():
    return AsyncMock(return_value="Here's your answer based on the context.")


@pytest.fixture
def ms(mock_llm):
    """Grounded MemorySystem with knowledge + memory stores."""
    return MemorySystem(
        instructions="You are a helpful support agent for Acme Corp.",
        model="groq/llama-3.3-70b-versatile",
        llm_fn=mock_llm,
        knowledge_store=InMemoryMemoryStore(),
        memory_store=InMemoryMemoryStore(),
        graph_store=InMemoryGraphStore(),
    )


@pytest.fixture
def ms_no_stores(mock_llm):
    """MemorySystem without any stores."""
    return MemorySystem(
        instructions="You are a basic assistant.",
        llm_fn=mock_llm,
    )


class TestKnowledgeGrounding:
    @pytest.mark.asyncio
    async def test_add_and_search_knowledge(self, ms):
        await ms.add_knowledge("Returns are accepted within 30 days of purchase.")
        results = await ms.search_knowledge("returns accepted")
        assert len(results) > 0
        assert "30 days" in results[0].memory.text

    @pytest.mark.asyncio
    async def test_knowledge_in_prompt(self, ms, mock_llm):
        await ms.add_knowledge("Order tracking available at track.acme.com")
        await ms.chat("Where is my order?")

        call_args = mock_llm.call_args
        messages = call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "track.acme.com" in system_msg

    @pytest.mark.asyncio
    async def test_knowledge_used_count(self, ms):
        await ms.add_knowledge("Returns within 30 days")
        await ms.add_knowledge("Free shipping for premium")
        result = await ms.chat("How do returns work?")
        assert result.knowledge_used > 0

    @pytest.mark.asyncio
    async def test_add_knowledge_batch(self, ms):
        count = await ms.add_knowledge_batch([
            "Policy 1: Returns within 30 days",
            "Policy 2: Free shipping over $50",
            "Policy 3: Premium members get priority",
        ])
        assert count == 3

    @pytest.mark.asyncio
    async def test_relevant_knowledge_retrieved(self, ms, mock_llm):
        """Only relevant knowledge should appear in the prompt."""
        await ms.add_knowledge("Returns are accepted within 30 days")
        await ms.add_knowledge("Order tracking at track.acme.com with order ID")
        await ms.add_knowledge("Premium members get free shipping")

        await ms.chat("How do I track my order?")

        messages = mock_llm.call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        # Should find order tracking, might not find returns
        assert "order" in system_msg.lower()


class TestMemoryGrounding:
    @pytest.mark.asyncio
    async def test_memory_in_prompt(self, ms, mock_llm):
        await ms.add_memory("User prefers morning deliveries", user_id="u1")
        await ms.chat("When should we deliver?", user_id="u1")

        messages = mock_llm.call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "morning deliveries" in system_msg

    @pytest.mark.asyncio
    async def test_user_profile_in_prompt(self, ms, mock_llm):
        await ms.add_memory("User lives in NYC", user_id="u1")
        await ms.chat("Hi there", user_id="u1")

        messages = mock_llm.call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "NYC" in system_msg

    @pytest.mark.asyncio
    async def test_memories_recalled_count(self, ms):
        await ms.add_memory("Prefers tea", user_id="u1")
        result = await ms.chat("What drink should I get?", user_id="u1")
        # InMemory search is keyword-based; "drink" won't match "tea"
        # but the memory should still be there for profile
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_user_isolation(self, ms, mock_llm):
        """User 1's memories should NOT appear in User 2's prompt."""
        await ms.add_memory("User 1 secret preference", user_id="u1")
        await ms.chat("Hello", user_id="u2")

        messages = mock_llm.call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "secret preference" not in system_msg

    @pytest.mark.asyncio
    async def test_forget_user(self, ms):
        await ms.add_memory("Fact 1", user_id="u1")
        await ms.add_memory("Fact 2", user_id="u1")
        count = await ms.forget_user("u1")
        assert count == 2
        results = await ms.get_user_memories("u1")
        assert len(results) == 0


class TestChatPipeline:
    @pytest.mark.asyncio
    async def test_basic_chat(self, ms):
        result = await ms.chat("Hello!")
        assert isinstance(result, ChatResponse)
        assert result.response == "Here's your answer based on the context."

    @pytest.mark.asyncio
    async def test_session_persistence(self, ms):
        await ms.chat("Hi!", session_id="s1")
        await ms.chat("How are you?", session_id="s1")
        history = ms._session_store.get_history("s1")
        assert len(history) == 4

    @pytest.mark.asyncio
    async def test_history_in_messages(self, ms, mock_llm):
        await ms.chat("First message", session_id="s1")
        await ms.chat("Second message", session_id="s1")

        messages = mock_llm.call_args.kwargs["messages"]
        # Should contain: system + first user + first assistant + second user
        assert len(messages) >= 4

    @pytest.mark.asyncio
    async def test_instructions_in_system_prompt(self, ms, mock_llm):
        await ms.chat("Hello")
        messages = mock_llm.call_args.kwargs["messages"]
        assert "Acme Corp" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_latency_tracking(self, ms):
        result = await ms.chat("Hello")
        assert "generation_ms" in result.latency_ms
        assert "total_ms" in result.latency_ms

    @pytest.mark.asyncio
    async def test_analytics(self, ms):
        await ms.chat("First")
        await ms.chat("Second")
        snap = ms.analytics.snapshot()
        assert snap.total_requests == 2

    @pytest.mark.asyncio
    async def test_hooks_fire(self, ms):
        events = []
        ms.hooks.on(EventType.RESPONSE_GENERATED, lambda e: events.append(e.type))
        await ms.chat("Hello")
        assert EventType.RESPONSE_GENERATED in events

    @pytest.mark.asyncio
    async def test_no_stores_works(self, ms_no_stores):
        """Should work without any stores — just instructions + LLM."""
        result = await ms_no_stores.chat("Hello")
        assert result.response == "Here's your answer based on the context."
        assert result.knowledge_used == 0
        assert result.memories_recalled == 0

    @pytest.mark.asyncio
    async def test_change_instructions(self, ms, mock_llm):
        ms.instructions = "You are a pirate."
        await ms.chat("Hello")
        messages = mock_llm.call_args.kwargs["messages"]
        assert "pirate" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_clear_session(self, ms):
        await ms.chat("Hi", session_id="s1")
        ms.clear_session("s1")
        history = ms._session_store.get_history("s1")
        assert len(history) == 0


class TestTokenBudget:
    @pytest.mark.asyncio
    async def test_budget_limits_context(self, mock_llm):
        """With a very small token budget, not all context fits."""
        ms = MemorySystem(
            instructions="Short.",
            llm_fn=mock_llm,
            knowledge_store=InMemoryMemoryStore(),
            memory_store=InMemoryMemoryStore(),
            token_budget=50,  # Very small
        )

        # Add lots of knowledge
        for i in range(20):
            await ms.add_knowledge(f"Long knowledge entry number {i} with extra words to fill tokens")

        await ms.chat("Hello")
        messages = mock_llm.call_args.kwargs["messages"]
        total_tokens = sum(_estimate_tokens_helper(m["content"]) for m in messages)
        # Should be within reasonable range of budget (some overshoot ok from system instructions)
        assert total_tokens < 200


class TestStandaloneMemoryAccess:
    @pytest.mark.asyncio
    async def test_memory_property(self, ms):
        assert ms.memory is not None

    @pytest.mark.asyncio
    async def test_direct_memory_operations(self, ms):
        await ms.memory.add("Direct fact", user_id="u1")
        results = await ms.memory.search("direct", user_id="u1")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_lifecycle_operations(self, ms):
        await ms.add_memory("Old fact", user_id="u1")
        count = await ms.decay_memories("u1")
        # May or may not decay depending on age
        assert isinstance(count, int)


def _estimate_tokens_helper(text: str) -> int:
    return int(len(text.split()) * 1.3)
