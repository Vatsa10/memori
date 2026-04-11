import pytest
from unittest.mock import AsyncMock

from memory_system import MemorySystem, ChatResponse, EventType, Event
from memory_system.core.models import IntentPrediction, PredictionMethod


@pytest.fixture
def mock_llm():
    return AsyncMock(return_value="Mocked response from LLM")


@pytest.fixture
def mock_intent_llm():
    """Returns a fallback IntentPrediction when keyword/embedding tiers don't match."""
    return AsyncMock(return_value=IntentPrediction(
        intent_name="fallback",
        confidence=0.5,
        method=PredictionMethod.LLM,
    ))


@pytest.fixture
def ctx(sample_bot_config, mock_llm, mock_intent_llm):
    return MemorySystem(
        config=sample_bot_config,
        llm_fn=mock_llm,
        intent_llm_fn=mock_intent_llm,
        enable_embeddings=False,
        cache_size=64,
        enable_analytics=True,
    )


class TestMemorySystem:
    @pytest.mark.asyncio
    async def test_chat_returns_response(self, ctx):
        result = await ctx.chat("Where is my order?")
        assert isinstance(result, ChatResponse)
        assert result.response == "Mocked response from LLM"
        assert result.intent.intent_name == "check_order"

    @pytest.mark.asyncio
    async def test_chat_tracks_reduction(self, ctx):
        result = await ctx.chat("Where is my order?")
        assert result.reduction_percent > 0
        assert result.token_estimate > 0
        assert result.full_prompt_estimate > result.token_estimate

    @pytest.mark.asyncio
    async def test_session_persistence(self, ctx):
        await ctx.chat("Where is my order?", session_id="s1")
        await ctx.chat("What about ORD-123?", session_id="s1")

        history = ctx._session_store.get_history("s1")
        assert len(history) == 4  # 2 user + 2 assistant turns

    @pytest.mark.asyncio
    async def test_separate_sessions(self, ctx):
        await ctx.chat("Where is my order?", session_id="s1")
        await ctx.chat("I want a refund", session_id="s2")

        assert len(ctx._session_store.get_history("s1")) == 2
        assert len(ctx._session_store.get_history("s2")) == 2

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeat(self, ctx):
        await ctx.chat("Where is my order?")
        await ctx.chat("Where is my order?")

        assert ctx.cache.hits == 1
        assert ctx.cache.misses == 1

    @pytest.mark.asyncio
    async def test_analytics_populated(self, ctx):
        await ctx.chat("Where is my order?")
        await ctx.chat("I want a refund")

        snapshot = ctx.analytics.snapshot()
        assert snapshot.total_requests == 2
        assert "check_order" in snapshot.intent_distribution
        assert "return_item" in snapshot.intent_distribution

    @pytest.mark.asyncio
    async def test_hooks_fire(self, ctx):
        events_received = []

        def on_intent(event: Event):
            events_received.append(event.type)

        ctx.hooks.on(EventType.INTENT_PREDICTED, on_intent)
        ctx.hooks.on(EventType.RESPONSE_GENERATED, on_intent)

        await ctx.chat("Where is my order?")

        assert EventType.INTENT_PREDICTED in events_received
        assert EventType.RESPONSE_GENERATED in events_received

    @pytest.mark.asyncio
    async def test_clear_session(self, ctx):
        await ctx.chat("Hi", session_id="s1")
        assert len(ctx._session_store.get_history("s1")) == 2
        ctx.clear_session("s1")
        assert len(ctx._session_store.get_history("s1")) == 0

    @pytest.mark.asyncio
    async def test_clear_cache(self, ctx):
        await ctx.chat("Where is my order?")
        assert ctx.cache.size > 0
        ctx.clear_cache()
        assert ctx.cache.size == 0

    @pytest.mark.asyncio
    async def test_export_analytics(self, ctx):
        await ctx.chat("Where is my order?")
        data = ctx.export_analytics()
        assert "total_requests" in data
        assert data["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_predict_intent_standalone(self, ctx):
        prediction = await ctx.predict_intent("I want to return this")
        assert prediction.intent_name == "return_item"

    @pytest.mark.asyncio
    async def test_latency_breakdown(self, ctx):
        result = await ctx.chat("Where is my order?")
        assert "intent_prediction_ms" in result.latency_ms
        assert "context_assembly_ms" in result.latency_ms
        assert "generation_ms" in result.latency_ms
        assert "total_ms" in result.latency_ms


class TestMemorySystemFactory:
    def test_from_yaml(self, tmp_path):
        config_path = tmp_path / "test.yaml"
        config_path.write_text("""
bot_id: yaml_test
bot_name: YamlBot
base_instructions: "Test bot"
intents:
  - name: greet
    description: User says hello
    keywords: ["hello", "hi"]
    instructions: "Say hello back"
""")
        ctx = MemorySystem.from_yaml(
            config_path,
            llm_fn=AsyncMock(return_value="hi"),
            enable_embeddings=False,
        )
        assert ctx.config.bot_id == "yaml_test"

    def test_from_dict(self):
        ctx = MemorySystem.from_dict(
            {
                "bot_id": "dict_test",
                "bot_name": "DictBot",
                "base_instructions": "Test",
                "intents": [],
            },
            llm_fn=AsyncMock(return_value="ok"),
            enable_embeddings=False,
        )
        assert ctx.config.bot_id == "dict_test"
