import pytest
from unittest.mock import AsyncMock

from memory_system.core.pipeline import Pipeline
from memory_system.core.intent_predictor import IntentPredictor
from memory_system.core.models import IntentPrediction, PredictionMethod
from memory_system.providers.memory import InMemoryProvider


def _make_pipeline(llm_mock: AsyncMock, intent_llm_mock: AsyncMock | None = None):
    predictor = IntentPredictor()
    memory = InMemoryProvider()
    memory.add("Order ORD-123 shipped on Jan 5")
    return Pipeline(
        intent_predictor=predictor,
        memory_provider=memory,
        llm_fn=llm_mock,
        intent_llm_fn=intent_llm_mock or AsyncMock(),
    )


class TestPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocked_llm(self, sample_bot_config):
        mock_response = "Your order ORD-123 shipped on Jan 5 and is on its way!"
        mock_llm = AsyncMock(return_value=mock_response)
        pipeline = _make_pipeline(mock_llm)

        result = await pipeline.run(
            bot_config=sample_bot_config,
            user_message="Where is my order ORD-123?",
            conversation_history=[],
        )

        assert result.response == mock_response
        assert result.intent.intent_name == "check_order"
        assert "intent_prediction_ms" in result.latency_ms
        assert "context_assembly_ms" in result.latency_ms
        assert "generation_ms" in result.latency_ms
        assert "total_ms" in result.latency_ms

    @pytest.mark.asyncio
    async def test_pipeline_token_reduction(self, sample_bot_config, sample_history):
        mock_llm = AsyncMock(return_value="Here's your order info.")
        pipeline = _make_pipeline(mock_llm)

        result = await pipeline.run(
            bot_config=sample_bot_config,
            user_message="Where is my order?",
            conversation_history=sample_history,
        )

        smart_tokens = result.smart_prompt.token_estimate
        full_tokens = result.smart_prompt.full_prompt_estimate
        assert full_tokens > smart_tokens

    @pytest.mark.asyncio
    async def test_pipeline_fallback_intent(self, sample_bot_config):
        fallback_prediction = IntentPrediction(
            intent_name="fallback",
            confidence=0.3,
            method=PredictionMethod.FALLBACK,
        )
        mock_llm = AsyncMock(return_value="I'm not sure how to help with that.")
        pipeline = _make_pipeline(mock_llm)

        result = await pipeline.run(
            bot_config=sample_bot_config,
            user_message="xyzabc completely unrelated gibberish",
            conversation_history=[],
            cached_intent=fallback_prediction,
        )

        assert result.response == "I'm not sure how to help with that."
        assert result.latency_ms["intent_prediction_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_pipeline_calls_llm_with_correct_model(self, sample_bot_config):
        mock_llm = AsyncMock(return_value="response")
        pipeline = _make_pipeline(mock_llm)

        await pipeline.run(
            bot_config=sample_bot_config,
            user_message="Where is my order?",
            conversation_history=[],
        )

        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs["model"] == sample_bot_config.generation_model

    @pytest.mark.asyncio
    async def test_pipeline_with_cached_intent(self, sample_bot_config):
        """When a cached intent is provided, skip intent prediction."""
        cached = IntentPrediction(
            intent_name="check_order",
            confidence=0.95,
            method=PredictionMethod.KEYWORD,
        )
        mock_llm = AsyncMock(return_value="Cached intent response")
        pipeline = _make_pipeline(mock_llm)

        result = await pipeline.run(
            bot_config=sample_bot_config,
            user_message="Where is my order?",
            conversation_history=[],
            cached_intent=cached,
        )

        assert result.intent == cached
        assert result.latency_ms["intent_prediction_ms"] == 0.0
        assert result.response == "Cached intent response"
