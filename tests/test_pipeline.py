import pytest
from unittest.mock import AsyncMock, patch

from smartcontext.core.pipeline import Pipeline
from smartcontext.core.intent_predictor import IntentPredictor
from smartcontext.core.models import IntentPrediction, PredictionMethod
from smartcontext.providers.memory import InMemoryProvider


@pytest.fixture
def pipeline():
    predictor = IntentPredictor()
    memory = InMemoryProvider()
    memory.add("Order ORD-123 shipped on Jan 5")
    return Pipeline(intent_predictor=predictor, memory_provider=memory)


def _mock_llm_calls():
    """Mock both LLM calls: generation and intent prediction fallback."""
    return (
        patch("smartcontext.core.pipeline.call_llm", new_callable=AsyncMock),
        patch("smartcontext.core.pipeline.predict_intent_llm", new_callable=AsyncMock),
    )


class TestPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocked_llm(self, pipeline, sample_bot_config):
        mock_response = "Your order ORD-123 shipped on Jan 5 and is on its way!"

        gen_mock, intent_mock = _mock_llm_calls()
        with gen_mock as mock_llm, intent_mock:
            mock_llm.return_value = mock_response

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
    async def test_pipeline_token_reduction(self, pipeline, sample_bot_config, sample_history):
        gen_mock, intent_mock = _mock_llm_calls()
        with gen_mock as mock_llm, intent_mock:
            mock_llm.return_value = "Here's your order info."

            result = await pipeline.run(
                bot_config=sample_bot_config,
                user_message="Where is my order?",
                conversation_history=sample_history,
            )

            smart_tokens = result.smart_prompt.token_estimate
            full_tokens = result.smart_prompt.full_prompt_estimate
            assert full_tokens > smart_tokens

    @pytest.mark.asyncio
    async def test_pipeline_fallback_intent(self, pipeline, sample_bot_config):
        fallback_prediction = IntentPrediction(
            intent_name="fallback",
            confidence=0.3,
            method=PredictionMethod.FALLBACK,
        )
        gen_mock, intent_mock = _mock_llm_calls()
        with gen_mock as mock_llm, intent_mock as mock_intent:
            mock_llm.return_value = "I'm not sure how to help with that."
            mock_intent.return_value = fallback_prediction

            result = await pipeline.run(
                bot_config=sample_bot_config,
                user_message="xyzabc completely unrelated gibberish",
                conversation_history=[],
            )

            assert result.response == "I'm not sure how to help with that."

    @pytest.mark.asyncio
    async def test_pipeline_calls_llm_with_correct_model(
        self, pipeline, sample_bot_config
    ):
        gen_mock, intent_mock = _mock_llm_calls()
        with gen_mock as mock_llm, intent_mock:
            mock_llm.return_value = "response"

            await pipeline.run(
                bot_config=sample_bot_config,
                user_message="Where is my order?",
                conversation_history=[],
            )

            mock_llm.assert_called_once()
            call_kwargs = mock_llm.call_args
            assert call_kwargs.kwargs["model"] == sample_bot_config.generation_model
