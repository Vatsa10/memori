import pytest
from app.core.intent_predictor import IntentPredictor
from app.core.models import PredictionMethod


@pytest.fixture
def predictor():
    return IntentPredictor()


class TestKeywordMatching:
    @pytest.mark.asyncio
    async def test_exact_keyword_match(self, predictor, sample_bot_config):
        prediction, elapsed = await predictor.predict(
            bot_config=sample_bot_config,
            user_message="Where is my order?",
            recent_history=[],
        )
        assert prediction.intent_name == "check_order"
        assert prediction.method == PredictionMethod.KEYWORD
        assert prediction.confidence > 0

    @pytest.mark.asyncio
    async def test_return_keyword_match(self, predictor, sample_bot_config):
        prediction, elapsed = await predictor.predict(
            bot_config=sample_bot_config,
            user_message="I want to return this item and get a refund",
            recent_history=[],
        )
        assert prediction.intent_name == "return_item"
        assert prediction.method == PredictionMethod.KEYWORD

    @pytest.mark.asyncio
    async def test_product_keyword_match(self, predictor, sample_bot_config):
        prediction, elapsed = await predictor.predict(
            bot_config=sample_bot_config,
            user_message="What's the price of the headphones?",
            recent_history=[],
        )
        assert prediction.intent_name == "product_info"

    @pytest.mark.asyncio
    async def test_no_keyword_match_falls_through(self, predictor, sample_bot_config):
        """When no keywords match and no LLM provided, should return fallback."""
        prediction, elapsed = await predictor.predict(
            bot_config=sample_bot_config,
            user_message="xyzabc random gibberish",
            recent_history=[],
        )
        # Without LLM, should fall back
        assert prediction.intent_name == "fallback" or prediction.method in (
            PredictionMethod.EMBEDDING,
            PredictionMethod.FALLBACK,
        )

    @pytest.mark.asyncio
    async def test_keyword_match_latency(self, predictor, sample_bot_config):
        """Keyword matching should be near-instant (< 5ms)."""
        _, elapsed = await predictor.predict(
            bot_config=sample_bot_config,
            user_message="I want a refund",
            recent_history=[],
        )
        # Should be very fast when keyword matches
        assert elapsed < 50  # generous threshold for CI


class TestEmbeddingMatching:
    @pytest.mark.asyncio
    async def test_embedding_precompute(self, sample_bot_config):
        predictor = IntentPredictor()
        predictor.precompute_intent_embeddings(sample_bot_config)
        assert sample_bot_config.bot_id in predictor._intent_embeddings_cache
        assert len(predictor._intent_embeddings_cache[sample_bot_config.bot_id]) == 3

    @pytest.mark.asyncio
    async def test_semantic_match(self, sample_bot_config):
        """Embedding should catch semantically similar but keyword-different messages."""
        predictor = IntentPredictor()
        predictor.precompute_intent_embeddings(sample_bot_config)

        # Set keyword threshold very high so it doesn't match on keywords
        sample_bot_config.keyword_threshold = 1.0

        prediction, _ = await predictor.predict(
            bot_config=sample_bot_config,
            user_message="Can I send this product back to you?",
            recent_history=[],
        )
        # "send back" is semantically close to "return" intent
        assert prediction.intent_name == "return_item"
        assert prediction.method in (PredictionMethod.EMBEDDING, PredictionMethod.FALLBACK)
