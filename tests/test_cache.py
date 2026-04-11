import pytest
from smartcontext.cache import IntentCache
from smartcontext.core.models import IntentPrediction, PredictionMethod


@pytest.fixture
def cache():
    return IntentCache(maxsize=3)


@pytest.fixture
def prediction():
    return IntentPrediction(
        intent_name="check_order",
        confidence=0.9,
        method=PredictionMethod.KEYWORD,
    )


class TestIntentCache:
    def test_put_and_get(self, cache, prediction):
        cache.put("bot1", "Where is my order?", prediction)
        result = cache.get("bot1", "Where is my order?")
        assert result is not None
        assert result.intent_name == "check_order"

    def test_case_insensitive(self, cache, prediction):
        cache.put("bot1", "Where Is My Order?", prediction)
        result = cache.get("bot1", "where is my order?")
        assert result is not None

    def test_miss(self, cache):
        result = cache.get("bot1", "unknown message")
        assert result is None

    def test_different_bots(self, cache, prediction):
        cache.put("bot1", "hello", prediction)
        assert cache.get("bot2", "hello") is None
        assert cache.get("bot1", "hello") is not None

    def test_lru_eviction(self, cache, prediction):
        cache.put("bot1", "msg1", prediction)
        cache.put("bot1", "msg2", prediction)
        cache.put("bot1", "msg3", prediction)
        # Cache is full (maxsize=3). Adding one more evicts oldest.
        cache.put("bot1", "msg4", prediction)

        assert cache.get("bot1", "msg1") is None  # Evicted
        assert cache.get("bot1", "msg4") is not None

    def test_hit_rate(self, cache, prediction):
        cache.put("bot1", "hello", prediction)
        cache.get("bot1", "hello")  # hit
        cache.get("bot1", "missing")  # miss

        assert cache.hit_rate == 0.5
        assert cache.hits == 1
        assert cache.misses == 1

    def test_clear(self, cache, prediction):
        cache.put("bot1", "hello", prediction)
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_size(self, cache, prediction):
        assert cache.size == 0
        cache.put("bot1", "a", prediction)
        cache.put("bot1", "b", prediction)
        assert cache.size == 2
