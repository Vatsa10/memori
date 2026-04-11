import pytest
import importlib.util

skip_no_sklearn = pytest.mark.skipif(
    not importlib.util.find_spec("sklearn"),
    reason="scikit-learn not installed",
)


SAMPLE_MESSAGES = [
    # Order-related
    "Where is my order?",
    "Can you track my package?",
    "When will my order arrive?",
    "I need tracking information",
    "What's the status of my delivery?",
    "My package hasn't arrived yet",
    # Return-related
    "I want to return this product",
    "How do I get a refund?",
    "Can I send this back?",
    "I need to return my purchase",
    "This item is defective, I want money back",
    "How to return an item?",
    # Product questions
    "What are the specs of this laptop?",
    "How much does the headphone cost?",
    "Is this available in blue?",
    "Tell me about the features",
    "What's the price of the pro model?",
    "Do you have this in stock?",
]


@skip_no_sklearn
class TestIntentDiscovery:
    def test_discover_basic(self):
        from smartcontext.discovery.auto_intent import IntentDiscovery

        discovery = IntentDiscovery()
        result = discovery.discover(SAMPLE_MESSAGES, n_clusters=3)

        assert len(result.intents) >= 2  # At least 2 meaningful clusters
        assert result.silhouette_score is not None
        assert result.silhouette_score > 0

    def test_discover_auto_k(self):
        from smartcontext.discovery.auto_intent import IntentDiscovery

        discovery = IntentDiscovery()
        result = discovery.discover(SAMPLE_MESSAGES)

        assert len(result.intents) >= 2
        for intent in result.intents:
            assert intent.name
            assert intent.keywords
            assert intent.cluster_size >= 1

    def test_to_yaml(self):
        from smartcontext.discovery.auto_intent import IntentDiscovery

        discovery = IntentDiscovery()
        result = discovery.discover(SAMPLE_MESSAGES, n_clusters=3)
        yaml_str = discovery.to_yaml(result, "test_bot", "TestBot")

        assert "bot_id: test_bot" in yaml_str
        assert "bot_name: TestBot" in yaml_str
        assert "intents:" in yaml_str

    def test_too_few_messages(self):
        from smartcontext.discovery.auto_intent import IntentDiscovery

        discovery = IntentDiscovery()
        result = discovery.discover(["hello", "hi"])

        assert len(result.intents) == 0
        assert len(result.unclustered_messages) == 2

    def test_discovered_intents_have_samples(self):
        from smartcontext.discovery.auto_intent import IntentDiscovery

        discovery = IntentDiscovery()
        result = discovery.discover(SAMPLE_MESSAGES, n_clusters=3)

        for intent in result.intents:
            assert len(intent.sample_messages) > 0
            assert intent.description
            assert intent.instructions
