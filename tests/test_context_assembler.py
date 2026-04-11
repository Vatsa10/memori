import pytest
from smartcontext.core.context_assembler import ContextAssembler
from smartcontext.core.models import IntentPrediction, PredictionMethod
from smartcontext.providers.memory import InMemoryProvider


@pytest.fixture
def memory():
    provider = InMemoryProvider()
    provider.add("Order ORD-123 shipped on Jan 5, arriving Jan 8 via FedEx")
    provider.add("Return policy: 30 days from purchase. Items must be unused.")
    provider.add("Headphones Pro X: $99.99, noise cancelling, 20hr battery")
    return provider


@pytest.fixture
def assembler(memory):
    return ContextAssembler(memory_provider=memory)


class TestContextAssembly:
    @pytest.mark.asyncio
    async def test_high_confidence_uses_intent_instructions(
        self, assembler, sample_bot_config
    ):
        intent = IntentPrediction(
            intent_name="check_order",
            confidence=0.9,
            method=PredictionMethod.KEYWORD,
        )
        context = await assembler.assemble(
            bot_config=sample_bot_config,
            intent=intent,
            user_message="Where is order ORD-123?",
            full_history=[],
        )
        assert "Ask for order ID" in context.instructions
        assert "base_instructions" not in context.instructions  # base is included but by value
        assert "helpful test bot" in context.instructions  # base instructions present

    @pytest.mark.asyncio
    async def test_low_confidence_uses_fallback(
        self, assembler, sample_bot_config
    ):
        intent = IntentPrediction(
            intent_name="fallback",
            confidence=0.3,
            method=PredictionMethod.FALLBACK,
        )
        context = await assembler.assemble(
            bot_config=sample_bot_config,
            intent=intent,
            user_message="blah blah",
            full_history=[],
        )
        assert "Try to be helpful" in context.instructions

    @pytest.mark.asyncio
    async def test_history_trimming(
        self, assembler, sample_bot_config, sample_history
    ):
        intent = IntentPrediction(
            intent_name="product_info",
            confidence=0.9,
            method=PredictionMethod.KEYWORD,
        )
        context = await assembler.assemble(
            bot_config=sample_bot_config,
            intent=intent,
            user_message="Tell me about headphones",
            full_history=sample_history,
        )
        # product_info has max_history_turns=1, so 2 messages max
        assert len(context.history) <= 2

    @pytest.mark.asyncio
    async def test_return_intent_gets_more_history(
        self, assembler, sample_bot_config, sample_history
    ):
        intent = IntentPrediction(
            intent_name="return_item",
            confidence=0.9,
            method=PredictionMethod.KEYWORD,
        )
        context = await assembler.assemble(
            bot_config=sample_bot_config,
            intent=intent,
            user_message="I want to return this",
            full_history=sample_history,
        )
        # return_item has max_history_turns=3, so up to 6 messages
        assert len(context.history) <= 6

    @pytest.mark.asyncio
    async def test_tool_filtering(self, assembler, sample_bot_config):
        intent = IntentPrediction(
            intent_name="check_order",
            confidence=0.9,
            method=PredictionMethod.KEYWORD,
        )
        context = await assembler.assemble(
            bot_config=sample_bot_config,
            intent=intent,
            user_message="Where is my order?",
            full_history=[],
        )
        tool_names = [t["function"]["name"] for t in context.tools]
        assert tool_names == ["check_order"]

    @pytest.mark.asyncio
    async def test_memory_retrieval(self, assembler, sample_bot_config):
        intent = IntentPrediction(
            intent_name="check_order",
            confidence=0.9,
            method=PredictionMethod.KEYWORD,
        )
        context = await assembler.assemble(
            bot_config=sample_bot_config,
            intent=intent,
            user_message="Where is order ORD-123?",
            full_history=[],
        )
        assert context.retrieved_context is not None
        assert "ORD-123" in context.retrieved_context

    @pytest.mark.asyncio
    async def test_example_included(self, assembler, sample_bot_config):
        intent = IntentPrediction(
            intent_name="check_order",
            confidence=0.9,
            method=PredictionMethod.KEYWORD,
        )
        context = await assembler.assemble(
            bot_config=sample_bot_config,
            intent=intent,
            user_message="Where is my order?",
            full_history=[],
        )
        assert context.example is not None
        assert "order ID" in context.example
