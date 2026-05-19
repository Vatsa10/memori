import pytest
from memory_system.core.models import BotConfig, IntentDefinition, ConversationTurn

# Script-style runner (uses asyncio.run via __main__); not a proper pytest module.
collect_ignore = ["test_integration.py"]


@pytest.fixture
def sample_bot_config() -> BotConfig:
    return BotConfig(
        bot_id="test_bot",
        bot_name="TestBot",
        base_instructions="You are a helpful test bot. Be concise.",
        intents=[
            IntentDefinition(
                name="check_order",
                description="Customer wants to check order status",
                keywords=["order status", "where is my order", "tracking"],
                instructions="Ask for order ID. Look up status. Report delivery estimate.",
                example="User: Where is my order?\nAssistant: Please share your order ID.",
                tools=["check_order"],
                max_history_turns=2,
            ),
            IntentDefinition(
                name="return_item",
                description="Customer wants to return or refund an item",
                keywords=["return", "refund", "send back", "money back"],
                instructions="Verify order ID. Check eligibility. Initiate return if eligible.",
                example="User: I want to return this\nAssistant: I can help. What's your order ID?",
                tools=["check_order", "create_return"],
                max_history_turns=3,
            ),
            IntentDefinition(
                name="product_info",
                description="Customer asking about product details or pricing",
                keywords=["price", "specs", "features", "tell me about"],
                instructions="Search catalog. Provide key specs and pricing.",
                tools=["search_products"],
                max_history_turns=1,
            ),
        ],
        fallback_instructions="Try to be helpful. Offer to connect with a human.",
        confidence_threshold=0.6,
        keyword_threshold=0.4,
        embedding_threshold=0.7,
    )


@pytest.fixture
def sample_history() -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content="Hi there"),
        ConversationTurn(role="assistant", content="Hello! How can I help you?"),
        ConversationTurn(role="user", content="I need to check something"),
        ConversationTurn(role="assistant", content="Sure, what would you like to check?"),
    ]
