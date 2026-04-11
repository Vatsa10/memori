import pytest
from memory_system.core.models import AssembledContext, ConversationTurn, BotConfig, IntentDefinition
from memory_system.core.prompt_builder import (
    build_smart_prompt,
    build_full_prompt_estimate,
    smart_prompt_to_messages,
    estimate_tokens,
)


@pytest.fixture
def assembled_context():
    return AssembledContext(
        instructions="You are a helpful bot.\n\n## Task Instructions\nAsk for order ID.",
        history=[
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ],
        retrieved_context="Order ORD-123 shipped Jan 5",
        example="User: Where is my order?\nAssistant: Please share your order ID.",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "check_order",
                    "description": "Tool: check_order",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )


class TestPromptBuilder:
    def test_build_smart_prompt(self, assembled_context):
        prompt = build_smart_prompt(assembled_context)

        assert "helpful bot" in prompt.system_message
        assert "Relevant Information" in prompt.system_message
        assert "ORD-123" in prompt.system_message
        assert "Example" in prompt.system_message
        assert len(prompt.history) == 2
        assert len(prompt.tools) == 1
        assert prompt.token_estimate > 0

    def test_token_estimate_reasonable(self, assembled_context):
        prompt = build_smart_prompt(assembled_context)
        # Should be a reasonable estimate (not zero, not absurdly high)
        assert 10 < prompt.token_estimate < 500

    def test_messages_format(self, assembled_context):
        prompt = build_smart_prompt(assembled_context)
        messages = smart_prompt_to_messages(prompt)

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hello!"

    def test_full_prompt_larger_than_smart(self, sample_bot_config, sample_history):
        context = AssembledContext(
            instructions="You are a helpful bot.\n\n## Task\nCheck order.",
            history=sample_history[-2:],  # Only 2 turns
            tools=[],
        )
        smart = build_smart_prompt(context)
        full_estimate = build_full_prompt_estimate(sample_bot_config, sample_history)

        # Full prompt should always be larger than smart prompt
        assert full_estimate > smart.token_estimate

    def test_prompt_size_reduction(self, sample_bot_config, sample_history):
        """The core value prop: smart prompt should be significantly smaller."""
        context = AssembledContext(
            instructions="You are a helpful bot.\n\n## Task\nCheck order status.",
            history=sample_history[-2:],
            tools=[{"type": "function", "function": {"name": "check_order", "description": "x", "parameters": {}}}],
        )
        smart = build_smart_prompt(context)
        full_estimate = build_full_prompt_estimate(sample_bot_config, sample_history)

        reduction = (full_estimate - smart.token_estimate) / full_estimate * 100
        # Should achieve at least 40% reduction even with small configs
        assert reduction > 40, f"Only {reduction:.1f}% reduction"

    def test_no_context_no_example(self):
        """When no context or example, system message is just instructions."""
        context = AssembledContext(
            instructions="Be helpful.",
            history=[],
            tools=[],
        )
        prompt = build_smart_prompt(context)
        assert prompt.system_message == "Be helpful."
        assert "Relevant Information" not in prompt.system_message
        assert "Example" not in prompt.system_message
