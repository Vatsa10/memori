#!/usr/bin/env python3
"""Quick integration test for MemorySystem factory methods."""

import asyncio
import os
from memory_system import MemorySystem
from memory_system.config.factory import create_providers
from memory_system.core.models import BotConfig, IntentDefinition


async def test_basic_memory_system():
    """Test basic MemorySystem creation and usage."""
    print("Testing basic MemorySystem...")

    # Basic instantiation
    ms = MemorySystem(
        instructions="You are a helpful assistant.",
        llm_fn=lambda model, messages: asyncio.sleep(0.01, result="Test response"),
    )

    result = await ms.chat("Hello", user_id="test_user")
    print(f"Basic chat response: {result.response}")
    assert result.response == "Test response"
    print("✓ Basic MemorySystem works")


async def test_from_config():
    """Test MemorySystem.from_config with BotConfig."""
    print("\nTesting MemorySystem.from_config...")

    # Create a simple BotConfig
    config = BotConfig(
        bot_id="test_bot",
        bot_name="Test Bot",
        base_instructions="You are a helpful test bot.",
        intents=[
            IntentDefinition(
                name="greeting",
                description="Simple greeting",
                keywords=["hello", "hi"],
                instructions="Respond warmly to greetings.",
                tools=[],
                max_history_turns=1,
            )
        ],
    )

    # Test from_config
    ms = MemorySystem.from_config(
        config,
        llm_fn=lambda model, messages: asyncio.sleep(0.01, result="Greeting response"),
    )

    result = await ms.chat("Hello there", user_id="test_user")
    print(f"Intent-aware chat response: {result.response}")
    assert result.response == "Greeting response"
    assert result.intent is not None
    assert result.intent.intent_name == "greeting"
    print("✓ MemorySystem.from_config works")


async def test_from_yaml():
    """Test MemorySystem.from_yaml with YAML file."""
    print("\nTesting MemorySystem.from_yaml...")

    # Test from_yaml
    ms = MemorySystem.from_yaml(
        "configs/test_bot.yaml",
        llm_fn=lambda model, messages: asyncio.sleep(0.01, result="YAML response"),
    )

    result = await ms.chat("Hi", user_id="test_user")
    print(f"YAML-based chat response: {result.response}")
    assert result.response == "YAML response"
    assert result.intent is not None
    assert result.intent.intent_name == "check_order"  # from test_bot.yaml
    print("✓ MemorySystem.from_yaml works")


async def test_from_env_defaults():
    """Test MemorySystem.from_env with defaults."""
    print("\nTesting MemorySystem.from_env...")

    # Clear any BOT_CONFIG_PATH env var
    if "BOT_CONFIG_PATH" in os.environ:
        del os.environ["BOT_CONFIG_PATH"]

    ms = MemorySystem.from_env(
        llm_fn=lambda model, messages: asyncio.sleep(0.01, result="Env response"),
    )

    result = await ms.chat("Hello from env", user_id="test_user")
    print(f"Env-based chat response: {result.response}")
    assert result.response == "Env response"
    # Should be basic mode since no BOT_CONFIG_PATH
    print("✓ MemorySystem.from_env works")


async def test_provider_factory():
    """Test the provider factory directly."""
    print("\nTesting provider factory...")

    providers = create_providers()
    assert "memory_store" in providers
    assert "graph_store" in providers
    assert "session_store" in providers
    assert "cache" in providers
    print(f"Created providers: {list(providers.keys())}")
    print("✓ Provider factory works")


async def main():
    """Run all integration tests."""
    print("Running MemorySystem integration tests...\n")

    await test_basic_memory_system()
    await test_from_config()
    await test_from_yaml()
    await test_from_env_defaults()
    await test_provider_factory()

    print("\n🎉 All integration tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
