"""Drop-in memory_system-aware replacement for litellm.acompletion."""

from pathlib import Path
from typing import Any


async def memory_system_completion(
    model: str,
    messages: list[dict],
    bot_config: Any,
    session_id: str = "default",
    **kwargs,
) -> str:
    """
    Drop-in replacement for litellm.acompletion with intent-aware context.

    Usage:
        from memory_system.integrations.litellm_wrapper import memory_system_completion

        response = await memory_system_completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Where is my order?"}],
            bot_config="my_bot.yaml",
        )
    """
    from memory_system._client import MemorySystem
    from memory_system.core.models import BotConfig

    # Resolve bot config
    if isinstance(bot_config, (str, Path)):
        ctx = MemorySystem.from_yaml(bot_config)
    elif isinstance(bot_config, BotConfig):
        ctx = MemorySystem.from_config(bot_config)
    elif isinstance(bot_config, dict):
        ctx = MemorySystem.from_config(BotConfig(**bot_config))
    else:
        raise TypeError(
            f"bot_config must be a path, BotConfig, or dict, got {type(bot_config)}"
        )

    # Override generation model if specified
    if ctx.bot_config:
        ctx.bot_config.generation_model = model
    else:
        raise ValueError("bot_config must be provided to use memory_system_completion")

    # Extract last user message
    user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    if not user_msg:
        raise ValueError("No user message found in messages list")

    result = await ctx.chat(user_msg, session_id=session_id)
    return result.response
