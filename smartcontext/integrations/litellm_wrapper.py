"""Drop-in smartcontext-aware replacement for litellm.acompletion."""

from pathlib import Path
from typing import Any


async def smartcontext_completion(
    model: str,
    messages: list[dict],
    bot_config: Any,
    session_id: str = "default",
    **kwargs,
) -> str:
    """
    Drop-in replacement for litellm.acompletion with intent-aware context.

    Usage:
        from smartcontext.integrations.litellm_wrapper import smartcontext_completion

        response = await smartcontext_completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Where is my order?"}],
            bot_config="my_bot.yaml",
        )
    """
    from smartcontext._client import SmartContext
    from smartcontext.core.models import BotConfig

    # Resolve bot config
    if isinstance(bot_config, (str, Path)):
        ctx = SmartContext.from_yaml(bot_config)
    elif isinstance(bot_config, BotConfig):
        ctx = SmartContext(bot_config)
    elif isinstance(bot_config, dict):
        ctx = SmartContext.from_dict(bot_config)
    else:
        raise TypeError(f"bot_config must be a path, BotConfig, or dict, got {type(bot_config)}")

    # Override generation model if specified
    ctx.config.generation_model = model

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
