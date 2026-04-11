from memory_system.core.models import AssembledContext, BotConfig, ConversationTurn, SmartPrompt


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word."""
    return int(len(text.split()) * 1.3)


def build_smart_prompt(context: AssembledContext) -> SmartPrompt:
    """Build a minimal prompt from assembled context."""

    # Build system message
    system_parts = [context.instructions]

    if context.retrieved_context:
        system_parts.append(f"\n## Relevant Information\n{context.retrieved_context}")

    if context.example:
        system_parts.append(f"\n## Example\n{context.example}")

    system_message = "\n".join(system_parts)

    # Estimate token count for this smart prompt
    token_count = estimate_tokens(system_message)
    for turn in context.history:
        token_count += estimate_tokens(turn.content)
    for tool in context.tools:
        token_count += estimate_tokens(str(tool))

    return SmartPrompt(
        system_message=system_message,
        history=context.history,
        tools=context.tools,
        token_estimate=token_count,
    )


def build_full_prompt_estimate(
    bot_config: BotConfig,
    full_history: list[ConversationTurn],
) -> int:
    """Estimate what the token count WOULD be with full context injection.
    Used for comparison/metrics only."""

    # Full system: base + ALL intent instructions + ALL examples
    full_system = bot_config.base_instructions
    for intent in bot_config.intents:
        full_system += f"\n\n## {intent.name}\n{intent.instructions}"
        if intent.example:
            full_system += f"\n### Example\n{intent.example}"
    if bot_config.fallback_instructions:
        full_system += f"\n\n## Fallback\n{bot_config.fallback_instructions}"

    tokens = estimate_tokens(full_system)

    # Full history (all turns)
    for turn in full_history:
        tokens += estimate_tokens(turn.content)

    # All tools from all intents
    all_tools = set()
    for intent in bot_config.intents:
        all_tools.update(intent.tools)
    tokens += len(all_tools) * 50  # ~50 tokens per tool schema

    return tokens


def smart_prompt_to_messages(prompt: SmartPrompt) -> list[dict]:
    """Convert SmartPrompt to LLM messages array."""
    messages = [{"role": "system", "content": prompt.system_message}]

    for turn in prompt.history:
        messages.append({"role": turn.role, "content": turn.content})

    return messages
