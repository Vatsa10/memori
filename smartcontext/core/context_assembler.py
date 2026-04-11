from typing import Optional, Protocol, runtime_checkable

from smartcontext.core.models import (
    AssembledContext,
    BotConfig,
    ConversationTurn,
    IntentDefinition,
    IntentPrediction,
)


@runtime_checkable
class MemorySearcher(Protocol):
    async def search(self, query: str, k: int = 2) -> list[str]: ...


class ContextAssembler:
    def __init__(self, memory_provider: Optional[MemorySearcher] = None):
        self.memory = memory_provider

    async def assemble(
        self,
        bot_config: BotConfig,
        intent: IntentPrediction,
        user_message: str,
        full_history: list[ConversationTurn],
    ) -> AssembledContext:
        # Find the matched intent definition
        matched_intent = self._find_intent(bot_config, intent.intent_name)

        # Build instructions
        instructions = self._build_instructions(bot_config, matched_intent, intent)

        # Trim history
        history = self._trim_history(full_history, matched_intent, intent, bot_config)

        # Retrieve relevant context from memory/vector DB
        retrieved_context = await self._retrieve_context(
            user_message, matched_intent, intent
        )

        # Get example
        example = matched_intent.example if matched_intent else None

        # Filter tools
        tools = self._get_tools(matched_intent)

        return AssembledContext(
            instructions=instructions,
            history=history,
            retrieved_context=retrieved_context,
            example=example,
            tools=tools,
        )

    def _find_intent(
        self, bot_config: BotConfig, intent_name: str
    ) -> Optional[IntentDefinition]:
        for intent_def in bot_config.intents:
            if intent_def.name == intent_name:
                return intent_def
        return None

    def _build_instructions(
        self,
        bot_config: BotConfig,
        matched_intent: Optional[IntentDefinition],
        prediction: IntentPrediction,
    ) -> str:
        parts = [bot_config.base_instructions.strip()]

        if matched_intent and prediction.confidence >= bot_config.confidence_threshold:
            parts.append(f"\n## Task Instructions\n{matched_intent.instructions.strip()}")
        else:
            # Low confidence or no match — use fallback with broader instructions
            if bot_config.fallback_instructions:
                parts.append(f"\n## Instructions\n{bot_config.fallback_instructions.strip()}")

        return "\n".join(parts)

    def _trim_history(
        self,
        full_history: list[ConversationTurn],
        matched_intent: Optional[IntentDefinition],
        prediction: IntentPrediction,
        bot_config: BotConfig,
    ) -> list[ConversationTurn]:
        if prediction.confidence < bot_config.confidence_threshold:
            # Low confidence — include more history for context
            max_turns = 4
        elif matched_intent:
            max_turns = matched_intent.max_history_turns
        else:
            max_turns = 2

        # Each "turn" is a user+assistant pair, so multiply by 2 for individual messages
        max_messages = max_turns * 2
        return full_history[-max_messages:] if full_history else []

    async def _retrieve_context(
        self,
        user_message: str,
        matched_intent: Optional[IntentDefinition],
        prediction: IntentPrediction,
    ) -> Optional[str]:
        if not self.memory:
            return None

        # Build intent-augmented query
        if matched_intent and matched_intent.retrieval_query_prefix:
            query = f"{matched_intent.retrieval_query_prefix}: {user_message}"
        elif matched_intent:
            query = f"{matched_intent.name}: {user_message}"
        else:
            query = user_message

        results = await self.memory.search(query, k=2)
        if results:
            return "\n---\n".join(results)
        return None

    def _get_tools(self, matched_intent: Optional[IntentDefinition]) -> list[dict]:
        """Return tool schemas filtered to this intent's tools.
        For now, returns tool name stubs. In production, map to full OpenAI tool schemas."""
        if not matched_intent or not matched_intent.tools:
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Tool: {tool_name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for tool_name in matched_intent.tools
        ]
