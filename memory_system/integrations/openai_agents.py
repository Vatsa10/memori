"""OpenAI Agents SDK integration — use Memory as a tool for OpenAI agents."""

from memory_system.integrations.base import MemoryAdapter
from memory_system.memory.memory import Memory


class OpenAIAgentsMemory(MemoryAdapter):
    """
    Use as a memory tool for OpenAI Agents SDK.

    Usage:
        from memory_system import StandaloneMemory
        from memory_system.integrations.openai_agents import OpenAIAgentsMemory
        from memory_system.providers.in_memory_stores import InMemoryMemoryStore

        memory = StandaloneMemory(store=InMemoryMemoryStore())
        agent_memory = OpenAIAgentsMemory(memory, user_id="user1")

        # Get context for prompt augmentation
        context = await agent_memory.get_context("user preferences")

        # Remember conversation
        await agent_memory.save_context([
            {"role": "user", "content": "I prefer dark mode"},
            {"role": "assistant", "content": "Noted!"},
        ])
    """

    def __init__(self, memory: Memory, user_id: str = "openai_default"):
        super().__init__(memory, user_id)

    def get_tool_definitions(self) -> list[dict]:
        """Return OpenAI-compatible tool definitions for memory operations."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search user's long-term memory for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Save an important fact about the user to long-term memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact": {"type": "string", "description": "The fact to remember"},
                        },
                        "required": ["fact"],
                    },
                },
            },
        ]

    async def handle_tool_call(self, name: str, arguments: dict) -> str:
        """Handle a tool call from the OpenAI agent."""
        if name == "search_memory":
            results = await self.memory.search(
                arguments["query"], user_id=self.user_id, k=5
            )
            if not results:
                return "No relevant memories found."
            return self.memory.format_memories(results)
        elif name == "save_memory":
            await self.memory.add(arguments["fact"], user_id=self.user_id)
            return f"Saved: {arguments['fact']}"
        return f"Unknown tool: {name}"
