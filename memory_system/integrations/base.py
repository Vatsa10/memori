"""Base adapter for framework integrations."""

from typing import Optional

from memory_system.memory.memory import Memory


class MemoryAdapter:
    """Base class for framework-specific memory adapters."""

    def __init__(self, memory: Memory, user_id: str = "default"):
        self.memory = memory
        self.user_id = user_id

    async def get_context(self, query: str, k: int = 5) -> str:
        results = await self.memory.recall(query, user_id=self.user_id, k=k)
        return self.memory.format_memories(results)

    async def save_context(self, messages: list[dict]) -> None:
        await self.memory.remember(messages, user_id=self.user_id)

    async def clear(self) -> None:
        await self.memory.forget(user_id=self.user_id)
