"""CrewAI integration — use Memory as a CrewAI memory backend."""

from memory_system.integrations.base import MemoryAdapter
from memory_system.memory.memory import Memory


class CrewAIMemory(MemoryAdapter):
    """
    Use as a memory backend for CrewAI agents.

    Usage:
        from memory_system import StandaloneMemory
        from memory_system.integrations.crewai import CrewAIMemory
        from memory_system.providers.in_memory_stores import InMemoryMemoryStore

        memory = StandaloneMemory(store=InMemoryMemoryStore())
        crew_memory = CrewAIMemory(memory, user_id="agent1")

        # In your CrewAI task:
        context = await crew_memory.get_context("project requirements")
    """

    def __init__(self, memory: Memory, user_id: str = "crewai_default"):
        super().__init__(memory, user_id)

    async def search(self, query: str, k: int = 5) -> list[dict]:
        """Search memories and return as dicts (CrewAI-compatible format)."""
        results = await self.memory.search(query, user_id=self.user_id, k=k)
        return [
            {
                "content": r.memory.text,
                "score": r.score,
                "type": r.memory.memory_type.value,
                "metadata": r.memory.metadata,
            }
            for r in results
        ]

    async def add(self, content: str, metadata: dict | None = None) -> None:
        """Add a memory from CrewAI context."""
        await self.memory.add(content, user_id=self.user_id, metadata=metadata)
