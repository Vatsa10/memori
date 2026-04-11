from typing import Any, Optional, Protocol, runtime_checkable

from memory_system.core.models import ConversationTurn
from memory_system.core.memory_models import (
    Entity,
    Memory,
    MemorySearchResult,
    Relationship,
)


@runtime_checkable
class MemoryStore(Protocol):
    """Vector store for persistent user memories."""

    async def add(self, memory: Memory) -> str: ...

    async def search(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MemorySearchResult]: ...

    async def update(self, memory_id: str, text: str) -> None: ...

    async def delete(self, memory_id: str) -> None: ...

    async def get_all(self, user_id: str, k: int = 50) -> list[MemorySearchResult]: ...


@runtime_checkable
class GraphStore(Protocol):
    """Entity-relationship graph store."""

    async def add_entity(self, entity: Entity) -> None: ...

    async def add_relationship(self, relationship: Relationship) -> None: ...

    async def search_entities(
        self, query: str, user_id: str, k: int = 5
    ) -> list[Entity]: ...

    async def get_related(
        self,
        entity_name: str,
        user_id: str,
        relation_type: Optional[str] = None,
    ) -> list[Relationship]: ...


@runtime_checkable
class KnowledgeSearcher(Protocol):
    """Simple search for knowledge bases (backward compatible)."""

    async def search(self, query: str, k: int = 2) -> list[str]: ...
