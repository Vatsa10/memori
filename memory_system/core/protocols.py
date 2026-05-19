from datetime import datetime
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
        include_invalidated: bool = False,
    ) -> list[MemorySearchResult]: ...

    async def update(self, memory_id: str, text: str) -> None: ...

    async def delete(self, memory_id: str) -> None: ...

    async def get_all(
        self,
        user_id: str,
        k: int = 50,
        include_invalidated: bool = False,
    ) -> list[MemorySearchResult]: ...

    async def invalidate(
        self,
        memory_id: str,
        valid_to: datetime,
        superseded_by: Optional[str] = None,
    ) -> None: ...

    async def search_at(
        self,
        query: str,
        user_id: str,
        as_of: datetime,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MemorySearchResult]: ...


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

    async def traverse(
        self,
        start_entity: str,
        user_id: str,
        max_hops: int = 2,
        relation_filter: Optional[list[str]] = None,
        max_results: int = 20,
    ) -> list[list[Relationship]]: ...


@runtime_checkable
class KnowledgeSearcher(Protocol):
    """Simple search for knowledge bases (backward compatible)."""

    async def search(self, query: str, k: int = 2) -> list[str]: ...
