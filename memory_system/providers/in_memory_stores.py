"""In-memory implementations of MemoryStore and GraphStore for testing."""

import re
from datetime import datetime, timezone
from typing import Optional

from memory_system.core.memory_models import (
    Entity,
    Memory,
    MemorySearchResult,
    Relationship,
)


class InMemoryMemoryStore:
    """In-memory vector-like store. Uses keyword overlap for search (no embeddings)."""

    def __init__(self):
        self._memories: dict[str, Memory] = {}

    async def add(self, memory: Memory) -> str:
        self._memories[memory.id] = memory
        return memory.id

    async def search(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MemorySearchResult]:
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []

        for mem in self._memories.values():
            if mem.user_id != user_id:
                continue
            if filters:
                skip = False
                for fk, fv in filters.items():
                    if mem.metadata.get(fk) != fv:
                        skip = True
                        break
                if skip:
                    continue

            mem_words = set(re.findall(r"\w+", mem.text.lower()))
            overlap = len(query_words & mem_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append(MemorySearchResult(memory=mem, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    async def update(self, memory_id: str, text: str) -> None:
        if memory_id in self._memories:
            mem = self._memories[memory_id]
            mem.text = text
            mem.updated_at = datetime.now(timezone.utc)

    async def delete(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)

    async def get_all(self, user_id: str, k: int = 50) -> list[MemorySearchResult]:
        results = []
        for mem in self._memories.values():
            if mem.user_id == user_id:
                results.append(MemorySearchResult(memory=mem, score=1.0))
        return results[:k]


class InMemoryGraphStore:
    """In-memory graph store for testing entity-relationship queries."""

    def __init__(self):
        self._entities: list[Entity] = []
        self._relationships: list[Relationship] = []

    async def add_entity(self, entity: Entity) -> None:
        # Deduplicate by name + user_id
        for existing in self._entities:
            if existing.name == entity.name and existing.user_id == entity.user_id:
                existing.properties.update(entity.properties)
                return
        self._entities.append(entity)

    async def add_relationship(self, relationship: Relationship) -> None:
        self._relationships.append(relationship)

    async def search_entities(
        self, query: str, user_id: str, k: int = 5
    ) -> list[Entity]:
        query_lower = query.lower()
        results = []
        for entity in self._entities:
            if entity.user_id != user_id:
                continue
            if query_lower in entity.name.lower() or query_lower in entity.entity_type.lower():
                results.append(entity)
        return results[:k]

    async def get_related(
        self,
        entity_name: str,
        user_id: str,
        relation_type: Optional[str] = None,
    ) -> list[Relationship]:
        results = []
        for rel in self._relationships:
            if rel.user_id != user_id:
                continue
            if rel.source_entity == entity_name or rel.target_entity == entity_name:
                if relation_type is None or rel.relation_type == relation_type:
                    results.append(rel)
        return results
