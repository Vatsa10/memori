"""MemoryManager: recall before LLM, remember after LLM."""

import re
from typing import Callable, Optional

from memory_system.core.memory_models import (
    Memory,
    MemoryExtractionResult,
    MemorySearchResult,
    MemoryType,
)
from memory_system.core.protocols import GraphStore, MemoryStore
from memory_system.memory.extractor import extract_memories


class MemoryManager:
    """
    Manages the full memory lifecycle:
    - recall(): Pre-LLM retrieval of relevant user memories
    - remember(): Post-LLM extraction and storage of new facts
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        graph_store: Optional[GraphStore] = None,
        extraction_llm_fn: Optional[Callable] = None,
        extraction_model: str = "groq/llama-3.1-8b-instant",
        dedup_threshold: float = 0.85,
    ):
        self.memory_store = memory_store
        self.graph_store = graph_store
        self.extraction_llm_fn = extraction_llm_fn
        self.extraction_model = extraction_model
        self.dedup_threshold = dedup_threshold

    async def recall(
        self,
        query: str,
        user_id: str,
        intent_name: Optional[str] = None,
        k: int = 5,
    ) -> list[MemorySearchResult]:
        """Pre-LLM: Retrieve relevant memories for this user + intent."""
        if not self.memory_store:
            return []

        # Build intent-augmented query
        search_query = f"{intent_name}: {query}" if intent_name else query

        # Search vector store
        filters = {}
        if intent_name:
            filters["intent"] = intent_name

        vector_results = await self.memory_store.search(
            query=search_query, user_id=user_id, k=k, filters=filters
        )

        # If filters returned too few results, retry without intent filter
        if len(vector_results) < 2 and intent_name:
            vector_results = await self.memory_store.search(
                query=query, user_id=user_id, k=k
            )

        # Also query graph store for related entities
        if self.graph_store:
            entities = await self.graph_store.search_entities(
                query=query, user_id=user_id, k=3
            )
            for entity in entities:
                relationships = await self.graph_store.get_related(
                    entity_name=entity.name, user_id=user_id
                )
                for rel in relationships:
                    text = f"{rel.source_entity} {rel.relation_type} {rel.target_entity}"
                    # Add as a graph-sourced memory result
                    graph_mem = Memory(
                        text=text,
                        memory_type=MemoryType.SEMANTIC,
                        user_id=user_id,
                        source="graph",
                    )
                    vector_results.append(
                        MemorySearchResult(memory=graph_mem, score=0.7, source="graph")
                    )

        # Deduplicate by text similarity
        seen_texts = set()
        unique_results = []
        for r in vector_results:
            normalized = r.memory.text.lower().strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_results.append(r)

        return unique_results[:k]

    async def remember(
        self,
        user_message: str,
        assistant_response: str,
        user_id: str,
        session_id: str = "default",
    ) -> MemoryExtractionResult:
        """Post-LLM: Extract facts and store in memory + graph."""

        extraction = await extract_memories(
            user_message=user_message,
            assistant_response=assistant_response,
            user_id=user_id,
            llm_fn=self.extraction_llm_fn,
            model=self.extraction_model,
        )

        if not extraction.memories and not extraction.entities:
            return extraction

        # Store memories (with dedup)
        stored_memories = []
        if self.memory_store:
            for memory in extraction.memories:
                is_dup = await self._is_duplicate(memory)
                if not is_dup:
                    memory.metadata["session_id"] = session_id
                    await self.memory_store.add(memory)
                    stored_memories.append(memory)

        # Store entities and relationships in graph
        if self.graph_store:
            for entity in extraction.entities:
                await self.graph_store.add_entity(entity)
            for relationship in extraction.relationships:
                await self.graph_store.add_relationship(relationship)

        return MemoryExtractionResult(
            memories=stored_memories,
            entities=extraction.entities,
            relationships=extraction.relationships,
        )

    async def add_memory(
        self,
        text: str,
        user_id: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """Manually add a memory."""
        memory = Memory(
            text=text,
            memory_type=memory_type,
            user_id=user_id,
            metadata=metadata or {},
            source="manual",
        )
        if self.memory_store:
            await self.memory_store.add(memory)
        return memory

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory (GDPR compliance)."""
        if self.memory_store:
            await self.memory_store.delete(memory_id)

    async def get_user_memories(
        self, user_id: str, k: int = 50
    ) -> list[MemorySearchResult]:
        """Get all memories for a user."""
        if not self.memory_store:
            return []
        return await self.memory_store.get_all(user_id, k=k)

    async def _is_duplicate(self, memory: Memory) -> bool:
        """Check if a similar memory already exists."""
        if not self.memory_store:
            return False

        existing = await self.memory_store.search(
            query=memory.text, user_id=memory.user_id, k=1
        )
        if existing and existing[0].score >= self.dedup_threshold:
            return True
        return False
