from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    SEMANTIC = "semantic"        # Facts, preferences, knowledge
    EPISODIC = "episodic"        # Events, experiences, conversations
    PROCEDURAL = "procedural"    # How-tos, processes, instructions


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    user_id: str
    metadata: dict[str, Any] = {}
    source: str = "chat"  # chat, email, voice, manual
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemorySearchResult(BaseModel):
    memory: Memory
    score: float
    source: str = "vector"  # "vector" | "graph" | "hybrid"


class Entity(BaseModel):
    name: str
    entity_type: str  # person, product, preference, location, etc.
    properties: dict[str, Any] = {}
    user_id: str


class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relation_type: str  # prefers, bought, lives_in, works_at, etc.
    properties: dict[str, Any] = {}
    user_id: str


class MemoryExtractionResult(BaseModel):
    memories: list[Memory] = []
    entities: list[Entity] = []
    relationships: list[Relationship] = []
