from datetime import datetime, timedelta, timezone
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
    importance: float = 0.5  # 0.0 = trivial, 1.0 = critical
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    ttl: Optional[int] = None  # Seconds until expiry (None = forever)
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context):
        if self.ttl and not self.expires_at:
            self.expires_at = self.created_at + timedelta(seconds=self.ttl)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at


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


class UserProfile(BaseModel):
    user_id: str
    properties: dict[str, Any] = {}  # name, location, preferences, etc.
    summary: str = ""
    memory_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


class MemoryStats(BaseModel):
    total_memories: int = 0
    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    avg_importance: float = 0.0
    oldest: Optional[datetime] = None
    newest: Optional[datetime] = None


class ConversationSummary(BaseModel):
    summary: str
    key_facts: list[str] = []
    turn_count: int = 0
