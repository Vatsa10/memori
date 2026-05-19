from memory_system._client import MemorySystem
from memory_system.memory.memory import Memory
from memory_system.core.models import (
    ChatResponse,
    ConversationTurn,
)
from memory_system.core.memory_models import (
    Memory as MemoryModel,
    MemoryType,
    MemorySearchResult,
    MemoryStats,
    UserProfile,
    ConversationSummary,
    Entity,
    Relationship,
    MemoryExtractionResult,
)
from memory_system.hooks import HookManager, EventType, Event
from memory_system.analytics import AnalyticsCollector

StandaloneMemory = Memory  # For clarity; matches README

__all__ = [
    # Main entry points
    "MemorySystem",  # Grounded chat system (knowledge + memory + LLM)
    "Memory",  # Standalone memory API (plug into any agent)
    "StandaloneMemory",  # Alias for Memory (for clarity; matches README)
    # Response
    "ChatResponse",
    "ConversationTurn",
    # Memory models
    "MemoryModel",
    "MemoryType",
    "MemorySearchResult",
    "MemoryStats",
    "UserProfile",
    "ConversationSummary",
    "Entity",
    "Relationship",
    "MemoryExtractionResult",
    # Infrastructure
    "HookManager",
    "EventType",
    "Event",
    "AnalyticsCollector",
]
__version__ = "1.0.0"
