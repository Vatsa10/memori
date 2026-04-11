from memory_system._client import MemorySystem
from memory_system.memory.memory import Memory as StandaloneMemory
from memory_system.core.models import (
    BotConfig,
    IntentDefinition,
    IntentPrediction,
    PredictionMethod,
    ChatResponse,
    ConversationTurn,
    PipelineResult,
    AssembledContext,
    SmartPrompt,
)
from memory_system.core.memory_models import (
    Memory,
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
from memory_system.cache import IntentCache

__all__ = [
    # Two entry points
    "MemorySystem",       # Full pipeline: intent + context + memory
    "StandaloneMemory",   # Pure memory API: no intents, no YAML
    # Models
    "BotConfig",
    "IntentDefinition",
    "IntentPrediction",
    "PredictionMethod",
    "ChatResponse",
    "ConversationTurn",
    "PipelineResult",
    "AssembledContext",
    "SmartPrompt",
    "Memory",
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
    "IntentCache",
]
__version__ = "0.3.0"
