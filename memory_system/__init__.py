from memory_system._client import MemorySystem
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
    Entity,
    Relationship,
    MemoryExtractionResult,
)
from memory_system.hooks import HookManager, EventType, Event
from memory_system.analytics import AnalyticsCollector
from memory_system.cache import IntentCache

__all__ = [
    "MemorySystem",
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
    "Entity",
    "Relationship",
    "MemoryExtractionResult",
    "HookManager",
    "EventType",
    "Event",
    "AnalyticsCollector",
    "IntentCache",
]
__version__ = "0.3.0"
