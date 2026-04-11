from smartcontext._client import SmartContext
from smartcontext.core.models import (
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
from smartcontext.hooks import HookManager, EventType, Event
from smartcontext.analytics import AnalyticsCollector
from smartcontext.cache import IntentCache

__all__ = [
    "SmartContext",
    "BotConfig",
    "IntentDefinition",
    "IntentPrediction",
    "PredictionMethod",
    "ChatResponse",
    "ConversationTurn",
    "PipelineResult",
    "AssembledContext",
    "SmartPrompt",
    "HookManager",
    "EventType",
    "Event",
    "AnalyticsCollector",
    "IntentCache",
]
__version__ = "0.2.0"
