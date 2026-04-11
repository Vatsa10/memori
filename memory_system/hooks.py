import asyncio
import time
from enum import Enum
from typing import Callable, Any
from dataclasses import dataclass, field


class EventType(str, Enum):
    INTENT_PREDICTED = "intent_predicted"
    CONTEXT_ASSEMBLED = "context_assembled"
    PROMPT_BUILT = "prompt_built"
    RESPONSE_GENERATED = "response_generated"
    CACHE_HIT = "cache_hit"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HookManager:
    def __init__(self):
        self._hooks: dict[EventType, list[Callable]] = {e: [] for e in EventType}

    def on(self, event_type: EventType, callback: Callable[[Event], None]) -> "HookManager":
        self._hooks[event_type].append(callback)
        return self

    def off(self, event_type: EventType, callback: Callable):
        self._hooks[event_type] = [h for h in self._hooks[event_type] if h != callback]

    def clear(self):
        for event_type in EventType:
            self._hooks[event_type] = []

    async def emit(self, event: Event):
        for callback in self._hooks[event.type]:
            result = callback(event)
            if asyncio.iscoroutine(result):
                await result
