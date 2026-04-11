from pathlib import Path
from typing import Callable, Optional

from memory_system.core.models import (
    BotConfig,
    ChatResponse,
    ConversationTurn,
    IntentPrediction,
)
from memory_system.core.memory_models import (
    Memory,
    MemorySearchResult,
    MemoryType,
)
from memory_system.core.intent_predictor import IntentPredictor
from memory_system.core.pipeline import Pipeline
from memory_system.core.context_assembler import MemorySearcher
from memory_system.core.protocols import GraphStore, MemoryStore
from memory_system.config.loader import load_bot_config
from memory_system.providers.session import SessionStore
from memory_system.memory.manager import MemoryManager
from memory_system.memory.memory import Memory as StandaloneMemory
from memory_system.hooks import HookManager, EventType, Event
from memory_system.cache import IntentCache
from memory_system.analytics import AnalyticsCollector


class MemorySystem:
    """
    Intent-aware memory + context management system.

    Usage:
        ms = MemorySystem.from_yaml("bot.yaml")
        result = await ms.chat("Where is my order?", user_id="user1")
    """

    def __init__(
        self,
        config: BotConfig,
        *,
        # LLM functions
        llm_fn: Optional[Callable] = None,
        intent_llm_fn: Optional[Callable] = None,
        extraction_llm_fn: Optional[Callable] = None,
        extraction_model: str = "groq/llama-3.1-8b-instant",
        # Knowledge base (static docs, FAQs)
        memory_provider: Optional[MemorySearcher] = None,
        # Persistent memory stores
        memory_store: Optional[MemoryStore] = None,
        graph_store: Optional[GraphStore] = None,
        enable_memory: bool = True,
        dedup_threshold: float = 0.85,
        # Session
        session_store: Optional[SessionStore] = None,
        # Embeddings
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_embeddings: bool = True,
        # Cache + analytics
        cache_size: int = 256,
        enable_analytics: bool = True,
    ):
        self.config = config
        self._session_store = session_store or SessionStore()
        self._hooks = HookManager()
        self._cache = IntentCache(maxsize=cache_size) if cache_size > 0 else None
        self._analytics = AnalyticsCollector() if enable_analytics else None

        # Intent predictor
        self._predictor = IntentPredictor(embedding_model_name=embedding_model)
        if enable_embeddings:
            self._predictor.precompute_intent_embeddings(config)

        # Standalone memory API (usable directly via ms.memory)
        self._standalone_memory = None
        if enable_memory and (memory_store or graph_store):
            self._standalone_memory = StandaloneMemory(
                store=memory_store,
                graph=graph_store,
                llm_fn=extraction_llm_fn,
                extraction_model=extraction_model,
                dedup_threshold=dedup_threshold,
            )

        # Memory manager (for pipeline integration)
        self._memory_manager = None
        if enable_memory and (memory_store or graph_store):
            self._memory_manager = MemoryManager(
                memory_store=memory_store,
                graph_store=graph_store,
                extraction_llm_fn=extraction_llm_fn,
                extraction_model=extraction_model,
                dedup_threshold=dedup_threshold,
            )

        # Pipeline
        self._pipeline = Pipeline(
            intent_predictor=self._predictor,
            memory_provider=memory_provider,
            llm_fn=llm_fn,
            intent_llm_fn=intent_llm_fn,
            memory_manager=self._memory_manager,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs) -> "MemorySystem":
        config = load_bot_config(Path(path))
        return cls(config, **kwargs)

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "MemorySystem":
        config = BotConfig(**data)
        return cls(config, **kwargs)

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        # user_id defaults to session_id for memory scoping
        effective_user_id = user_id or session_id
        history = self._session_store.get_history(session_id)
        cache_hit = False

        # Check intent cache
        cached_intent = None
        if self._cache:
            cached_intent = self._cache.get(self.config.bot_id, message)
            if cached_intent:
                cache_hit = True
                await self._hooks.emit(Event(
                    type=EventType.CACHE_HIT,
                    data={"intent": cached_intent.intent_name, "message": message},
                ))

        # Run pipeline (includes recall + remember stages if memory enabled)
        result = await self._pipeline.run(
            bot_config=self.config,
            user_message=message,
            conversation_history=history,
            cached_intent=cached_intent,
            user_id=effective_user_id,
            session_id=session_id,
        )

        # Cache the intent prediction
        if self._cache and not cache_hit:
            self._cache.put(self.config.bot_id, message, result.intent)

        # Emit hooks
        await self._hooks.emit(Event(
            type=EventType.INTENT_PREDICTED,
            data={"prediction": result.intent.model_dump()},
        ))
        if result.memories_recalled > 0:
            await self._hooks.emit(Event(
                type=EventType.MEMORIES_RECALLED,
                data={"count": result.memories_recalled},
            ))
        await self._hooks.emit(Event(
            type=EventType.RESPONSE_GENERATED,
            data={"response": result.response, "latency": result.latency_ms},
        ))
        if result.memories_stored > 0:
            await self._hooks.emit(Event(
                type=EventType.MEMORIES_STORED,
                data={"count": result.memories_stored},
            ))

        # Store conversation turns
        self._session_store.add_turn(
            session_id, ConversationTurn(role="user", content=message)
        )
        self._session_store.add_turn(
            session_id, ConversationTurn(role="assistant", content=result.response)
        )

        # Build response
        smart_tokens = result.smart_prompt.token_estimate
        full_tokens = result.smart_prompt.full_prompt_estimate
        reduction = (
            ((full_tokens - smart_tokens) / full_tokens * 100) if full_tokens > 0 else 0.0
        )

        response = ChatResponse(
            response=result.response,
            intent=result.intent,
            token_estimate=smart_tokens,
            full_prompt_estimate=full_tokens,
            reduction_percent=round(reduction, 1),
            latency_ms=result.latency_ms,
            memories_recalled=result.memories_recalled,
            memories_stored=result.memories_stored,
        )

        if self._analytics:
            self._analytics.record(response, cache_hit=cache_hit)

        return response

    def chat_sync(self, message: str, session_id: str = "default", user_id: Optional[str] = None) -> ChatResponse:
        from memory_system._sync import run_sync
        return run_sync(self.chat(message, session_id, user_id))

    # --- Memory API ---

    async def recall(self, query: str, user_id: str, k: int = 5) -> list[MemorySearchResult]:
        if not self._memory_manager:
            return []
        return await self._memory_manager.recall(query=query, user_id=user_id, k=k)

    async def add_memory(
        self, text: str, user_id: str, memory_type: MemoryType = MemoryType.SEMANTIC
    ) -> Memory:
        if not self._memory_manager:
            raise RuntimeError("Memory not enabled. Provide memory_store to enable.")
        return await self._memory_manager.add_memory(text=text, user_id=user_id, memory_type=memory_type)

    async def delete_memory(self, memory_id: str) -> None:
        if not self._memory_manager:
            raise RuntimeError("Memory not enabled.")
        await self._memory_manager.delete_memory(memory_id)

    async def get_user_memories(self, user_id: str, k: int = 50) -> list[MemorySearchResult]:
        if not self._memory_manager:
            return []
        return await self._memory_manager.get_user_memories(user_id, k=k)

    async def predict_intent(self, message: str) -> IntentPrediction:
        prediction, _ = await self._predictor.predict(
            bot_config=self.config,
            user_message=message,
            recent_history=[],
            llm_predict_fn=self._pipeline._intent_llm_fn,
        )
        return prediction

    # --- Properties ---

    @property
    def memory(self) -> StandaloneMemory | None:
        """Direct access to the standalone Memory API."""
        return self._standalone_memory

    @property
    def hooks(self) -> HookManager:
        return self._hooks

    @property
    def analytics(self) -> AnalyticsCollector | None:
        return self._analytics

    @property
    def cache(self) -> IntentCache | None:
        return self._cache

    @property
    def memory_manager(self) -> MemoryManager | None:
        return self._memory_manager

    def clear_session(self, session_id: str = "default"):
        self._session_store.clear(session_id)

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

    def export_analytics(self) -> dict:
        if self._analytics:
            return self._analytics.export()
        return {}
