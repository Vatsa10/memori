from pathlib import Path
from typing import Callable, Optional

from smartcontext.core.models import (
    BotConfig,
    ChatResponse,
    ConversationTurn,
    IntentPrediction,
    PipelineResult,
)
from smartcontext.core.intent_predictor import IntentPredictor
from smartcontext.core.pipeline import Pipeline
from smartcontext.core.context_assembler import MemorySearcher
from smartcontext.config.loader import load_bot_config
from smartcontext.providers.session import SessionStore
from smartcontext.hooks import HookManager, EventType, Event
from smartcontext.cache import IntentCache
from smartcontext.analytics import AnalyticsCollector


class SmartContext:
    """
    Main entry point for the intent-aware context management system.

    Usage:
        ctx = SmartContext.from_yaml("my_bot.yaml")
        result = await ctx.chat("Where is my order?", session_id="user123")
    """

    def __init__(
        self,
        config: BotConfig,
        *,
        llm_fn: Optional[Callable] = None,
        intent_llm_fn: Optional[Callable] = None,
        memory_provider: Optional[MemorySearcher] = None,
        session_store: Optional[SessionStore] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_embeddings: bool = True,
        cache_size: int = 256,
        enable_analytics: bool = True,
    ):
        self.config = config
        self._session_store = session_store or SessionStore()
        self._hooks = HookManager()
        self._cache = IntentCache(maxsize=cache_size) if cache_size > 0 else None
        self._analytics = AnalyticsCollector() if enable_analytics else None

        # Initialize intent predictor
        self._predictor = IntentPredictor(embedding_model_name=embedding_model)
        if enable_embeddings:
            self._predictor.precompute_intent_embeddings(config)

        # Initialize pipeline with injectable LLM functions
        self._pipeline = Pipeline(
            intent_predictor=self._predictor,
            memory_provider=memory_provider,
            llm_fn=llm_fn,
            intent_llm_fn=intent_llm_fn,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs) -> "SmartContext":
        config = load_bot_config(Path(path))
        return cls(config, **kwargs)

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "SmartContext":
        config = BotConfig(**data)
        return cls(config, **kwargs)

    async def chat(self, message: str, session_id: str = "default") -> ChatResponse:
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

        # Run pipeline
        result = await self._pipeline.run(
            bot_config=self.config,
            user_message=message,
            conversation_history=history,
            cached_intent=cached_intent,
        )

        # Cache the intent prediction
        if self._cache and not cache_hit:
            self._cache.put(self.config.bot_id, message, result.intent)

        # Emit hooks
        await self._hooks.emit(Event(
            type=EventType.INTENT_PREDICTED,
            data={"prediction": result.intent.model_dump()},
        ))
        await self._hooks.emit(Event(
            type=EventType.RESPONSE_GENERATED,
            data={"response": result.response, "latency": result.latency_ms},
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
        )

        # Record analytics
        if self._analytics:
            self._analytics.record(response, cache_hit=cache_hit)

        return response

    def chat_sync(self, message: str, session_id: str = "default") -> ChatResponse:
        from smartcontext._sync import run_sync
        return run_sync(self.chat(message, session_id))

    async def predict_intent(self, message: str) -> IntentPrediction:
        prediction, _ = await self._predictor.predict(
            bot_config=self.config,
            user_message=message,
            recent_history=[],
            llm_predict_fn=self._pipeline._intent_llm_fn,
        )
        return prediction

    @property
    def hooks(self) -> HookManager:
        return self._hooks

    @property
    def analytics(self) -> AnalyticsCollector | None:
        return self._analytics

    @property
    def cache(self) -> IntentCache | None:
        return self._cache

    def clear_session(self, session_id: str = "default"):
        self._session_store.clear(session_id)

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

    def export_analytics(self) -> dict:
        if self._analytics:
            return self._analytics.export()
        return {}
