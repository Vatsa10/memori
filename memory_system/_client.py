"""
MemorySystem — database-grounded memory + context system for AI agents.

Every response is grounded in:
1. Knowledge base (business docs, FAQs — searched from DB in real-time)
2. User memory (facts about THIS user from past conversations)
3. User profile (accumulated preferences, properties)
4. Conversation history (current session)

Optional Intent-Aware Mode:
    When bot_config is provided, MemorySystem uses the Pipeline class
    for intent-aware processing, which predicts user intent and builds
    optimized prompts.
"""

import time
from typing import Any, Callable, Optional

from memory_system.core.models import (
    BotConfig,
    ChatResponse,
    ConversationTurn,
    IntentPrediction,
)
from memory_system.core.memory_models import (
    Memory as MemoryModel,
    MemorySearchResult,
    MemoryType,
    UserProfile,
)
from memory_system.core.protocols import GraphStore, MemoryStore
from memory_system.providers.session import SessionStore
from memory_system.memory.memory import Memory as StandaloneMemory
from memory_system.memory.manager import MemoryManager
from memory_system.core.pipeline import Pipeline
from memory_system.core.intent_predictor import IntentPredictor
from memory_system.hooks import HookManager, EventType, Event
from memory_system.analytics import AnalyticsCollector

KNOWLEDGE_USER_ID = "__knowledge__"


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


class _KnowledgeSearcherAdapter:
    """Adapter that wraps a MemoryStore to conform to the MemorySearcher protocol."""

    def __init__(self, store: MemoryStore):
        self.store = store

    async def search(self, query: str, k: int = 2) -> list[str]:
        results = await self.store.search(query=query, user_id=KNOWLEDGE_USER_ID, k=k)
        return [r.memory.text for r in results]


class MemorySystem:
    """
    Database-grounded memory + context system.

    Usage (Basic):
        ms = MemorySystem(
            instructions="You are a support agent for Acme Corp.",
            llm_fn=my_llm,
            knowledge_store=InMemoryMemoryStore(),
            memory_store=InMemoryMemoryStore(),
        )

        await ms.add_knowledge("Returns accepted within 30 days.")
        result = await ms.chat("How do I return something?", user_id="user1")

    Usage (Intent-Aware):
        config = load_bot_config(Path("bot.yaml"))
        ms = MemorySystem.from_config(config)

        result = await ms.chat("I want to return this item", user_id="user1")
    """

    def __init__(
        self,
        instructions: str = "You are a helpful AI assistant.",
        model: str = "groq/llama-3.3-70b-versatile",
        *,
        llm_fn: Optional[Callable] = None,
        # Knowledge base (business docs, FAQs, policies)
        knowledge_store: Optional[MemoryStore] = None,
        # User memory (per-user facts, preferences)
        memory_store: Optional[MemoryStore] = None,
        # Entity-relationship graph
        graph_store: Optional[GraphStore] = None,
        # Extraction
        extraction_llm_fn: Optional[Callable] = None,
        extraction_model: str = "groq/llama-3.1-8b-instant",
        dedup_threshold: float = 0.85,
        # Session
        session_store: Optional[SessionStore] = None,
        # Intent mode (optional)
        bot_config: Optional[BotConfig] = None,
        intent_llm_fn: Optional[Callable] = None,
        # Retrieval settings
        knowledge_top_k: int = 3,
        memory_top_k: int = 5,
        max_history_turns: int = 10,
        token_budget: int = 4000,
        # Smart memory ops (LLM-judged ADD/UPDATE/MERGE/DELETE/NOOP)
        enable_smart_ops: bool = False,
        smart_ops_k: int = 5,
        smart_ops_model: Optional[str] = None,
        smart_ops_prompt: Optional[str] = None,
        # Hybrid retrieval + rerank
        retriever: Optional[Any] = None,
        reranker: Optional[Any] = None,
        hybrid_top_n: int = 20,
        rerank_top_k: int = 5,
        # Analytics
        enable_analytics: bool = True,
    ):
        self._instructions = instructions
        self._model = model
        self._knowledge_store = knowledge_store
        self._knowledge_top_k = knowledge_top_k
        self._memory_top_k = memory_top_k
        self._max_history_turns = max_history_turns
        self._token_budget = token_budget
        self._session_store = session_store or SessionStore()
        self._hooks = HookManager()
        self._analytics = AnalyticsCollector() if enable_analytics else None

        # LLM
        self._llm_fn = llm_fn
        self._intent_llm_fn = intent_llm_fn

        # Intent mode
        self._bot_config = bot_config
        self._pipeline: Optional[Pipeline] = None

        # Standalone memory API (user memories + lifecycle + profiles)
        self._memory = StandaloneMemory(
            store=memory_store,
            graph=graph_store,
            llm_fn=extraction_llm_fn,
            extraction_model=extraction_model,
            dedup_threshold=dedup_threshold,
            enable_smart_ops=enable_smart_ops,
            smart_ops_k=smart_ops_k,
            smart_ops_model=smart_ops_model,
            smart_ops_prompt=smart_ops_prompt,
        )

        # Hybrid retrieval + rerank
        self._retriever = retriever
        self._reranker = reranker
        self._hybrid_top_n = hybrid_top_n
        self._rerank_top_k = rerank_top_k

        # If retriever has a BM25 cache, wire memory-mutation events to bump it
        bm25 = getattr(retriever, "bm25", None) if retriever is not None else None
        if bm25 is not None and hasattr(bm25, "bump"):
            async def _bump(event):
                uid = event.data.get("user_id") if hasattr(event, "data") else None
                if uid:
                    bm25.bump(uid)

            self._hooks.on(EventType.MEMORIES_STORED, _bump)

        # Initialize intent-aware pipeline if BotConfig provided
        if bot_config:
            self._pipeline = Pipeline(
                intent_predictor=IntentPredictor(),
                memory_provider=self._create_memory_searcher(self._knowledge_store),
                llm_fn=llm_fn,
                intent_llm_fn=intent_llm_fn,
                memory_manager=MemoryManager(
                    memory_store=memory_store,
                    graph_store=graph_store,
                    extraction_llm_fn=extraction_llm_fn,
                    extraction_model=extraction_model,
                    dedup_threshold=dedup_threshold,
                )
                if memory_store
                else None,
            )

    def _create_memory_searcher(
        self, store: Optional[MemoryStore]
    ) -> Optional["_KnowledgeSearcherAdapter"]:
        """Create a KnowledgeSearcher adapter for the Pipeline."""
        if not store:
            return None
        return _KnowledgeSearcherAdapter(store)

    # --- Class Factories ---

    @classmethod
    def from_env(cls, **overrides) -> "MemorySystem":
        """Create MemorySystem from environment configuration.

        Environment variables (via Settings):
        - BOT_CONFIG_PATH: Path to YAML config file (uses intent mode if set)
        - MEMORY_STORE, GRAPH_STORE, SESSION_STORE, CACHE: provider types
        - QDRANT_URL, NEO4J_URI, REDIS_URL, etc.: connection details

        Kwargs override environment settings and can include:
        - bot_config_path: alternative to BOT_CONFIG_PATH env var
        - memory_store_type, graph_store_type, session_store_type, cache_type: provider type strings
        - Any MemorySystem __init__ parameter (instructions, model, llm_fn, token_budget, etc.)
        - Direct store instances (memory_store=..., graph_store=..., etc.) to bypass factory
        """
        import os
        from memory_system.config.settings import settings
        from memory_system.config.factory import create_providers

        # If a bot config path is provided, delegate to from_yaml
        config_path = overrides.pop("bot_config_path", None) or os.getenv(
            "BOT_CONFIG_PATH"
        )
        if config_path:
            return cls.from_yaml(config_path, **overrides)

        # Extract provider type overrides for the factory (these are strings like "qdrant", "redis")
        factory_type_overrides = {}
        for key in [
            "memory_store_type",
            "graph_store_type",
            "session_store_type",
            "cache_type",
        ]:
            if key in overrides:
                factory_type_overrides[key] = overrides.pop(key)

        # Create providers using factory (which respects Settings and type overrides)
        providers = create_providers(**factory_type_overrides)

        # Build default kwargs for MemorySystem
        init_kwargs = dict(
            instructions=overrides.pop(
                "instructions", "You are a helpful AI assistant."
            ),
            model=overrides.pop("model", settings.generation_model),
            llm_fn=overrides.pop("llm_fn", None),
            knowledge_store=providers["memory_store"],
            memory_store=providers["memory_store"],
            graph_store=providers["graph_store"],
            session_store=providers["session_store"],
            extraction_llm_fn=overrides.pop("extraction_llm_fn", None),
            extraction_model=overrides.pop("extraction_model", settings.intent_model),
        )
        # Remaining overrides (token_budget, knowledge_top_k, memory_top_k, etc.) are applied
        init_kwargs.update(overrides)

        return cls(**init_kwargs)

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "MemorySystem":
        """Create MemorySystem from a YAML bot config file."""
        from memory_system.config.loader import load_bot_config
        from pathlib import Path

        config = load_bot_config(Path(path))
        return cls.from_config(config, **overrides)

    @classmethod
    def from_config(cls, config: BotConfig, **overrides) -> "MemorySystem":
        """Create MemorySystem from a BotConfig object."""
        from memory_system.config.factory import create_providers

        providers = create_providers(config)

        init_kwargs = dict(
            instructions=config.base_instructions,
            model=config.generation_model,
            knowledge_store=providers["memory_store"],
            memory_store=providers["memory_store"],
            graph_store=providers["graph_store"],
            session_store=providers["session_store"],
            bot_config=config,
        )
        init_kwargs.update(overrides)

        return cls(**init_kwargs)

    # --- Knowledge Management ---

    async def add_knowledge(
        self,
        text: str,
        metadata: Optional[dict] = None,
        source: str = "manual",
    ) -> MemoryModel:
        """Add business knowledge (docs, FAQs, policies) to the knowledge base."""
        if not self._knowledge_store:
            raise RuntimeError("No knowledge_store provided.")
        memory = MemoryModel(
            text=text,
            user_id=KNOWLEDGE_USER_ID,
            memory_type=MemoryType.PROCEDURAL,
            metadata=metadata or {},
            source=source,
            importance=0.8,
        )
        await self._knowledge_store.add(memory)
        return memory

    async def add_knowledge_batch(
        self, texts: list[str], source: str = "manual"
    ) -> int:
        """Add multiple knowledge entries at once."""
        count = 0
        for text in texts:
            await self.add_knowledge(text, source=source)
            count += 1
        return count

    async def search_knowledge(
        self, query: str, k: Optional[int] = None
    ) -> list[MemorySearchResult]:
        """Search the knowledge base."""
        if not self._knowledge_store:
            return []
        return await self._knowledge_store.search(
            query=query,
            user_id=KNOWLEDGE_USER_ID,
            k=k or self._knowledge_top_k,
        )

    # --- Document Ingestion (PDF / URL / Image / Audio) ---

    async def ingest_document(
        self,
        path_or_url,
        *,
        target: str = "knowledge",  # "knowledge" | "memory"
        user_id: Optional[str] = None,
        chunker=None,
        metadata: Optional[dict] = None,
        stream: bool = False,
    ):
        """Ingest a document into the knowledge_store or per-user memory_store.

        Dispatches by source type (PDF/URL/image/audio/text). Each chunk becomes
        a Memory record. Returns either a list[Memory] (default) or an async
        iterator yielding Memory as each chunk persists (when stream=True).
        """
        from memory_system.ingestion import (
            detect_source_type,
        )
        from memory_system.ingestion.chunker import SemanticChunker

        if target == "memory" and not user_id:
            raise ValueError("user_id is required when target='memory'")
        if target == "knowledge" and not self._knowledge_store:
            raise RuntimeError("No knowledge_store provided.")
        if target == "memory" and not self._memory.store:
            raise RuntimeError("No memory_store provided.")

        active_chunker = chunker or SemanticChunker()
        kind = detect_source_type(path_or_url)

        if kind == "pdf":
            from memory_system.ingestion.pdf import ingest_pdf
            chunks = await ingest_pdf(path_or_url, chunker=active_chunker)
        elif kind == "url":
            from memory_system.ingestion.url import ingest_url
            chunks = await ingest_url(path_or_url, chunker=active_chunker)
        elif kind == "image":
            from memory_system.ingestion.image import ingest_image
            llm = self._llm_fn
            if llm is None:
                raise RuntimeError(
                    "Image ingestion requires llm_fn on MemorySystem (vision-capable)."
                )
            chunks = await ingest_image(path_or_url, llm_fn=llm)
        elif kind == "audio":
            from memory_system.ingestion.audio import ingest_audio
            chunks = await ingest_audio(path_or_url, chunker=active_chunker)
        else:  # plain text
            from memory_system.ingestion.chunker import Chunk
            text = path_or_url if isinstance(path_or_url, str) else path_or_url.decode(
                "utf-8", errors="ignore"
            )
            chunks = active_chunker.chunk(
                text, base_metadata={"source": "text"}
            )

        if stream:
            return self._ingest_chunks_stream(
                chunks, target=target, user_id=user_id, metadata=metadata
            )
        return await self._ingest_chunks_collect(
            chunks, target=target, user_id=user_id, metadata=metadata
        )

    async def _persist_chunk(
        self,
        chunk,
        *,
        target: str,
        user_id: Optional[str],
        metadata: Optional[dict],
    ) -> MemoryModel:
        merged_meta = {**chunk.metadata, **(metadata or {}), "chunk_index": chunk.index}
        if target == "knowledge":
            mem = MemoryModel(
                text=chunk.text,
                user_id=KNOWLEDGE_USER_ID,
                memory_type=MemoryType.PROCEDURAL,
                metadata=merged_meta,
                source=chunk.metadata.get("source", "manual"),
                importance=0.7,
            )
            await self._knowledge_store.add(mem)
            return mem
        # target == "memory"
        mem = MemoryModel(
            text=chunk.text,
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            metadata=merged_meta,
            source=chunk.metadata.get("source", "manual"),
            importance=0.6,
        )
        await self._memory.store.add(mem)
        return mem

    async def _ingest_chunks_collect(
        self,
        chunks,
        *,
        target: str,
        user_id: Optional[str],
        metadata: Optional[dict],
    ) -> list[MemoryModel]:
        results: list[MemoryModel] = []
        for chunk in chunks:
            results.append(
                await self._persist_chunk(
                    chunk, target=target, user_id=user_id, metadata=metadata
                )
            )
        return results

    async def _ingest_chunks_stream(
        self,
        chunks,
        *,
        target: str,
        user_id: Optional[str],
        metadata: Optional[dict],
    ):
        for chunk in chunks:
            yield await self._persist_chunk(
                chunk, target=target, user_id=user_id, metadata=metadata
            )

    async def ingest_stream(
        self,
        chunks,
        *,
        target: str = "knowledge",
        user_id: Optional[str] = None,
        batch_size: int = 10,
        metadata: Optional[dict] = None,
    ):
        """Persist an (async)iterable of Chunks; yield each stored Memory.

        Concurrent within each batch via asyncio.gather; results yielded in
        completion order.
        """
        import asyncio
        import inspect

        if target == "memory" and not user_id:
            raise ValueError("user_id is required when target='memory'")
        if target == "knowledge" and not self._knowledge_store:
            raise RuntimeError("No knowledge_store provided.")
        if target == "memory" and not self._memory.store:
            raise RuntimeError("No memory_store provided.")

        is_async_iter = hasattr(chunks, "__aiter__")
        if is_async_iter:
            iterator = chunks.__aiter__()

            async def next_chunk():
                try:
                    return await iterator.__anext__()
                except StopAsyncIteration:
                    return None
        else:
            iterator = iter(chunks)

            async def next_chunk():
                try:
                    return next(iterator)
                except StopIteration:
                    return None

        batch: list = []
        while True:
            item = await next_chunk()
            if item is None:
                break
            batch.append(item)
            if len(batch) >= batch_size:
                results = await asyncio.gather(
                    *[
                        self._persist_chunk(
                            c, target=target, user_id=user_id, metadata=metadata
                        )
                        for c in batch
                    ]
                )
                for r in results:
                    yield r
                batch = []
        if batch:
            results = await asyncio.gather(
                *[
                    self._persist_chunk(
                        c, target=target, user_id=user_id, metadata=metadata
                    )
                    for c in batch
                ]
            )
            for r in results:
                yield r

    # --- Chat ---

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Send a message and get a grounded response.

        The system automatically:
        1. (Intent mode) Predicts user intent for optimized processing
        2. Retrieves relevant knowledge from the knowledge base
        3. Recalls user memories and profile
        4. Builds a token-budget-aware prompt
        5. Calls the LLM
        6. Extracts and stores new facts from the conversation
        """
        if self._pipeline and self._bot_config:
            return await self._chat_with_intents(
                message, session_id, user_id or session_id
            )
        else:
            return await self._chat_simple(message, session_id, user_id)

    async def _chat_with_intents(
        self, message: str, session_id: str, user_id: str
    ) -> ChatResponse:
        """Intent-aware chat using the Pipeline."""
        from memory_system.providers.llm import call_llm as default_call_llm

        history = self._session_store.get_history(session_id)
        latency = {}

        t0 = time.perf_counter()
        result = await self._pipeline.run(
            bot_config=self._bot_config,
            user_message=message,
            conversation_history=history,
            user_id=user_id,
            session_id=session_id,
        )
        latency["pipeline_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        response = result.response

        self._session_store.add_turn(
            session_id, ConversationTurn(role="user", content=message)
        )
        self._session_store.add_turn(
            session_id, ConversationTurn(role="assistant", content=response)
        )

        chat_response = ChatResponse(
            response=response,
            knowledge_used=0,
            memories_recalled=result.memories_recalled,
            memories_stored=result.memories_stored,
            token_estimate=result.smart_prompt.token_estimate,
            latency_ms=result.latency_ms,
            intent=result.intent,
            full_prompt_estimate=result.smart_prompt.full_prompt_estimate,
            reduction_percent=round(
                100
                * (
                    1
                    - result.smart_prompt.token_estimate
                    / result.smart_prompt.full_prompt_estimate
                )
                if result.smart_prompt.full_prompt_estimate > 0
                else 0,
                2,
            ),
        )

        if self._analytics:
            self._analytics.record(chat_response)

        return chat_response

    async def _chat_simple(
        self,
        message: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Simple (non-intent) chat mode - existing implementation.

        The system automatically:
        1. Retrieves relevant knowledge from the knowledge base
        2. Recalls user memories and profile
        3. Builds a token-budget-aware prompt
        4. Calls the LLM
        5. Extracts and stores new facts from the conversation
        """
        from memory_system.providers.llm import call_llm as default_call_llm

        effective_user_id = user_id or session_id
        history = self._session_store.get_history(session_id)
        latency: dict[str, float] = {}
        knowledge_used = 0
        memories_recalled = 0
        memories_stored = 0

        # 1. Retrieve relevant knowledge
        knowledge_results: list[MemorySearchResult] = []
        if self._knowledge_store:
            t0 = time.perf_counter()
            knowledge_results = await self._knowledge_store.search(
                query=message,
                user_id=KNOWLEDGE_USER_ID,
                k=self._knowledge_top_k,
            )
            knowledge_used = len(knowledge_results)
            latency["knowledge_retrieval_ms"] = round(
                (time.perf_counter() - t0) * 1000, 2
            )

        # 2. Recall user memories + profile
        memory_results: list[MemorySearchResult] = []
        profile = UserProfile(user_id=effective_user_id)
        if self._memory.store:
            t0 = time.perf_counter()
            if self._retriever is not None:
                memory_results = await self._retriever.search(
                    query=message,
                    user_id=effective_user_id,
                    k=self._hybrid_top_n,
                )
                latency["hybrid_retrieval_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
            else:
                memory_results = await self._memory.recall(
                    query=message,
                    user_id=effective_user_id,
                    k=self._memory_top_k,
                )
                latency["memory_recall_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )

            if self._reranker is not None and memory_results:
                t0 = time.perf_counter()
                memory_results = await self._reranker.rerank(
                    message, memory_results, top_k=self._rerank_top_k
                )
                latency["rerank_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
            elif self._retriever is not None:
                memory_results = memory_results[: self._memory_top_k]

            memories_recalled = len(memory_results)
            profile = await self._memory.get_user_profile(effective_user_id)

        # 3. Build grounded prompt (token-budget aware)
        t0 = time.perf_counter()
        messages = self._build_grounded_prompt(
            profile=profile,
            knowledge=knowledge_results,
            memories=memory_results,
            history=history,
            user_message=message,
        )
        token_estimate = sum(_estimate_tokens(m["content"]) for m in messages)
        latency["prompt_build_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 4. Call LLM
        llm = self._llm_fn or default_call_llm
        t0 = time.perf_counter()
        response = await llm(model=self._model, messages=messages)
        latency["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 5. Remember new facts
        if self._memory.store:
            t0 = time.perf_counter()
            extraction = await self._memory.remember(
                messages=[
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ],
                user_id=effective_user_id,
                session_id=session_id,
            )
            memories_stored = len(extraction.memories)
            latency["memory_store_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        latency["total_ms"] = round(sum(latency.values()), 2)

        # Store conversation turns
        self._session_store.add_turn(
            session_id, ConversationTurn(role="user", content=message)
        )
        self._session_store.add_turn(
            session_id, ConversationTurn(role="assistant", content=response)
        )

        # Emit hooks
        if knowledge_used > 0:
            await self._hooks.emit(
                Event(
                    type=EventType.CONTEXT_ASSEMBLED,
                    data={"knowledge_used": knowledge_used},
                )
            )
        if memories_recalled > 0:
            await self._hooks.emit(
                Event(
                    type=EventType.MEMORIES_RECALLED,
                    data={"count": memories_recalled},
                )
            )
        await self._hooks.emit(
            Event(
                type=EventType.RESPONSE_GENERATED,
                data={"response": response, "latency": latency},
            )
        )
        if memories_stored > 0:
            await self._hooks.emit(
                Event(
                    type=EventType.MEMORIES_STORED,
                    data={"count": memories_stored, "user_id": effective_user_id},
                )
            )

        chat_response = ChatResponse(
            response=response,
            knowledge_used=knowledge_used,
            memories_recalled=memories_recalled,
            memories_stored=memories_stored,
            token_estimate=token_estimate,
            latency_ms=latency,
        )

        if self._analytics:
            self._analytics.record(chat_response)

        return chat_response

    def chat_sync(
        self, message: str, session_id: str = "default", user_id: Optional[str] = None
    ) -> ChatResponse:
        from memory_system._sync import run_sync

        return run_sync(self.chat(message, session_id, user_id))

    # --- Memory API (delegates to StandaloneMemory) ---

    async def add_memory(
        self,
        text: str,
        user_id: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        **kwargs,
    ) -> MemoryModel:
        """Add a user memory."""
        return await self._memory.add(
            text, user_id=user_id, memory_type=memory_type, **kwargs
        )

    async def search_memory(
        self, query: str, user_id: str, k: int = 5
    ) -> list[MemorySearchResult]:
        """Search user memories."""
        return await self._memory.search(query, user_id=user_id, k=k)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory."""
        await self._memory.delete(memory_id)

    async def get_user_memories(
        self, user_id: str, k: int = 50
    ) -> list[MemorySearchResult]:
        """Get all memories for a user."""
        return await self._memory.get_all(user_id, k=k)

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile."""
        return await self._memory.get_user_profile(user_id)

    async def forget_user(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR)."""
        return await self._memory.forget(user_id)

    # --- Lifecycle ---

    async def decay_memories(self, user_id: Optional[str] = None) -> int:
        return await self._memory.decay(user_id)

    async def consolidate_memories(self, user_id: str) -> int:
        return await self._memory.consolidate(user_id)

    async def cleanup_memories(self, user_id: Optional[str] = None) -> int:
        return await self._memory.cleanup(user_id)

    # --- Properties ---

    @property
    def memory(self) -> StandaloneMemory:
        """Direct access to the standalone Memory API."""
        return self._memory

    @property
    def hooks(self) -> HookManager:
        return self._hooks

    @property
    def analytics(self) -> Optional[AnalyticsCollector]:
        return self._analytics

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str):
        self._instructions = value

    @property
    def pipeline(self) -> Optional[Pipeline]:
        """The intent-aware Pipeline, if initialized."""
        return self._pipeline

    @property
    def bot_config(self) -> Optional[BotConfig]:
        """The BotConfig, if provided."""
        return self._bot_config

    def clear_session(self, session_id: str = "default"):
        self._session_store.clear(session_id)

    def export_analytics(self) -> dict:
        if self._analytics:
            return self._analytics.export()
        return {}

    # --- Internal ---

    def _build_grounded_prompt(
        self,
        profile: UserProfile,
        knowledge: list[MemorySearchResult],
        memories: list[MemorySearchResult],
        history: list[ConversationTurn],
        user_message: str,
    ) -> list[dict]:
        """Build LLM messages grounded in knowledge + memory, within token budget."""
        budget = self._token_budget
        used = 0

        # 1. System instructions (always included)
        system_parts = [self._instructions]
        used += _estimate_tokens(self._instructions)

        # 2. User profile
        if profile.summary:
            section = f"\n\n## About This User\n{profile.summary}"
            cost = _estimate_tokens(section)
            if used + cost <= budget:
                system_parts.append(section)
                used += cost

            if profile.properties:
                props = []
                for k, v in profile.properties.items():
                    if k == "preferences" and isinstance(v, list):
                        props.append(f"- Preferences: {', '.join(v)}")
                    else:
                        props.append(f"- {k.title()}: {v}")
                props_text = "\n".join(props)
                cost = _estimate_tokens(props_text)
                if used + cost <= budget:
                    system_parts.append(props_text)
                    used += cost

        # 3. Knowledge context (highest relevance first)
        if knowledge:
            knowledge_lines = []
            for r in sorted(knowledge, key=lambda x: x.score, reverse=True):
                line = f"- {r.memory.text}"
                cost = _estimate_tokens(line)
                if used + cost > budget:
                    break
                knowledge_lines.append(line)
                used += cost
            if knowledge_lines:
                system_parts.append(
                    "\n\n## Relevant Knowledge\n" + "\n".join(knowledge_lines)
                )

        # 4. User memories (highest relevance first)
        if memories:
            memory_lines = []
            for r in sorted(
                memories, key=lambda x: (x.score, x.memory.importance), reverse=True
            ):
                line = f"- {r.memory.text}"
                cost = _estimate_tokens(line)
                if used + cost > budget:
                    break
                memory_lines.append(line)
                used += cost
            if memory_lines:
                system_parts.append("\n\n## User Memories\n" + "\n".join(memory_lines))

        system_message = "".join(system_parts)
        messages = [{"role": "system", "content": system_message}]

        # 5. Conversation history (most recent, within budget)
        trimmed = history[-(self._max_history_turns * 2) :]
        for turn in trimmed:
            cost = _estimate_tokens(turn.content)
            if used + cost > budget:
                break
            messages.append({"role": turn.role, "content": turn.content})
            used += cost

        # 6. Current user message (always included)
        messages.append({"role": "user", "content": user_message})

        return messages
