"""
MemorySystem — database-grounded memory + context system for AI agents.

Every response is grounded in:
1. Knowledge base (business docs, FAQs — searched from DB in real-time)
2. User memory (facts about THIS user from past conversations)
3. User profile (accumulated preferences, properties)
4. Conversation history (current session)
"""

import time
from typing import Any, Callable, Optional

from memory_system.core.models import ChatResponse, ConversationTurn
from memory_system.core.memory_models import (
    Memory as MemoryModel,
    MemorySearchResult,
    MemoryType,
    UserProfile,
)
from memory_system.core.protocols import GraphStore, MemoryStore
from memory_system.providers.session import SessionStore
from memory_system.memory.memory import Memory as StandaloneMemory
from memory_system.hooks import HookManager, EventType, Event
from memory_system.analytics import AnalyticsCollector

KNOWLEDGE_USER_ID = "__knowledge__"


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


class MemorySystem:
    """
    Database-grounded memory + context system.

    Usage:
        ms = MemorySystem(
            instructions="You are a support agent for Acme Corp.",
            llm_fn=my_llm,
            knowledge_store=InMemoryMemoryStore(),
            memory_store=InMemoryMemoryStore(),
        )

        await ms.add_knowledge("Returns accepted within 30 days.")
        result = await ms.chat("How do I return something?", user_id="user1")
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
        # Retrieval
        knowledge_top_k: int = 3,
        memory_top_k: int = 5,
        max_history_turns: int = 10,
        token_budget: int = 4000,
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

        # Standalone memory API (user memories + lifecycle + profiles)
        self._memory = StandaloneMemory(
            store=memory_store,
            graph=graph_store,
            llm_fn=extraction_llm_fn,
            extraction_model=extraction_model,
            dedup_threshold=dedup_threshold,
        )

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

    async def add_knowledge_batch(self, texts: list[str], source: str = "manual") -> int:
        """Add multiple knowledge entries at once."""
        count = 0
        for text in texts:
            await self.add_knowledge(text, source=source)
            count += 1
        return count

    async def search_knowledge(self, query: str, k: Optional[int] = None) -> list[MemorySearchResult]:
        """Search the knowledge base."""
        if not self._knowledge_store:
            return []
        return await self._knowledge_store.search(
            query=query, user_id=KNOWLEDGE_USER_ID, k=k or self._knowledge_top_k,
        )

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
                query=message, user_id=KNOWLEDGE_USER_ID, k=self._knowledge_top_k,
            )
            knowledge_used = len(knowledge_results)
            latency["knowledge_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Recall user memories + profile
        memory_results: list[MemorySearchResult] = []
        profile = UserProfile(user_id=effective_user_id)
        if self._memory.store:
            t0 = time.perf_counter()
            memory_results = await self._memory.recall(
                query=message, user_id=effective_user_id, k=self._memory_top_k,
            )
            memories_recalled = len(memory_results)
            profile = await self._memory.get_user_profile(effective_user_id)
            latency["memory_recall_ms"] = round((time.perf_counter() - t0) * 1000, 2)

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
        self._session_store.add_turn(session_id, ConversationTurn(role="user", content=message))
        self._session_store.add_turn(session_id, ConversationTurn(role="assistant", content=response))

        # Emit hooks
        if knowledge_used > 0:
            await self._hooks.emit(Event(
                type=EventType.CONTEXT_ASSEMBLED,
                data={"knowledge_used": knowledge_used},
            ))
        if memories_recalled > 0:
            await self._hooks.emit(Event(
                type=EventType.MEMORIES_RECALLED,
                data={"count": memories_recalled},
            ))
        await self._hooks.emit(Event(
            type=EventType.RESPONSE_GENERATED,
            data={"response": response, "latency": latency},
        ))
        if memories_stored > 0:
            await self._hooks.emit(Event(
                type=EventType.MEMORIES_STORED,
                data={"count": memories_stored},
            ))

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

    def chat_sync(self, message: str, session_id: str = "default", user_id: Optional[str] = None) -> ChatResponse:
        from memory_system._sync import run_sync
        return run_sync(self.chat(message, session_id, user_id))

    # --- Memory API (delegates to StandaloneMemory) ---

    async def add_memory(self, text: str, user_id: str, memory_type: MemoryType = MemoryType.SEMANTIC, **kwargs) -> MemoryModel:
        """Add a user memory."""
        return await self._memory.add(text, user_id=user_id, memory_type=memory_type, **kwargs)

    async def search_memory(self, query: str, user_id: str, k: int = 5) -> list[MemorySearchResult]:
        """Search user memories."""
        return await self._memory.search(query, user_id=user_id, k=k)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory."""
        await self._memory.delete(memory_id)

    async def get_user_memories(self, user_id: str, k: int = 50) -> list[MemorySearchResult]:
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
    def analytics(self) -> AnalyticsCollector | None:
        return self._analytics

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str):
        self._instructions = value

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
                system_parts.append("\n\n## Relevant Knowledge\n" + "\n".join(knowledge_lines))

        # 4. User memories (highest relevance first)
        if memories:
            memory_lines = []
            for r in sorted(memories, key=lambda x: (x.score, x.memory.importance), reverse=True):
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
        trimmed = history[-(self._max_history_turns * 2):]
        for turn in trimmed:
            cost = _estimate_tokens(turn.content)
            if used + cost > budget:
                break
            messages.append({"role": turn.role, "content": turn.content})
            used += cost

        # 6. Current user message (always included)
        messages.append({"role": "user", "content": user_message})

        return messages
