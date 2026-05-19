"""Standalone Memory API — use without intents, YAML, or pipeline."""

from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from memory_system.core.memory_models import (
    ConversationSummary,
    Memory as MemoryModel,
    MemoryExtractionResult,
    MemorySearchResult,
    MemoryStats,
    MemoryType,
    UserProfile,
)
from memory_system.core.protocols import GraphStore, MemoryStore
from memory_system.memory.extractor import extract_memories
from memory_system.memory.lifecycle import decay_memories, consolidate_memories, cleanup_expired
from memory_system.memory.profiles import build_user_profile
from memory_system.memory.context import build_context_window
from memory_system.memory.smart_ops import execute_decision, judge_memory_op


class Memory:
    """
    Standalone memory API. No intents, no YAML, no pipeline required.

    Usage:
        from memory_system import Memory
        from memory_system.providers.in_memory_stores import InMemoryMemoryStore

        mem = Memory(store=InMemoryMemoryStore())
        await mem.add("User prefers morning deliveries", user_id="user1")
        results = await mem.search("delivery preferences", user_id="user1")
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        graph: Optional[GraphStore] = None,
        llm_fn: Optional[Callable] = None,
        extraction_model: str = "groq/llama-3.1-8b-instant",
        extraction_prompt: Optional[str] = None,
        dedup_threshold: float = 0.85,
        default_ttl: Optional[int] = None,
        enable_smart_ops: bool = False,
        smart_ops_k: int = 5,
        smart_ops_model: Optional[str] = None,
        smart_ops_prompt: Optional[str] = None,
    ):
        self.store = store
        self.graph = graph
        self.llm_fn = llm_fn
        self.extraction_model = extraction_model
        self.extraction_prompt = extraction_prompt
        self.dedup_threshold = dedup_threshold
        self.default_ttl = default_ttl
        self.enable_smart_ops = enable_smart_ops
        self.smart_ops_k = smart_ops_k
        self.smart_ops_model = smart_ops_model or extraction_model
        self.smart_ops_prompt = smart_ops_prompt

    # --- Core CRUD ---

    async def add(
        self,
        text: str,
        user_id: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: Optional[dict] = None,
        importance: float = 0.5,
        ttl: Optional[int] = None,
        source: str = "manual",
    ) -> MemoryModel:
        memory = MemoryModel(
            text=text,
            user_id=user_id,
            memory_type=memory_type,
            metadata=metadata or {},
            importance=importance,
            ttl=ttl or self.default_ttl,
            source=source,
        )
        if self.store:
            await self.store.add(memory)
        return memory

    async def search(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        filters: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> list[MemorySearchResult]:
        if not self.store:
            return []
        results = await self.store.search(query=query, user_id=user_id, k=k, filters=filters)
        # Filter by min score and exclude expired
        return [
            r for r in results
            if r.score >= min_score and not r.memory.is_expired
        ]

    async def update(self, memory_id: str, text: str) -> None:
        if self.store:
            await self.store.update(memory_id, text)

    async def delete(self, memory_id: str) -> None:
        if self.store:
            await self.store.delete(memory_id)

    async def get_all(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        k: int = 50,
    ) -> list[MemorySearchResult]:
        if not self.store:
            return []
        results = await self.store.get_all(user_id, k=k)
        if memory_type:
            results = [r for r in results if r.memory.memory_type == memory_type]
        return [r for r in results if not r.memory.is_expired]

    # --- Smart Operations ---

    async def remember(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        session_id: Optional[str] = None,
    ) -> MemoryExtractionResult:
        """Auto-extract and store facts from conversation messages."""
        # Combine messages into user/assistant pairs
        user_msgs = []
        assistant_msgs = []
        for msg in messages:
            if msg.get("role") == "user":
                user_msgs.append(msg.get("content", ""))
            elif msg.get("role") == "assistant":
                assistant_msgs.append(msg.get("content", ""))

        user_text = " ".join(user_msgs)
        assistant_text = " ".join(assistant_msgs)

        if not user_text:
            return MemoryExtractionResult()

        turn_id = str(uuid4())
        extraction = await extract_memories(
            user_message=user_text,
            assistant_response=assistant_text,
            user_id=user_id,
            llm_fn=self.llm_fn,
            model=self.extraction_model,
            custom_prompt=self.extraction_prompt,
            source_text=user_text,
            turn_id=turn_id,
        )

        # Store: smart_ops path (LLM-judged) or simple dedup path
        stored = []
        if self.store:
            for mem in extraction.memories:
                mem.ttl = mem.ttl or self.default_ttl
                if session_id:
                    mem.metadata["session_id"] = session_id

                if self.enable_smart_ops and self.llm_fn:
                    candidates = await self.store.search(
                        mem.text, user_id=user_id, k=self.smart_ops_k
                    )
                    decision = await judge_memory_op(
                        mem,
                        candidates,
                        llm_fn=self.llm_fn,
                        model=self.smart_ops_model,
                        prompt_template=self.smart_ops_prompt,
                    )
                    result = await execute_decision(
                        decision, mem, candidates, self.store
                    )
                    if result is not None:
                        stored.append(result)
                else:
                    if not await self._is_duplicate(mem):
                        await self.store.add(mem)
                        stored.append(mem)

        # Store entities and relationships
        if self.graph:
            for entity in extraction.entities:
                await self.graph.add_entity(entity)
            for rel in extraction.relationships:
                await self.graph.add_relationship(rel)

        return MemoryExtractionResult(
            memories=stored,
            entities=extraction.entities,
            relationships=extraction.relationships,
        )

    async def recall(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        context: Optional[str] = None,
        graph_max_hops: int = 1,
        include_summaries: bool = False,
    ) -> list[MemorySearchResult]:
        """Search memories with optional context augmentation.

        When the graph store implements multi-hop `traverse`, paths up to
        `graph_max_hops` are surfaced as synthetic results (text formatted as
        ``A [rel] B -> B [rel] C``, score decayed per hop).
        """
        search_query = f"{context}: {query}" if context else query
        results = await self.search(search_query, user_id=user_id, k=k)

        # Also search graph for related entities
        if self.graph:
            entities = await self.graph.search_entities(query, user_id=user_id, k=3)
            traverse = getattr(self.graph, "traverse", None) if graph_max_hops > 1 else None
            for entity in entities:
                if traverse is not None:
                    paths = await traverse(
                        entity.name, user_id=user_id, max_hops=graph_max_hops
                    )
                    for path in paths:
                        text = " -> ".join(
                            f"{r.source_entity} [{r.relation_type}] {r.target_entity}"
                            for r in path
                        )
                        score = 0.7 * (0.8 ** (len(path) - 1))
                        graph_mem = MemoryModel(
                            text=text, memory_type=MemoryType.SEMANTIC,
                            user_id=user_id, source="graph",
                        )
                        results.append(
                            MemorySearchResult(memory=graph_mem, score=score, source="graph")
                        )
                else:
                    rels = await self.graph.get_related(entity.name, user_id=user_id)
                    for rel in rels:
                        text = f"{rel.source_entity} {rel.relation_type} {rel.target_entity}"
                        graph_mem = MemoryModel(
                            text=text, memory_type=MemoryType.SEMANTIC,
                            user_id=user_id, source="graph",
                        )
                        results.append(MemorySearchResult(memory=graph_mem, score=0.7, source="graph"))

        # Drop summary-tree nodes unless explicitly requested
        if not include_summaries:
            results = [
                r for r in results
                if not r.memory.metadata.get("summary_level")
            ]

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            key = r.memory.text.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:k]

    async def recall_at(
        self,
        query: str,
        user_id: str,
        as_of: datetime,
        k: int = 5,
    ) -> list[MemorySearchResult]:
        """Point-in-time recall: return memories that were valid at `as_of`."""
        if not self.store:
            return []
        search_at = getattr(self.store, "search_at", None)
        if search_at is not None:
            return await search_at(query, user_id=user_id, as_of=as_of, k=k)
        # Fallback: fetch all (including invalidated) then filter
        results = await self.store.search(
            query, user_id=user_id, k=k * 4, include_invalidated=True
        )
        return [r for r in results if r.memory.is_valid_at(as_of)][:k]

    async def forget(
        self,
        user_id: str,
        before: Optional[datetime] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Bulk delete memories for GDPR compliance."""
        if not self.store:
            return 0
        all_memories = await self.store.get_all(user_id, k=10000)
        count = 0
        for r in all_memories:
            should_delete = True
            if before and r.memory.created_at > before:
                should_delete = False
            if memory_type and r.memory.memory_type != memory_type:
                should_delete = False
            if should_delete:
                await self.store.delete(r.memory.id)
                count += 1
        return count

    # --- User Profile ---

    async def get_user_profile(self, user_id: str) -> UserProfile:
        memories = await self.get_all(user_id, k=100)
        return build_user_profile(user_id, memories)

    # --- Conversation ---

    async def summarize_conversation(
        self,
        turns: list[dict[str, str]],
        max_sentences: int = 3,
    ) -> ConversationSummary:
        """Summarize conversation turns into a compact summary."""
        if not self.llm_fn:
            # Fallback: just extract key content
            texts = [t.get("content", "") for t in turns if t.get("role") == "user"]
            return ConversationSummary(
                summary=" ".join(texts[:max_sentences]),
                key_facts=texts[:5],
                turn_count=len(turns),
            )

        from memory_system.providers.llm import call_llm
        _llm = self.llm_fn or call_llm

        conversation = "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}" for t in turns
        )
        prompt = f"""Summarize this conversation in {max_sentences} sentences. Also list 3-5 key facts.

{conversation}

Format:
SUMMARY: <summary>
FACTS:
- <fact1>
- <fact2>"""

        response = await _llm(
            model=self.extraction_model,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        summary = response
        facts = []
        for line in response.split("\n"):
            if line.startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()
            elif line.strip().startswith("- "):
                facts.append(line.strip()[2:])

        return ConversationSummary(
            summary=summary,
            key_facts=facts,
            turn_count=len(turns),
        )

    async def get_context_window(
        self,
        user_id: str,
        query: str,
        token_budget: int = 2000,
    ) -> str:
        """Build a context string that fits within token budget."""
        profile = await self.get_user_profile(user_id)
        memories = await self.recall(query, user_id=user_id, k=10)
        return build_context_window(profile, memories, query, token_budget)

    # --- Lifecycle ---

    async def decay(self, user_id: Optional[str] = None) -> int:
        """Reduce importance of old memories over time."""
        if not self.store:
            return 0
        return await decay_memories(self.store, user_id)

    async def consolidate(self, user_id: str) -> int:
        """Merge similar memories into stronger ones."""
        if not self.store:
            return 0
        return await consolidate_memories(self.store, user_id, self.dedup_threshold)

    async def cleanup(self, user_id: Optional[str] = None) -> int:
        """Remove expired and zero-importance memories."""
        if not self.store:
            return 0
        return await cleanup_expired(self.store, user_id)

    # --- Utility ---

    def format_memories(
        self, results: list[MemorySearchResult], format: str = "bullet"
    ) -> str:
        """Format memory results for prompt injection."""
        if not results:
            return ""
        if format == "bullet":
            return "\n".join(f"- {r.memory.text}" for r in results)
        elif format == "numbered":
            return "\n".join(f"{i+1}. {r.memory.text}" for i, r in enumerate(results))
        elif format == "plain":
            return "\n".join(r.memory.text for r in results)
        return "\n".join(f"- {r.memory.text}" for r in results)

    async def stats(self, user_id: Optional[str] = None) -> MemoryStats:
        """Get memory statistics."""
        if not self.store:
            return MemoryStats()

        # Determine which user(s) to get stats for
        if user_id:
            all_mems = await self.store.get_all(user_id, k=10000)
        else:
            all_mems = []

        by_type: dict[str, int] = {}
        by_source: dict[str, int] = {}
        total_importance = 0.0
        oldest = None
        newest = None

        for r in all_mems:
            m = r.memory
            by_type[m.memory_type.value] = by_type.get(m.memory_type.value, 0) + 1
            by_source[m.source] = by_source.get(m.source, 0) + 1
            total_importance += m.importance
            if oldest is None or m.created_at < oldest:
                oldest = m.created_at
            if newest is None or m.created_at > newest:
                newest = m.created_at

        n = len(all_mems)
        return MemoryStats(
            total_memories=n,
            by_type=by_type,
            by_source=by_source,
            avg_importance=round(total_importance / n, 3) if n > 0 else 0.0,
            oldest=oldest,
            newest=newest,
        )

    async def _is_duplicate(self, memory: MemoryModel) -> bool:
        if not self.store:
            return False
        existing = await self.store.search(query=memory.text, user_id=memory.user_id, k=1)
        return bool(existing and existing[0].score >= self.dedup_threshold)
