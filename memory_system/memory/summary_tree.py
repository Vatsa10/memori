"""Hierarchical summary tree: turn → session → day → month.

Summaries are stored as Memory(memory_type=EPISODIC) with metadata flags so
they live in the same memory_store and ride existing retrieval (dense + BM25).
Default recall filters them out; search_hierarchical opts in.
"""

from datetime import date, datetime, time, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from memory_system.core.memory_models import (
    Memory,
    MemorySearchResult,
    MemoryType,
    SummaryLevel,
    SummaryNode,
)

TURN_PROMPT = """Summarize this conversation turn in one sentence.

User: {user_msg}
Assistant: {assistant_msg}

One-sentence summary:"""

ROLLUP_PROMPT = """Summarize the following {child_level}-level summaries into a single
higher-level summary. Pull out 3-5 key facts.

{children_block}

Format:
SUMMARY: <summary>
FACTS:
- <fact1>
- <fact2>"""


def _children_block(children: list[SummaryNode]) -> str:
    lines = []
    for c in children:
        lines.append(
            f"- ({c.time_range_start.isoformat()}–{c.time_range_end.isoformat()}) {c.content}"
        )
    return "\n".join(lines) or "(no children)"


def _parse_rollup(response: str) -> tuple[str, list[str]]:
    summary = response.strip()
    facts: list[str] = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif line.startswith("- "):
            facts.append(line[2:].strip())
    return summary, facts


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _summary_memory(node: SummaryNode) -> Memory:
    """Materialize a SummaryNode as a Memory record for storage."""
    return Memory(
        id=node.id,
        text=node.content,
        memory_type=MemoryType.EPISODIC,
        user_id=node.user_id,
        source="summary_tree",
        metadata={
            **node.metadata,
            "summary_level": node.level.value,
            "child_ids": node.child_ids,
            "key_facts": node.key_facts,
            "time_range_start": node.time_range_start.isoformat(),
            "time_range_end": node.time_range_end.isoformat(),
            "bitemporal_immutable": True,
        },
    )


def _node_from_memory(mem: Memory) -> Optional[SummaryNode]:
    level_str = mem.metadata.get("summary_level")
    if not level_str:
        return None
    try:
        level = SummaryLevel(level_str)
    except ValueError:
        return None
    return SummaryNode(
        id=mem.id,
        user_id=mem.user_id,
        level=level,
        content=mem.text,
        key_facts=mem.metadata.get("key_facts", []),
        child_ids=mem.metadata.get("child_ids", []),
        time_range_start=datetime.fromisoformat(mem.metadata["time_range_start"]),
        time_range_end=datetime.fromisoformat(mem.metadata["time_range_end"]),
        created_at=mem.created_at,
        metadata=mem.metadata,
    )


class SummaryTreeManager:
    """Builds & queries the hierarchical summary tree."""

    def __init__(
        self,
        memory_store: Any,
        llm_fn: Callable,
        model: str = "groq/llama-3.1-8b-instant",
        session_to_day_hour: int = 2,
        day_to_month_day: int = 1,
        turn_summary_min_chars: int = 80,
    ):
        self.store = memory_store
        self.llm_fn = llm_fn
        self.model = model
        self.session_to_day_hour = session_to_day_hour
        self.day_to_month_day = day_to_month_day
        self.turn_summary_min_chars = turn_summary_min_chars

    # ---------- TURN ----------

    async def summarize_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        user_id: str,
        turn_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[SummaryNode]:
        """Create a TURN-level summary node from a single conversation turn."""
        combined_len = len(user_msg or "") + len(assistant_msg or "")
        if combined_len < self.turn_summary_min_chars:
            return None  # not worth summarizing

        prompt = TURN_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
        try:
            content = await self.llm_fn(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            content = (content or "").strip()
        except Exception:
            return None
        if not content:
            return None

        now = _now()
        node = SummaryNode(
            id=str(uuid4()),
            user_id=user_id,
            level=SummaryLevel.TURN,
            content=content,
            key_facts=[],
            child_ids=[],
            time_range_start=now,
            time_range_end=now,
            metadata={"turn_id": turn_id, "session_id": session_id},
        )
        await self.store.add(_summary_memory(node))
        return node

    # ---------- internal: collect summaries at a level ----------

    async def _children_at(
        self,
        user_id: str,
        level: SummaryLevel,
        start: datetime,
        end: datetime,
    ) -> list[SummaryNode]:
        all_mems = await self.store.get_all(user_id, k=10_000)
        children: list[SummaryNode] = []
        for r in all_mems:
            node = _node_from_memory(r.memory)
            if node is None:
                continue
            if node.level != level:
                continue
            if node.time_range_end < start or node.time_range_start > end:
                continue
            children.append(node)
        children.sort(key=lambda n: n.time_range_start)
        return children

    async def _rollup(
        self,
        user_id: str,
        target_level: SummaryLevel,
        child_level: SummaryLevel,
        start: datetime,
        end: datetime,
        extra_metadata: Optional[dict] = None,
    ) -> Optional[SummaryNode]:
        children = await self._children_at(user_id, child_level, start, end)
        if not children:
            return None

        prompt = ROLLUP_PROMPT.format(
            child_level=child_level.value,
            children_block=_children_block(children),
        )
        try:
            response = await self.llm_fn(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
        except Exception:
            return None
        summary, facts = _parse_rollup(response or "")
        if not summary:
            return None

        node = SummaryNode(
            id=str(uuid4()),
            user_id=user_id,
            level=target_level,
            content=summary,
            key_facts=facts,
            child_ids=[c.id for c in children],
            time_range_start=start,
            time_range_end=end,
            metadata=dict(extra_metadata or {}),
        )
        await self.store.add(_summary_memory(node))
        return node

    async def rollup_session(
        self, session_id: str, user_id: str
    ) -> Optional[SummaryNode]:
        # find session's turn nodes via metadata
        all_mems = await self.store.get_all(user_id, k=10_000)
        nodes: list[SummaryNode] = []
        for r in all_mems:
            n = _node_from_memory(r.memory)
            if n and n.level == SummaryLevel.TURN and n.metadata.get("session_id") == session_id:
                nodes.append(n)
        if not nodes:
            return None
        start = min(n.time_range_start for n in nodes)
        end = max(n.time_range_end for n in nodes)
        return await self._rollup(
            user_id,
            target_level=SummaryLevel.SESSION,
            child_level=SummaryLevel.TURN,
            start=start,
            end=end,
            extra_metadata={"session_id": session_id},
        )

    async def rollup_day(
        self, user_id: str, day: date
    ) -> Optional[SummaryNode]:
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        end = datetime.combine(day, time.max, tzinfo=timezone.utc)
        return await self._rollup(
            user_id,
            target_level=SummaryLevel.DAY,
            child_level=SummaryLevel.SESSION,
            start=start,
            end=end,
            extra_metadata={"day": day.isoformat()},
        )

    async def rollup_month(
        self, user_id: str, year_month: str
    ) -> Optional[SummaryNode]:
        year, month = year_month.split("-")
        from calendar import monthrange

        last_day = monthrange(int(year), int(month))[1]
        start = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
        end = datetime(int(year), int(month), last_day, 23, 59, 59, tzinfo=timezone.utc)
        return await self._rollup(
            user_id,
            target_level=SummaryLevel.MONTH,
            child_level=SummaryLevel.DAY,
            start=start,
            end=end,
            extra_metadata={"year_month": year_month},
        )

    # ---------- retrieval ----------

    async def search_hierarchical(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        drill_threshold: float = 0.6,
        max_level: SummaryLevel = SummaryLevel.MONTH,
    ) -> list[MemorySearchResult]:
        """Coarse-first search; drill into children whose parent score ≥ threshold."""
        order = [
            SummaryLevel.MONTH,
            SummaryLevel.DAY,
            SummaryLevel.SESSION,
            SummaryLevel.TURN,
        ]
        results: list[MemorySearchResult] = []
        for level in order:
            if order.index(level) < order.index(max_level):
                continue  # skip levels above max_level
            level_results = await self.store.search(
                query=query, user_id=user_id, k=k,
                filters={"summary_level": level.value},
            )
            for r in level_results:
                results.append(r)
                # If high score, also surface direct children
                if r.score >= drill_threshold:
                    child_ids = r.memory.metadata.get("child_ids", [])
                    if child_ids:
                        # Fetch children by id from full corpus
                        all_mems = await self.store.get_all(user_id, k=10_000)
                        by_id = {m.memory.id: m for m in all_mems}
                        for cid in child_ids:
                            if cid in by_id:
                                results.append(by_id[cid])
        # Deduplicate by id, preserve first occurrence
        seen = set()
        deduped: list[MemorySearchResult] = []
        for r in results:
            if r.memory.id in seen:
                continue
            seen.add(r.memory.id)
            deduped.append(r)
        return deduped[: k * (order.index(max_level) + 1)]

    async def get_timeline(
        self,
        user_id: str,
        granularity: SummaryLevel,
        since: Optional[datetime] = None,
    ) -> list[SummaryNode]:
        all_mems = await self.store.get_all(user_id, k=10_000)
        nodes: list[SummaryNode] = []
        for r in all_mems:
            n = _node_from_memory(r.memory)
            if n is None or n.level != granularity:
                continue
            if since and n.time_range_end < since:
                continue
            nodes.append(n)
        nodes.sort(key=lambda x: x.time_range_start)
        return nodes
