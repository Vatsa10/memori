"""Smart memory operations: LLM-judged ADD/UPDATE/MERGE/DELETE/NOOP.

Replaces append-and-dedup with a reasoning layer that decides what to do
with each new extracted fact given the existing memories it overlaps with.

Usage (inside Memory.remember):
    candidates = await store.search(new_fact.text, user_id, k=5)
    decision = await judge_memory_op(new_fact, candidates, llm_fn, model)
    result = await execute_decision(decision, new_fact, candidates, store)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from memory_system.core.memory_models import Memory, MemorySearchResult


class MemoryAction(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class MemoryDecision(BaseModel):
    action: MemoryAction
    target_id: Optional[str] = None  # required for UPDATE/MERGE/DELETE
    new_text: Optional[str] = None  # required for ADD/UPDATE/MERGE
    reason: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


SMART_OPS_PROMPT = """You are a memory curator. Decide what to do with a new fact
given existing memories about the same user.

New fact:
  text: {new_text}
  importance: {new_importance}

Existing candidate memories (up to 5, most similar first):
{candidates_block}

Choose one action:
- ADD: the new fact is genuinely new information not represented above.
- UPDATE: the new fact supersedes one existing memory (e.g., user moved cities,
  changed preferences). Set target_id to the obsolete memory and new_text to
  the replacement.
- MERGE: the new fact and one existing memory describe the same thing but with
  complementary details. Set target_id and new_text to the combined statement.
- DELETE: the new fact directly contradicts an existing memory in a way that
  invalidates it without replacement. Set target_id.
- NOOP: the new fact is redundant, trivial, or already captured. No change.

Respond with a JSON object matching MemoryDecision."""


def _format_candidates(candidates: list[MemorySearchResult]) -> str:
    if not candidates:
        return "  (none — no similar memories exist)"
    lines = []
    for i, r in enumerate(candidates, 1):
        m = r.memory
        lines.append(
            f"  {i}. id={m.id} score={r.score:.2f} importance={m.importance:.2f}"
            f"\n     text: {m.text}"
        )
    return "\n".join(lines)


async def judge_memory_op(
    new_fact: Memory,
    candidates: list[MemorySearchResult],
    llm_fn: Callable,
    model: str = "deepseek/deepseek-v4-flash",
    prompt_template: Optional[str] = None,
) -> MemoryDecision:
    """Ask the LLM to choose ADD/UPDATE/MERGE/DELETE/NOOP for this new fact."""
    if not candidates:
        return MemoryDecision(
            action=MemoryAction.ADD,
            new_text=new_fact.text,
            reason="no existing candidates",
        )

    template = prompt_template or SMART_OPS_PROMPT
    prompt = template.format(
        new_text=new_fact.text,
        new_importance=new_fact.importance,
        candidates_block=_format_candidates(candidates),
    )

    try:
        import instructor
        from litellm import acompletion

        client = instructor.from_litellm(acompletion)
        decision = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=MemoryDecision,
            temperature=0.0,
            max_retries=1,
        )
        return decision
    except Exception:
        # Failsafe: don't crash chat — default to ADD (which preserves the old
        # behavior modulo the existing dedup layer).
        return MemoryDecision(
            action=MemoryAction.ADD,
            new_text=new_fact.text,
            reason="judge failed; defaulted to ADD",
            confidence=0.5,
        )


def _find_candidate(
    target_id: Optional[str], candidates: list[MemorySearchResult]
) -> Optional[Memory]:
    if not target_id:
        return None
    for r in candidates:
        if r.memory.id == target_id:
            return r.memory
    return None


async def execute_decision(
    decision: MemoryDecision,
    new_fact: Memory,
    candidates: list[MemorySearchResult],
    store: Any,
) -> Optional[Memory]:
    """Apply a MemoryDecision against the store. Returns the resulting Memory or None."""
    now = datetime.now(timezone.utc)

    if decision.action == MemoryAction.NOOP:
        return None

    if decision.action == MemoryAction.ADD:
        if decision.new_text:
            new_fact.text = decision.new_text
        await store.add(new_fact)
        return new_fact

    target = _find_candidate(decision.target_id, candidates)
    if target is None:
        # Bad LLM output — fall back to ADD so we don't lose the fact.
        await store.add(new_fact)
        return new_fact

    if decision.action == MemoryAction.DELETE:
        await _invalidate(store, target.id, now)
        return None

    # UPDATE / MERGE — invalidate target, create successor.
    text = decision.new_text or new_fact.text
    if decision.action == MemoryAction.MERGE and not decision.new_text:
        text = f"{target.text}; {new_fact.text}"

    successor = new_fact.model_copy(
        update={
            "text": text,
            "importance": max(target.importance, new_fact.importance),
            "valid_from": now,
        }
    )

    if decision.action == MemoryAction.UPDATE:
        successor.metadata = {**successor.metadata, "supersedes": target.id}
    else:  # MERGE
        merged = list(target.metadata.get("merged_from", []))
        merged.append(target.id)
        successor.metadata = {**successor.metadata, "merged_from": merged}

    await _invalidate(store, target.id, now, superseded_by=successor.id)
    await store.add(successor)
    return successor


async def _invalidate(
    store: Any,
    memory_id: str,
    valid_to: datetime,
    superseded_by: Optional[str] = None,
) -> None:
    """Call store.invalidate if available; otherwise simulate via update."""
    invalidate = getattr(store, "invalidate", None)
    if invalidate is not None:
        await invalidate(memory_id, valid_to=valid_to, superseded_by=superseded_by)
        return
    # Fallback for stores that don't implement invalidate yet
    existing = getattr(store, "_memories", {}).get(memory_id)
    if existing is not None:
        existing.valid_to = valid_to
        if superseded_by is not None:
            existing.superseded_by = superseded_by
