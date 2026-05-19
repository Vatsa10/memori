"""Smart memory ops: ADD / UPDATE / MERGE / DELETE / NOOP via LLM judge."""

from unittest.mock import AsyncMock

import pytest

from memory_system.core.memory_models import Memory, MemorySearchResult, MemoryType
from memory_system.memory.smart_ops import (
    MemoryAction,
    MemoryDecision,
    execute_decision,
    judge_memory_op,
)
from memory_system.providers.in_memory_stores import InMemoryMemoryStore


# ---------- judge_memory_op ----------

class TestJudgeMemoryOp:
    @pytest.mark.asyncio
    async def test_empty_candidates_short_circuits_to_add(self):
        mem = Memory(text="user likes tea", user_id="u1")
        decision = await judge_memory_op(
            mem, candidates=[], llm_fn=AsyncMock(), model="any"
        )
        assert decision.action == MemoryAction.ADD
        assert decision.new_text == "user likes tea"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_add(self, monkeypatch):
        mem = Memory(text="x", user_id="u1")
        candidates = [
            MemorySearchResult(memory=Memory(text="old", user_id="u1"), score=0.9)
        ]

        # Force the inner import path to raise
        import sys
        sys.modules.pop("instructor", None)
        sys.modules.pop("litellm", None)
        monkeypatch.setitem(sys.modules, "instructor", None)

        decision = await judge_memory_op(
            mem, candidates, llm_fn=AsyncMock(), model="any"
        )
        assert decision.action == MemoryAction.ADD
        assert "judge failed" in decision.reason


# ---------- execute_decision ----------

@pytest.mark.asyncio
async def test_execute_noop_does_nothing():
    store = InMemoryMemoryStore()
    target = Memory(text="existing", user_id="u1")
    await store.add(target)

    decision = MemoryDecision(action=MemoryAction.NOOP)
    new = Memory(text="redundant", user_id="u1")
    result = await execute_decision(decision, new, candidates=[], store=store)

    assert result is None
    assert (await store.get_all("u1"))[0].memory.id == target.id


@pytest.mark.asyncio
async def test_execute_add_stores_new_fact():
    store = InMemoryMemoryStore()
    new = Memory(text="user likes tea", user_id="u1")
    decision = MemoryDecision(action=MemoryAction.ADD, new_text="user likes tea")

    result = await execute_decision(decision, new, candidates=[], store=store)
    assert result is not None
    assert result.id == new.id
    all_mems = await store.get_all("u1")
    assert len(all_mems) == 1


@pytest.mark.asyncio
async def test_execute_update_invalidates_target_and_creates_successor():
    store = InMemoryMemoryStore()
    target = Memory(text="user lives in NYC", user_id="u1", importance=0.7)
    await store.add(target)
    candidates = [MemorySearchResult(memory=target, score=0.9)]

    new = Memory(text="user lives in SF", user_id="u1", importance=0.5)
    decision = MemoryDecision(
        action=MemoryAction.UPDATE,
        target_id=target.id,
        new_text="user lives in SF",
        reason="moved",
    )
    result = await execute_decision(decision, new, candidates, store)

    assert result is not None
    # New fact carries supersedes link
    assert result.metadata["supersedes"] == target.id
    # New fact keeps max(importance) = 0.7
    assert result.importance == 0.7

    # Default search excludes invalidated → only the successor
    visible = await store.search("lives", user_id="u1")
    ids = {r.memory.id for r in visible}
    assert result.id in ids
    assert target.id not in ids

    # include_invalidated reveals both, and target has superseded_by + valid_to
    all_incl = await store.get_all("u1", include_invalidated=True)
    by_id = {r.memory.id: r.memory for r in all_incl}
    assert by_id[target.id].valid_to is not None
    assert by_id[target.id].superseded_by == result.id


@pytest.mark.asyncio
async def test_execute_merge_combines_when_new_text_missing():
    store = InMemoryMemoryStore()
    target = Memory(text="likes tea", user_id="u1", importance=0.6)
    await store.add(target)
    candidates = [MemorySearchResult(memory=target, score=0.8)]

    new = Memory(text="prefers green tea in the morning", user_id="u1", importance=0.4)
    decision = MemoryDecision(
        action=MemoryAction.MERGE, target_id=target.id
    )  # no new_text → execute combines

    result = await execute_decision(decision, new, candidates, store)
    assert result is not None
    assert "likes tea" in result.text and "green tea" in result.text
    assert result.importance == 0.6  # max of the two
    assert target.id in result.metadata["merged_from"]


@pytest.mark.asyncio
async def test_execute_delete_invalidates_without_successor():
    store = InMemoryMemoryStore()
    target = Memory(text="wrong fact", user_id="u1")
    await store.add(target)
    candidates = [MemorySearchResult(memory=target, score=0.95)]

    decision = MemoryDecision(action=MemoryAction.DELETE, target_id=target.id)
    new = Memory(text="contradicts", user_id="u1")
    result = await execute_decision(decision, new, candidates, store)

    assert result is None
    visible = await store.get_all("u1")
    assert all(r.memory.id != target.id for r in visible)


@pytest.mark.asyncio
async def test_execute_with_invalid_target_id_falls_back_to_add():
    store = InMemoryMemoryStore()
    new = Memory(text="orphan", user_id="u1")
    decision = MemoryDecision(
        action=MemoryAction.UPDATE, target_id="does-not-exist", new_text="x"
    )
    result = await execute_decision(decision, new, candidates=[], store=store)
    # Bad target → fallback ADD so we don't lose the fact
    assert result is not None
    assert result.id == new.id


# ---------- Memory.remember integration ----------

@pytest.mark.asyncio
async def test_remember_routes_through_smart_ops_when_enabled(monkeypatch):
    from memory_system.memory.memory import Memory as MemoryAPI

    captured = {}

    async def fake_extract(**kwargs):
        from memory_system.core.memory_models import MemoryExtractionResult
        return MemoryExtractionResult(
            memories=[
                Memory(text="user lives in SF", user_id=kwargs["user_id"]),
            ]
        )

    async def fake_judge(new_fact, candidates, llm_fn, model, prompt_template=None):
        captured["candidates"] = candidates
        return MemoryDecision(
            action=MemoryAction.ADD, new_text=new_fact.text, reason="ok"
        )

    monkeypatch.setattr(
        "memory_system.memory.memory.extract_memories", fake_extract
    )
    monkeypatch.setattr(
        "memory_system.memory.memory.judge_memory_op", fake_judge
    )

    store = InMemoryMemoryStore()
    api = MemoryAPI(
        store=store,
        llm_fn=AsyncMock(),
        enable_smart_ops=True,
    )
    result = await api.remember(
        messages=[
            {"role": "user", "content": "I just moved to SF"},
            {"role": "assistant", "content": "Got it."},
        ],
        user_id="u1",
    )

    assert len(result.memories) == 1
    assert "candidates" in captured


@pytest.mark.asyncio
async def test_smart_ops_disabled_uses_legacy_dedup_path(monkeypatch):
    """Default enable_smart_ops=False must not call the judge."""
    from memory_system.memory.memory import Memory as MemoryAPI

    async def fake_extract(**kwargs):
        from memory_system.core.memory_models import MemoryExtractionResult
        return MemoryExtractionResult(
            memories=[Memory(text="fact", user_id=kwargs["user_id"])]
        )

    called = {"judge": 0}

    async def fake_judge(*args, **kwargs):
        called["judge"] += 1
        return MemoryDecision(action=MemoryAction.NOOP)

    monkeypatch.setattr(
        "memory_system.memory.memory.extract_memories", fake_extract
    )
    monkeypatch.setattr(
        "memory_system.memory.memory.judge_memory_op", fake_judge
    )

    api = MemoryAPI(store=InMemoryMemoryStore(), llm_fn=AsyncMock())
    await api.remember(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        user_id="u1",
    )
    assert called["judge"] == 0


# ---------- Memory.recall_at ----------

@pytest.mark.asyncio
async def test_recall_at_returns_point_in_time_state():
    from datetime import datetime, timedelta, timezone
    from memory_system.memory.memory import Memory as MemoryAPI

    store = InMemoryMemoryStore()
    t_old = datetime.now(timezone.utc) - timedelta(hours=2)
    t_mid = datetime.now(timezone.utc) - timedelta(hours=1)
    old_fact = Memory(
        text="user lives in NYC", user_id="u1",
        valid_from=t_old, valid_to=t_mid,
    )
    new_fact = Memory(
        text="user lives in SF", user_id="u1",
        valid_from=t_mid,
    )
    await store.add(old_fact)
    await store.add(new_fact)

    api = MemoryAPI(store=store)

    before = await api.recall_at("lives", "u1", as_of=t_old + timedelta(minutes=10))
    assert any("NYC" in r.memory.text for r in before)
    assert not any("SF" in r.memory.text for r in before)

    after = await api.recall_at("lives", "u1", as_of=datetime.now(timezone.utc))
    assert any("SF" in r.memory.text for r in after)
