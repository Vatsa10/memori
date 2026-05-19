"""Hierarchical summary tree: turn/session/day/month rollups."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from memory_system.core.memory_models import SummaryLevel
from memory_system.memory.memory import Memory as MemoryAPI
from memory_system.memory.summary_tree import SummaryTreeManager
from memory_system.providers.in_memory_stores import InMemoryMemoryStore


@pytest.mark.asyncio
async def test_summarize_turn_creates_summary_memory():
    store = InMemoryMemoryStore()
    fake_llm = AsyncMock(return_value="User asked about return policy and got an answer.")

    mgr = SummaryTreeManager(memory_store=store, llm_fn=fake_llm)
    node = await mgr.summarize_turn(
        user_msg="What is the return policy for headphones I bought last week?",
        assistant_msg="Returns accepted within 30 days of purchase with receipt.",
        user_id="u1",
        turn_id="turn-1",
        session_id="sess-1",
    )

    assert node is not None
    assert node.level == SummaryLevel.TURN
    assert "return policy" in node.content.lower()

    # Stored as Memory with summary_level metadata
    all_mems = await store.get_all("u1", k=100)
    assert any(m.memory.metadata.get("summary_level") == "turn" for m in all_mems)


@pytest.mark.asyncio
async def test_short_turn_skipped():
    store = InMemoryMemoryStore()
    fake_llm = AsyncMock(return_value="x")
    mgr = SummaryTreeManager(memory_store=store, llm_fn=fake_llm)

    node = await mgr.summarize_turn(
        user_msg="hi", assistant_msg="hello",
        user_id="u1", turn_id="t",
    )
    assert node is None
    fake_llm.assert_not_called()


@pytest.mark.asyncio
async def test_default_recall_excludes_summaries():
    """Memory.recall() default should not surface summary nodes."""
    store = InMemoryMemoryStore()
    fake_llm = AsyncMock(return_value="A summary about delivery times and tea orders.")
    mgr = SummaryTreeManager(memory_store=store, llm_fn=fake_llm)

    # Plant a turn summary + a regular memory with shared keywords
    await mgr.summarize_turn(
        user_msg="When can I expect my delivery of green tea this week?",
        assistant_msg="Standard shipping arrives within three days for tea orders.",
        user_id="u1",
        turn_id="t1",
        session_id="s1",
    )
    from memory_system.core.memory_models import Memory
    await store.add(Memory(text="user delivery preferences favor morning windows for tea", user_id="u1"))

    api = MemoryAPI(store=store)
    results = await api.recall("delivery", user_id="u1", k=10)
    assert all(not r.memory.metadata.get("summary_level") for r in results)
    # And summaries DO surface when requested
    incl = await api.recall("delivery", user_id="u1", k=10, include_summaries=True)
    assert any(r.memory.metadata.get("summary_level") for r in incl)


@pytest.mark.asyncio
async def test_rollup_session_aggregates_child_turns():
    store = InMemoryMemoryStore()
    fake_llm = AsyncMock()
    fake_llm.side_effect = [
        # turn 1 summary
        "Turn one: discussed delivery times.",
        # turn 2 summary
        "Turn two: discussed return policy.",
        # session rollup
        "SUMMARY: Session covered delivery and returns.\nFACTS:\n- delivery in 3 days\n- 30-day return window",
    ]
    mgr = SummaryTreeManager(memory_store=store, llm_fn=fake_llm)

    await mgr.summarize_turn(
        user_msg="Long enough message about delivery please respond informatively.",
        assistant_msg="Delivery details here for your reference and convenience.",
        user_id="u1", turn_id="t1", session_id="s1",
    )
    await mgr.summarize_turn(
        user_msg="Another long enough message about returns please respond.",
        assistant_msg="Returns are accepted within thirty days of purchase.",
        user_id="u1", turn_id="t2", session_id="s1",
    )
    session_node = await mgr.rollup_session("s1", "u1")
    assert session_node is not None
    assert session_node.level == SummaryLevel.SESSION
    assert len(session_node.child_ids) == 2
    assert any("delivery" in f.lower() or "return" in f.lower()
               for f in session_node.key_facts)


@pytest.mark.asyncio
async def test_search_hierarchical_returns_summaries():
    store = InMemoryMemoryStore()
    fake_llm = AsyncMock()
    fake_llm.side_effect = [
        "User asked about delivery options for tea purchase.",
    ]
    mgr = SummaryTreeManager(memory_store=store, llm_fn=fake_llm)

    await mgr.summarize_turn(
        user_msg="Long question about delivery options for my tea order today.",
        assistant_msg="Here are the available delivery options for tea purchases here.",
        user_id="u1", turn_id="t1", session_id="s1",
    )

    results = await mgr.search_hierarchical("delivery tea", "u1", k=5)
    assert len(results) >= 1
    assert any(r.memory.metadata.get("summary_level") for r in results)


@pytest.mark.asyncio
async def test_get_timeline_orders_by_time():
    store = InMemoryMemoryStore()
    fake_llm = AsyncMock(return_value="A summary sentence.")
    mgr = SummaryTreeManager(memory_store=store, llm_fn=fake_llm)

    await mgr.summarize_turn(
        user_msg="Message one with enough characters to clear the threshold.",
        assistant_msg="Response one with enough length to be summarized.",
        user_id="u1", turn_id="t1",
    )
    await mgr.summarize_turn(
        user_msg="Message two with enough characters to clear the threshold.",
        assistant_msg="Response two with enough length to be summarized.",
        user_id="u1", turn_id="t2",
    )
    timeline = await mgr.get_timeline("u1", SummaryLevel.TURN)
    assert len(timeline) == 2
    assert timeline[0].time_range_start <= timeline[1].time_range_start
