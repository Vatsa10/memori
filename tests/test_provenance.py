"""Provenance stamping: source_text, turn_id, confidence, extractor_model."""

from unittest.mock import AsyncMock

import pytest

from memory_system.core.memory_models import Memory
from memory_system.memory.extractor import (
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    ExtractionOutput,
    extract_memories,
)
from memory_system.memory.memory import Memory as MemoryAPI
from memory_system.providers.in_memory_stores import (
    InMemoryGraphStore,
    InMemoryMemoryStore,
)


def test_memory_defaults_provenance_to_safe_values():
    mem = Memory(text="x", user_id="u1")
    assert mem.source_text is None
    assert mem.turn_id is None
    assert mem.confidence == 1.0
    assert mem.extractor_model is None


@pytest.mark.asyncio
async def test_extract_memories_stamps_provenance(monkeypatch):
    """extract_memories must populate source_text, turn_id, confidence, extractor_model."""

    captured_output = ExtractionOutput(
        facts=[
            ExtractedFact(text="user likes tea", confidence=0.9),
            ExtractedFact(text="user dislikes coffee", confidence=0.6),
        ],
    )

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    return captured_output

    monkeypatch.setattr(
        "memory_system.memory.extractor.instructor",
        type("M", (), {"from_litellm": staticmethod(lambda _: FakeClient)})(),
        raising=False,
    )

    # patch the inner import: instructor + litellm
    import sys
    fake_instructor = type("M", (), {"from_litellm": staticmethod(lambda _: FakeClient)})()
    fake_litellm = type("M", (), {"acompletion": AsyncMock()})()
    sys.modules["instructor"] = fake_instructor
    sys.modules["litellm"] = fake_litellm

    result = await extract_memories(
        user_message="I love tea but hate coffee.",
        assistant_response="Noted.",
        user_id="u1",
        llm_fn=AsyncMock(),
        model="groq/llama-3.1-8b-instant",
        turn_id="turn-abc",
    )

    assert len(result.memories) == 2
    for mem in result.memories:
        assert mem.source_text == "I love tea but hate coffee."
        assert mem.turn_id == "turn-abc"
        assert mem.extractor_model == "groq/llama-3.1-8b-instant"
        assert 0.0 <= mem.confidence <= 1.0
    assert result.memories[0].confidence == 0.9
    assert result.memories[1].confidence == 0.6


@pytest.mark.asyncio
async def test_remember_assigns_one_turn_id_across_extracted_memories(monkeypatch):
    """A single remember() call should share one turn_id across all extracted memories."""

    async def fake_extract(**kwargs):
        from memory_system.core.memory_models import (
            Memory as MemModel,
            MemoryExtractionResult,
            MemoryType,
        )
        tid = kwargs["turn_id"]
        return MemoryExtractionResult(
            memories=[
                MemModel(
                    text=f"fact {i}",
                    user_id=kwargs["user_id"],
                    memory_type=MemoryType.SEMANTIC,
                    turn_id=tid,
                    source_text=kwargs["source_text"],
                    extractor_model=kwargs["model"],
                )
                for i in range(3)
            ]
        )

    monkeypatch.setattr(
        "memory_system.memory.memory.extract_memories", fake_extract
    )

    api = MemoryAPI(
        store=InMemoryMemoryStore(),
        graph=InMemoryGraphStore(),
        llm_fn=AsyncMock(),
    )

    result = await api.remember(
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        user_id="u1",
    )

    assert len(result.memories) == 3
    turn_ids = {m.turn_id for m in result.memories}
    assert len(turn_ids) == 1
    assert next(iter(turn_ids)) is not None


@pytest.mark.asyncio
async def test_extractor_clamps_confidence_to_unit_interval(monkeypatch):
    captured_output = ExtractionOutput(
        facts=[ExtractedFact(text="x", confidence=1.5)],
    )

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    return captured_output

    import sys
    sys.modules["instructor"] = type("M", (), {"from_litellm": staticmethod(lambda _: FakeClient)})()
    sys.modules["litellm"] = type("M", (), {"acompletion": AsyncMock()})()

    result = await extract_memories(
        user_message="x", assistant_response="y", user_id="u1",
        llm_fn=AsyncMock(), model="m",
    )
    assert result.memories[0].confidence == 1.0
