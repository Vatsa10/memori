"""SemanticChunker: token-bounded greedy packing with overlap."""

import pytest

from memory_system.ingestion.chunker import Chunk, SemanticChunker


def test_empty_input_returns_empty():
    assert SemanticChunker().chunk("") == []
    assert SemanticChunker().chunk("   ") == []


def test_single_sentence_one_chunk():
    chunker = SemanticChunker(max_tokens=100)
    chunks = chunker.chunk("A short sentence.")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == "A short sentence."
    assert chunks[0].token_count > 0


def test_multiple_chunks_when_exceeding_budget():
    # Use very small budget to force multiple chunks
    chunker = SemanticChunker(max_tokens=5, overlap_tokens=0)
    text = "Sentence one is here. Sentence two follows. Sentence three is last."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # Indexes are contiguous
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_trailing_context():
    chunker = SemanticChunker(max_tokens=8, overlap_tokens=4)
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # The second chunk should share some text with the first
    overlap = set(chunks[0].text.lower().split()) & set(chunks[1].text.lower().split())
    assert len(overlap) > 0


def test_metadata_propagates_to_each_chunk():
    chunker = SemanticChunker(max_tokens=5, overlap_tokens=0)
    chunks = chunker.chunk(
        "One. Two. Three. Four. Five.",
        base_metadata={"source": "test", "doc_id": 42},
    )
    assert len(chunks) >= 2
    assert all(c.metadata["source"] == "test" for c in chunks)
    assert all(c.metadata["doc_id"] == 42 for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        SemanticChunker(max_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        SemanticChunker(max_tokens=0)


def test_oversized_sentence_emitted_alone():
    chunker = SemanticChunker(max_tokens=3, overlap_tokens=0)
    text = "ok. supercalifragilisticexpialidocious overload sentence here. ok."
    chunks = chunker.chunk(text)
    # Oversized sentence is emitted intact
    assert any(c.token_count > 3 for c in chunks)
