"""Semantic chunker: sentence-boundary aware, token-budget driven."""

import re
from dataclasses import dataclass, field
from typing import Any, Optional


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    index: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _approx_token_count(text: str) -> int:
    """Fallback when tiktoken is unavailable: words * 1.3 (English heuristic)."""
    return max(1, int(len(text.split()) * 1.3))


class SemanticChunker:
    """Greedy sentence-aware chunker bounded by max_tokens with overlap.

    Tries to use tiktoken; falls back to word-count heuristic if unavailable.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        sentence_boundaries: bool = True,
        tokenizer: str = "cl100k_base",
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.sentence_boundaries = sentence_boundaries
        self.tokenizer_name = tokenizer
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(self.tokenizer_name)
        except Exception:
            self._encoder = None
        return self._encoder

    def count_tokens(self, text: str) -> int:
        enc = self._get_encoder()
        if enc is None:
            return _approx_token_count(text)
        return len(enc.encode(text))

    def _split_units(self, text: str) -> list[str]:
        if not self.sentence_boundaries:
            return [text]
        sentences = _SENTENCE_SPLIT.split(text.strip())
        return [s for s in sentences if s]

    def chunk(
        self, text: str, *, base_metadata: Optional[dict] = None
    ) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []
        base_metadata = dict(base_metadata or {})

        units = self._split_units(text)
        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_tokens = 0
        idx = 0

        def flush(carry_overlap: bool):
            nonlocal buf, buf_tokens, idx
            if not buf:
                return
            chunk_text = " ".join(buf).strip()
            tokens = self.count_tokens(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    token_count=tokens,
                    metadata=dict(base_metadata),
                )
            )
            idx += 1
            if carry_overlap and self.overlap_tokens > 0:
                # Carry trailing sentences whose combined token count ≤ overlap_tokens
                carried: list[str] = []
                carried_tokens = 0
                for s in reversed(buf):
                    t = self.count_tokens(s)
                    if carried_tokens + t > self.overlap_tokens:
                        break
                    carried.insert(0, s)
                    carried_tokens += t
                buf = carried
                buf_tokens = carried_tokens
            else:
                buf = []
                buf_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)
            if unit_tokens > self.max_tokens:
                # Sentence alone exceeds budget — flush current then emit it solo
                flush(carry_overlap=False)
                chunks.append(
                    Chunk(
                        text=unit,
                        index=idx,
                        token_count=unit_tokens,
                        metadata=dict(base_metadata),
                    )
                )
                idx += 1
                continue
            if buf_tokens + unit_tokens > self.max_tokens:
                flush(carry_overlap=True)
            buf.append(unit)
            buf_tokens += unit_tokens

        flush(carry_overlap=False)
        return chunks
