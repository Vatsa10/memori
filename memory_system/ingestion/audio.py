"""Audio ingestion via litellm transcription."""

import re
from pathlib import Path
from typing import Optional, Union

from memory_system.ingestion.chunker import Chunk, SemanticChunker


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


async def ingest_audio(
    path_or_bytes: Union[str, Path, bytes],
    *,
    model: str = "whisper-1",
    chunker: Optional[SemanticChunker] = None,
) -> list[Chunk]:
    """Transcribe audio via litellm; split on paragraph breaks then chunk."""
    try:
        from litellm import atranscription
    except ImportError:
        raise RuntimeError(
            "Audio ingestion requires 'litellm'. Install with: pip install memory-system"
        )

    if isinstance(path_or_bytes, (str, Path)):
        file_handle = open(path_or_bytes, "rb")
        filename = Path(path_or_bytes).name
    else:
        import io

        file_handle = io.BytesIO(path_or_bytes)
        filename = "<bytes>"

    try:
        response = await atranscription(model=model, file=file_handle)
    finally:
        try:
            file_handle.close()
        except Exception:
            pass

    text = getattr(response, "text", None) or response.get("text", "")  # type: ignore
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    use_chunker = chunker or SemanticChunker()
    chunks: list[Chunk] = []
    for paragraph in paragraphs:
        for c in use_chunker.chunk(
            paragraph,
            base_metadata={
                "source": "audio",
                "filename": filename,
                "transcription_model": model,
            },
        ):
            c.index = len(chunks)
            chunks.append(c)
    return chunks
