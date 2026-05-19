"""Image ingestion via vision LLM (OCR + scene description)."""

import base64
from pathlib import Path
from typing import Callable, Optional, Union

from memory_system.ingestion.chunker import Chunk


DEFAULT_VISION_PROMPT = (
    "Describe this image in detail. Include any visible text (OCR), objects, "
    "scene, and data from charts or tables. Be precise and factual."
)


def _to_data_url(source: Union[str, Path, bytes]) -> str:
    if isinstance(source, bytes):
        data = source
        ext = "png"
    else:
        p = Path(source)
        data = p.read_bytes()
        ext = (p.suffix.lstrip(".") or "png").lower()
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


async def ingest_image(
    path_or_bytes: Union[str, Path, bytes],
    *,
    llm_fn: Callable,
    vision_model: str = "openai/gpt-4o-mini",
    prompt: Optional[str] = None,
) -> list[Chunk]:
    """Send the image to a vision-capable LLM and capture its description."""
    data_url = _to_data_url(path_or_bytes)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or DEFAULT_VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    description = await llm_fn(model=vision_model, messages=messages)

    if not isinstance(description, str):
        description = str(description)
    description = description.strip()
    if not description:
        return []

    filename = (
        Path(path_or_bytes).name
        if isinstance(path_or_bytes, (str, Path))
        else "<bytes>"
    )
    from memory_system.ingestion.chunker import SemanticChunker  # local to avoid cycle

    tmp = SemanticChunker()
    return [
        Chunk(
            text=description,
            index=0,
            token_count=tmp.count_tokens(description),
            metadata={
                "source": "image",
                "filename": filename,
                "vision_model": vision_model,
            },
        )
    ]
