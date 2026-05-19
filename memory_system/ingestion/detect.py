"""Source-type detection for ingestion dispatch."""

import re
from pathlib import Path
from typing import Literal, Union

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_PDF_EXTS = {".pdf"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

SourceType = Literal["pdf", "url", "image", "audio", "text"]


def detect_source_type(value: Union[str, bytes, Path]) -> SourceType:
    if isinstance(value, bytes):
        head = value[:12]
        if head.startswith(b"%PDF"):
            return "pdf"
        if head.startswith(b"\xff\xd8\xff") or head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image"
        if head.startswith(b"RIFF") or head.startswith(b"ID3") or head[:2] == b"\xff\xfb":
            return "audio"
        return "text"

    s = str(value)
    if _URL_RE.match(s):
        return "url"

    p = Path(s)
    ext = p.suffix.lower()
    if ext in _PDF_EXTS:
        return "pdf"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "text"
