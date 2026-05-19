"""Multi-modal ingestion: PDF, URL, image, audio → Chunks → Memory."""

from memory_system.ingestion.chunker import Chunk, SemanticChunker
from memory_system.ingestion.detect import detect_source_type

__all__ = [
    "Chunk",
    "SemanticChunker",
    "detect_source_type",
    "ingest_pdf",
    "ingest_url",
    "ingest_image",
    "ingest_audio",
]


# Lazy re-exports — heavy deps stay inside their modules
def __getattr__(name):
    if name == "ingest_pdf":
        from memory_system.ingestion.pdf import ingest_pdf
        return ingest_pdf
    if name == "ingest_url":
        from memory_system.ingestion.url import ingest_url
        return ingest_url
    if name == "ingest_image":
        from memory_system.ingestion.image import ingest_image
        return ingest_image
    if name == "ingest_audio":
        from memory_system.ingestion.audio import ingest_audio
        return ingest_audio
    raise AttributeError(name)
