"""PDF ingestion via pypdf (default) or pymupdf (better quality)."""

from pathlib import Path
from typing import Literal, Optional, Union

from memory_system.ingestion.chunker import Chunk, SemanticChunker


def _require(pkg: str, extra: str = "ingestion"):
    raise RuntimeError(
        f"PDF ingestion requires '{pkg}'. Install with: pip install memory-system[{extra}]"
    )


def _extract_pages_pypdf(source: Union[str, Path, bytes], password: Optional[str]):
    try:
        from pypdf import PdfReader
    except ImportError:
        _require("pypdf")

    if isinstance(source, (str, Path)):
        reader = PdfReader(str(source))
    else:
        import io
        reader = PdfReader(io.BytesIO(source))

    if password is not None and reader.is_encrypted:
        reader.decrypt(password)

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, text))
    return pages


def _extract_pages_pymupdf(source: Union[str, Path, bytes], password: Optional[str]):
    try:
        import fitz  # pymupdf
    except ImportError:
        _require("pymupdf", extra="ingestion-pro")

    if isinstance(source, (str, Path)):
        doc = fitz.open(str(source))
    else:
        doc = fitz.open(stream=source, filetype="pdf")

    if password is not None and doc.is_encrypted:
        doc.authenticate(password)

    pages = []
    for i, page in enumerate(doc, start=1):
        pages.append((i, page.get_text("text") or ""))
    doc.close()
    return pages


async def ingest_pdf(
    path_or_bytes: Union[str, Path, bytes],
    *,
    chunker: SemanticChunker,
    password: Optional[str] = None,
    backend: Literal["auto", "pypdf", "pymupdf"] = "auto",
) -> list[Chunk]:
    """Extract text per-page, concatenate with page markers, then chunk."""
    if backend == "pymupdf":
        pages = _extract_pages_pymupdf(path_or_bytes, password)
    elif backend == "pypdf":
        pages = _extract_pages_pypdf(path_or_bytes, password)
    else:  # auto: try pymupdf, fall back to pypdf
        try:
            import fitz  # noqa: F401
            pages = _extract_pages_pymupdf(path_or_bytes, password)
        except ImportError:
            pages = _extract_pages_pypdf(path_or_bytes, password)

    filename = (
        Path(path_or_bytes).name
        if isinstance(path_or_bytes, (str, Path))
        else "<bytes>"
    )

    chunks: list[Chunk] = []
    for page_num, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            continue
        page_chunks = chunker.chunk(
            page_text,
            base_metadata={
                "source": "pdf",
                "filename": filename,
                "page_start": page_num,
                "page_end": page_num,
            },
        )
        for c in page_chunks:
            c.index = len(chunks)
            chunks.append(c)
    return chunks
