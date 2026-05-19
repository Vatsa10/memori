"""URL ingestion via httpx + trafilatura (bs4 fallback)."""

from datetime import datetime, timezone
from typing import Optional

from memory_system.ingestion.chunker import Chunk, SemanticChunker


def _require(pkg: str):
    raise RuntimeError(
        f"URL ingestion requires '{pkg}'. Install with: pip install memory-system[ingestion]"
    )


def _extract_main_content(html: str, url: str) -> tuple[str, Optional[str]]:
    """Return (main_text, title). Tries trafilatura, falls back to BeautifulSoup."""
    try:
        import trafilatura

        text = trafilatura.extract(html, url=url, include_comments=False) or ""
        meta = trafilatura.extract_metadata(html)
        title = getattr(meta, "title", None) if meta else None
        if text.strip():
            return text, title
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        text = soup.get_text(separator=" ", strip=True)
        return text, title
    except ImportError:
        _require("trafilatura or beautifulsoup4")


async def ingest_url(
    url: str,
    *,
    chunker: SemanticChunker,
    timeout: float = 30.0,
    headers: Optional[dict] = None,
) -> list[Chunk]:
    try:
        import httpx
    except ImportError:
        _require("httpx")

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers=headers or {})
        response.raise_for_status()
        html = response.text

    text, title = _extract_main_content(html, url)
    text = (text or "").strip()
    if not text:
        return []

    base_meta = {
        "source": "url",
        "url": url,
        "title": title,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return chunker.chunk(text, base_metadata=base_meta)
