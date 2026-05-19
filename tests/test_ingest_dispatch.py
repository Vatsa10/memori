"""detect_source_type dispatch + ingest_document end-to-end (mocked extractors)."""

import pytest

from memory_system.ingestion.detect import detect_source_type


def test_detect_pdf_extension():
    assert detect_source_type("report.PDF") == "pdf"
    assert detect_source_type("docs/x.pdf") == "pdf"


def test_detect_url():
    assert detect_source_type("https://example.com/page") == "url"
    assert detect_source_type("HTTP://example.com") == "url"


def test_detect_image_extensions():
    for ext in ["png", "jpg", "jpeg", "webp"]:
        assert detect_source_type(f"pic.{ext}") == "image"


def test_detect_audio_extensions():
    for ext in ["mp3", "wav", "m4a"]:
        assert detect_source_type(f"sound.{ext}") == "audio"


def test_detect_plain_text_fallback():
    assert detect_source_type("just a string") == "text"
    assert detect_source_type("file_with_no_ext") == "text"


def test_detect_pdf_magic_bytes():
    assert detect_source_type(b"%PDF-1.7\n...") == "pdf"


def test_detect_image_magic_bytes():
    assert detect_source_type(b"\xff\xd8\xff\xe0jpeg") == "image"
    assert detect_source_type(b"\x89PNG\r\n\x1a\nfoo") == "image"


def test_detect_audio_magic_bytes():
    assert detect_source_type(b"RIFF....WAVE") == "audio"
    assert detect_source_type(b"ID3\x04\x00") == "audio"


def test_detect_bytes_fallback_text():
    assert detect_source_type(b"hello world") == "text"
