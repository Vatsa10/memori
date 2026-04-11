import pytest

from memory_system.core.memory_models import Memory, MemorySearchResult, MemoryType
from memory_system.memory.profiles import build_user_profile


class TestUserProfiles:
    def test_build_from_memories(self):
        memories = [
            MemorySearchResult(
                memory=Memory(text="User lives in NYC", user_id="u1", memory_type=MemoryType.SEMANTIC, importance=0.7),
                score=1.0,
            ),
            MemorySearchResult(
                memory=Memory(text="User prefers dark mode", user_id="u1", memory_type=MemoryType.SEMANTIC, importance=0.5),
                score=1.0,
            ),
        ]
        profile = build_user_profile("u1", memories)
        assert profile.user_id == "u1"
        assert profile.memory_count == 2
        assert profile.summary  # Should have a summary

    def test_extracts_location(self):
        memories = [
            MemorySearchResult(
                memory=Memory(text="User lives in San Francisco", user_id="u1"),
                score=1.0,
            ),
        ]
        profile = build_user_profile("u1", memories)
        assert "location" in profile.properties
        assert "San Francisco" in profile.properties["location"]

    def test_extracts_preferences(self):
        memories = [
            MemorySearchResult(
                memory=Memory(text="User prefers morning deliveries", user_id="u1"),
                score=1.0,
            ),
        ]
        profile = build_user_profile("u1", memories)
        assert "preferences" in profile.properties
        assert any("morning" in p for p in profile.properties["preferences"])

    def test_empty_memories(self):
        profile = build_user_profile("u1", [])
        assert profile.user_id == "u1"
        assert profile.memory_count == 0
        assert profile.summary == ""

    def test_first_and_last_seen(self):
        from datetime import datetime, timezone
        memories = [
            MemorySearchResult(
                memory=Memory(text="A", user_id="u1", created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)),
                score=1.0,
            ),
            MemorySearchResult(
                memory=Memory(text="B", user_id="u1", created_at=datetime(2024, 6, 1, tzinfo=timezone.utc)),
                score=1.0,
            ),
        ]
        profile = build_user_profile("u1", memories)
        assert profile.first_seen.year == 2024
        assert profile.first_seen.month == 1
        assert profile.last_seen.month == 6
