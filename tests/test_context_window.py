import pytest

from memory_system.core.memory_models import Memory, MemorySearchResult, MemoryType, UserProfile
from memory_system.memory.context import build_context_window, estimate_tokens


class TestContextWindow:
    def test_includes_profile(self):
        profile = UserProfile(
            user_id="u1",
            summary="Active user who prefers morning deliveries.",
            properties={"location": "NYC"},
        )
        context = build_context_window(profile, [], "test", token_budget=500)
        assert "User Profile" in context
        assert "morning deliveries" in context

    def test_includes_memories(self):
        profile = UserProfile(user_id="u1")
        memories = [
            MemorySearchResult(
                memory=Memory(text="Likes coffee", user_id="u1", importance=0.8),
                score=0.9,
            ),
        ]
        context = build_context_window(profile, memories, "coffee", token_budget=500)
        assert "coffee" in context

    def test_respects_token_budget(self):
        profile = UserProfile(user_id="u1", summary="Short summary.")
        # Create many memories that exceed budget
        memories = [
            MemorySearchResult(
                memory=Memory(text=f"Memory number {i} with lots of text to fill up tokens", user_id="u1"),
                score=0.5,
            )
            for i in range(50)
        ]
        context = build_context_window(profile, memories, "test", token_budget=100)
        tokens = estimate_tokens(context)
        assert tokens <= 120  # Allow small overshoot from section headers

    def test_empty_returns_empty(self):
        profile = UserProfile(user_id="u1")
        context = build_context_window(profile, [], "test", token_budget=500)
        assert context == ""

    def test_prioritizes_high_score(self):
        profile = UserProfile(user_id="u1")
        memories = [
            MemorySearchResult(
                memory=Memory(text="Low relevance", user_id="u1", importance=0.3),
                score=0.3,
            ),
            MemorySearchResult(
                memory=Memory(text="High relevance", user_id="u1", importance=0.9),
                score=0.95,
            ),
        ]
        context = build_context_window(profile, memories, "test", token_budget=500)
        # High relevance should appear first
        assert context.index("High relevance") < context.index("Low relevance")
