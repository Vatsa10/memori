"""Context window builder — fit memories into a token budget."""

from memory_system.core.memory_models import MemorySearchResult, UserProfile


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word."""
    return int(len(text.split()) * 1.3)


def build_context_window(
    profile: UserProfile,
    memories: list[MemorySearchResult],
    query: str,
    token_budget: int = 2000,
) -> str:
    """
    Build a context string that fits within a token budget.

    Priority order:
    1. User profile summary (always included if fits)
    2. Highest-score memories (most relevant)
    3. Remaining memories by importance
    """
    sections = []
    tokens_used = 0

    # 1. User profile summary
    if profile.summary:
        profile_section = f"## User Profile\n{profile.summary}"
        profile_tokens = estimate_tokens(profile_section)
        if tokens_used + profile_tokens <= token_budget:
            sections.append(profile_section)
            tokens_used += profile_tokens

        # Add key properties
        if profile.properties:
            props = []
            for k, v in profile.properties.items():
                if k == "preferences" and isinstance(v, list):
                    props.append(f"- Preferences: {', '.join(v)}")
                else:
                    props.append(f"- {k.title()}: {v}")
            props_text = "\n".join(props)
            props_tokens = estimate_tokens(props_text)
            if tokens_used + props_tokens <= token_budget:
                sections.append(props_text)
                tokens_used += props_tokens

    # 2. Relevant memories (sorted by score, then importance)
    if memories:
        sorted_memories = sorted(
            memories,
            key=lambda r: (r.score, r.memory.importance),
            reverse=True,
        )

        memory_lines = []
        for r in sorted_memories:
            line = f"- {r.memory.text}"
            line_tokens = estimate_tokens(line)
            if tokens_used + line_tokens > token_budget:
                break
            memory_lines.append(line)
            tokens_used += line_tokens

        if memory_lines:
            sections.append("## Relevant Memories\n" + "\n".join(memory_lines))

    return "\n\n".join(sections) if sections else ""
