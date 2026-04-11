"""User profile management — auto-build profiles from memories."""

from memory_system.core.memory_models import (
    MemorySearchResult,
    MemoryType,
    UserProfile,
)


def build_user_profile(
    user_id: str,
    memories: list[MemorySearchResult],
) -> UserProfile:
    """Build a user profile from accumulated memories."""
    if not memories:
        return UserProfile(user_id=user_id)

    properties: dict = {}
    semantic_texts = []
    first_seen = None
    last_seen = None

    for r in memories:
        mem = r.memory
        # Track time range
        if first_seen is None or mem.created_at < first_seen:
            first_seen = mem.created_at
        if last_seen is None or mem.created_at > last_seen:
            last_seen = mem.created_at

        # Extract properties from semantic memories
        if mem.memory_type == MemoryType.SEMANTIC:
            semantic_texts.append(mem.text)
            _extract_properties(mem.text, properties)

        # Include metadata properties
        for k, v in mem.metadata.items():
            if k not in ("session_id", "intent"):
                properties[k] = v

    # Build summary from top semantic memories (by importance)
    sorted_memories = sorted(memories, key=lambda r: r.memory.importance, reverse=True)
    top_facts = [r.memory.text for r in sorted_memories[:5] if r.memory.memory_type == MemoryType.SEMANTIC]
    summary = ". ".join(top_facts) if top_facts else ""

    return UserProfile(
        user_id=user_id,
        properties=properties,
        summary=summary,
        memory_count=len(memories),
        first_seen=first_seen,
        last_seen=last_seen,
    )


def _extract_properties(text: str, properties: dict):
    """Simple heuristic extraction of user properties from memory text."""
    text_lower = text.lower()

    # Location patterns
    location_prefixes = ["lives in", "located in", "based in", "moved to", "from"]
    for prefix in location_prefixes:
        if prefix in text_lower:
            idx = text_lower.index(prefix) + len(prefix)
            location = text[idx:].strip().rstrip(".").strip()
            if location:
                properties["location"] = location
                break

    # Name patterns
    name_prefixes = ["name is", "called", "i'm", "i am"]
    for prefix in name_prefixes:
        if prefix in text_lower:
            idx = text_lower.index(prefix) + len(prefix)
            name = text[idx:].strip().split()[0].rstrip(".,!").strip()
            if name and len(name) > 1:
                properties["name"] = name
                break

    # Preference patterns
    pref_prefixes = ["prefers", "likes", "loves", "enjoys", "wants"]
    for prefix in pref_prefixes:
        if prefix in text_lower:
            idx = text_lower.index(prefix) + len(prefix)
            pref = text[idx:].strip().rstrip(".").strip()
            if pref:
                preferences = properties.get("preferences", [])
                if isinstance(preferences, list) and pref not in preferences:
                    preferences.append(pref)
                    properties["preferences"] = preferences
                break
