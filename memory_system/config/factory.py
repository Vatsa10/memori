"""Factory methods to create storage providers from configuration."""

from typing import TYPE_CHECKING, Any, Optional

from memory_system.config.settings import Settings
from memory_system.core.models import BotConfig

if TYPE_CHECKING:
    from memory_system.providers.in_memory_stores import (
        InMemoryGraphStore,
        InMemoryMemoryStore,
    )
    from memory_system.providers.neo4j_store import Neo4jGraphStore
    from memory_system.providers.qdrant_store import QdrantMemoryStore
    from memory_system.providers.redis_store import RedisCacheStore, RedisSessionStore
    from memory_system.providers.session import SessionStore


def create_providers(
    config: Optional[BotConfig] = None,
    *,
    memory_store_type: Optional[str] = None,
    graph_store_type: Optional[str] = None,
    session_store_type: Optional[str] = None,
    cache_type: Optional[str] = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Create storage providers from config/env.

    Args:
        config: BotConfig with optional provider type overrides.
        memory_store_type: Override type - "qdrant" or "inmemory".
        graph_store_type: Override type - "neo4j" or "inmemory".
        session_store_type: Override type - "redis" or "inmemory".
        cache_type: Override type - "redis" or "none".
        **overrides: Connection parameters (e.g., qdrant_url="...", neo4j_uri="...").

    Returns:
        Dict with keys: memory_store, graph_store, session_store, cache.

     Provider priority:
         1. Explicit type parameter (memory_store_type, etc.)
         2. Config provider_choice if provided in BotConfig
         3. Environment-based default from Settings (.env variables)
    """
    settings = Settings()
    result: dict[str, Any] = {}

    _memory_store_type = _resolve_type(
        explicit=memory_store_type,
        config_choice=_get_config_provider(config, "memory_store_type"),
        env_default=settings.memory_store,
    )

    _graph_store_type = _resolve_type(
        explicit=graph_store_type,
        config_choice=_get_config_provider(config, "graph_store_type"),
        env_default=settings.graph_store,
    )

    _session_store_type = _resolve_type(
        explicit=session_store_type,
        config_choice=_get_config_provider(config, "session_store_type"),
        env_default=settings.session_store,
    )

    _cache_type = _resolve_type(
        explicit=cache_type,
        config_choice=_get_config_provider(config, "cache_type"),
        env_default=settings.cache,
    )

    result["memory_store"] = _create_memory_store(
        _memory_store_type, settings, overrides
    )
    result["graph_store"] = _create_graph_store(_graph_store_type, settings, overrides)
    result["session_store"] = _create_session_store(
        _session_store_type, settings, overrides
    )
    result["cache"] = _create_cache(_cache_type, settings, overrides)

    return result


def _resolve_type(
    explicit: Optional[str],
    config_choice: Optional[str],
    env_default: Optional[str],
) -> str:
    """Resolve the provider type from explicit, config, or default."""
    if explicit:
        return explicit
    if config_choice:
        return config_choice
    return env_default or "inmemory"


def _get_config_provider(config: Optional[BotConfig], key: str) -> Optional[str]:
    """Extract provider choice from BotConfig if available."""
    if config and hasattr(config, "provider_choice"):
        return getattr(config.provider_choice, key, None)
    return None


def _create_memory_store(
    store_type: str, settings: Settings, overrides: dict[str, Any]
) -> Any:
    """Create a MemoryStore instance."""
    if store_type == "qdrant":
        try:
            from memory_system.providers.qdrant_store import QdrantMemoryStore
        except ImportError as e:
            raise ImportError(
                "qdrant_client is required for QdrantMemoryStore. "
                "Install with: pip install qdrant-client"
            ) from e

        return QdrantMemoryStore(
            url=overrides.get("qdrant_url") or settings.qdrant_url,
            api_key=overrides.get("qdrant_api_key") or settings.qdrant_api_key or None,
            collection=overrides.get("qdrant_collection") or settings.qdrant_collection,
            embedding_model_name=overrides.get("embedding_model")
            or settings.embedding_model,
            vector_size=overrides.get("vector_size", 384),
        )

    from memory_system.providers.in_memory_stores import InMemoryMemoryStore

    return InMemoryMemoryStore()


def _create_graph_store(
    store_type: str, settings: Settings, overrides: dict[str, Any]
) -> Any:
    """Create a GraphStore instance."""
    if store_type == "neo4j":
        try:
            from memory_system.providers.neo4j_store import Neo4jGraphStore
        except ImportError as e:
            raise ImportError(
                "neo4j is required for Neo4jGraphStore. Install with: pip install neo4j"
            ) from e

        return Neo4jGraphStore(
            uri=overrides.get("neo4j_uri") or "bolt://localhost:7687",
            user=overrides.get("neo4j_user") or "neo4j",
            password=overrides.get("neo4j_password") or "password",
        )

    from memory_system.providers.in_memory_stores import InMemoryGraphStore

    return InMemoryGraphStore()


def _create_session_store(
    store_type: str, settings: Settings, overrides: dict[str, Any]
) -> Any:
    """Create a SessionStore instance."""
    if store_type == "redis":
        try:
            from memory_system.providers.redis_store import RedisSessionStore
        except ImportError as e:
            raise ImportError(
                "redis is required for RedisSessionStore. "
                "Install with: pip install redis"
            ) from e

        return RedisSessionStore(
            url=overrides.get("redis_url") or "redis://localhost:6379",
            prefix=overrides.get("session_prefix") or "memory_system:session:",
            ttl=overrides.get("session_ttl") or 86400,
        )

    from memory_system.providers.session import SessionStore

    return SessionStore()


def _create_cache(
    store_type: str, settings: Settings, overrides: dict[str, Any]
) -> Optional[Any]:
    """Create a Cache instance."""
    if store_type == "none" or store_type is None:
        return None

    if store_type == "redis":
        try:
            from memory_system.providers.redis_store import RedisCacheStore
        except ImportError as e:
            raise ImportError(
                "redis is required for RedisCacheStore. Install with: pip install redis"
            ) from e

        return RedisCacheStore(
            url=overrides.get("redis_url") or "redis://localhost:6379",
            prefix=overrides.get("cache_prefix") or "memory_system:cache:",
            default_ttl=overrides.get("cache_ttl") or 3600,
        )

    return None
