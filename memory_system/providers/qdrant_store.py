"""Production Qdrant-backed memory store."""

from datetime import datetime, timezone
from typing import Optional

from memory_system.core.memory_models import Memory, MemorySearchResult, MemoryType


def _iso_or_none(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value, fallback: Optional[datetime] = None) -> Optional[datetime]:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return fallback


def _to_payload(memory: Memory) -> dict:
    return {
        "text": memory.text,
        "user_id": memory.user_id,
        "memory_type": memory.memory_type.value,
        "source": memory.source,
        "metadata": memory.metadata,
        "importance": memory.importance,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        # Bi-temporal
        "valid_from": memory.valid_from.isoformat(),
        "valid_to": _iso_or_none(memory.valid_to),
        "recorded_at": memory.recorded_at.isoformat(),
        "superseded_by": memory.superseded_by,
        # Provenance
        "source_text": memory.source_text,
        "turn_id": memory.turn_id,
        "confidence": memory.confidence,
        "extractor_model": memory.extractor_model,
    }


def _from_payload(payload: dict, point_id, user_id: str) -> Memory:
    """Hydrate Memory from a Qdrant payload; old rows lacking new fields get defaults."""
    created = _parse_dt(payload.get("created_at"))
    fallback_dt = created or datetime.now(timezone.utc)
    return Memory(
        id=str(point_id),
        text=payload.get("text", ""),
        memory_type=MemoryType(payload.get("memory_type", "semantic")),
        user_id=payload.get("user_id", user_id),
        metadata=payload.get("metadata", {}) or {},
        source=payload.get("source", "vector"),
        importance=payload.get("importance", 0.5),
        created_at=created or fallback_dt,
        updated_at=_parse_dt(payload.get("updated_at"), fallback_dt) or fallback_dt,
        valid_from=_parse_dt(payload.get("valid_from"), fallback_dt) or fallback_dt,
        valid_to=_parse_dt(payload.get("valid_to")),
        recorded_at=_parse_dt(payload.get("recorded_at"), fallback_dt) or fallback_dt,
        superseded_by=payload.get("superseded_by"),
        source_text=payload.get("source_text"),
        turn_id=payload.get("turn_id"),
        confidence=payload.get("confidence", 1.0),
        extractor_model=payload.get("extractor_model"),
    )


class QdrantMemoryStore:
    """Full CRUD memory store backed by Qdrant."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection: str = "user_memories",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        vector_size: int = 384,
    ):
        self._url = url
        self._api_key = api_key
        self._collection = collection
        self._embedding_model_name = embedding_model_name
        self._vector_size = vector_size
        self._client = None
        self._embedding_model = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self._url, api_key=self._api_key or None)
        return self._client

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    async def ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        client = self._get_client()
        collections = [c.name for c in client.get_collections().collections]
        if self._collection not in collections:
            client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    async def add(self, memory: Memory) -> str:
        from qdrant_client.models import PointStruct
        client = self._get_client()
        model = self._get_embedding_model()

        vector = model.encode(memory.text).tolist()
        payload = _to_payload(memory)

        client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=memory.id, vector=vector, payload=payload)],
        )
        return memory.id

    async def search(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        filters: Optional[dict] = None,
        include_invalidated: bool = False,
    ) -> list[MemorySearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        model = self._get_embedding_model()

        query_vector = model.encode(query).tolist()

        conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if filters:
            for fk, fv in filters.items():
                conditions.append(FieldCondition(key=f"metadata.{fk}", match=MatchValue(value=fv)))

        # Over-fetch when we'll post-filter for bi-temporal
        fetch_limit = k if include_invalidated else k * 4

        results = client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=Filter(must=conditions),
            limit=fetch_limit,
        )

        hydrated = [
            MemorySearchResult(
                memory=_from_payload(hit.payload, hit.id, user_id),
                score=hit.score,
                source="vector",
            )
            for hit in results
            if hit.payload
        ]
        if not include_invalidated:
            hydrated = [r for r in hydrated if r.memory.is_current]
        return hydrated[:k]

    async def update(self, memory_id: str, text: str) -> None:
        client = self._get_client()
        model = self._get_embedding_model()

        vector = model.encode(text).tolist()
        client.set_payload(
            collection_name=self._collection,
            payload={"text": text, "updated_at": datetime.now(timezone.utc).isoformat()},
            points=[memory_id],
        )
        client.update_vectors(
            collection_name=self._collection,
            points=[{"id": memory_id, "vector": vector}],
        )

    async def delete(self, memory_id: str) -> None:
        client = self._get_client()
        client.delete(collection_name=self._collection, points_selector=[memory_id])

    async def get_all(
        self,
        user_id: str,
        k: int = 50,
        include_invalidated: bool = False,
    ) -> list[MemorySearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        fetch_limit = k if include_invalidated else k * 4

        results = client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]),
            limit=fetch_limit,
        )

        hydrated = [
            MemorySearchResult(
                memory=_from_payload(point.payload, point.id, user_id),
                score=1.0,
                source="vector",
            )
            for point in results[0]
            if point.payload
        ]
        if not include_invalidated:
            hydrated = [r for r in hydrated if r.memory.is_current]
        return hydrated[:k]

    async def invalidate(
        self,
        memory_id: str,
        valid_to: datetime,
        superseded_by: Optional[str] = None,
    ) -> None:
        client = self._get_client()
        payload = {
            "valid_to": valid_to.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if superseded_by is not None:
            payload["superseded_by"] = superseded_by
        client.set_payload(
            collection_name=self._collection,
            payload=payload,
            points=[memory_id],
        )

    async def search_at(
        self,
        query: str,
        user_id: str,
        as_of: datetime,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MemorySearchResult]:
        results = await self.search(
            query, user_id, k=k * 4, filters=filters, include_invalidated=True
        )
        valid = [r for r in results if r.memory.is_valid_at(as_of)]
        return valid[:k]
