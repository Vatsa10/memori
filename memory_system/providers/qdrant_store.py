"""Production Qdrant-backed memory store."""

from datetime import datetime, timezone
from typing import Optional

from memory_system.core.memory_models import Memory, MemorySearchResult, MemoryType


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

        payload = {
            "text": memory.text,
            "user_id": memory.user_id,
            "memory_type": memory.memory_type.value,
            "source": memory.source,
            "metadata": memory.metadata,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
        }

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
    ) -> list[MemorySearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        model = self._get_embedding_model()

        query_vector = model.encode(query).tolist()

        # Build Qdrant filter
        conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if filters:
            for fk, fv in filters.items():
                conditions.append(FieldCondition(key=f"metadata.{fk}", match=MatchValue(value=fv)))

        results = client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=Filter(must=conditions),
            limit=k,
        )

        return [
            MemorySearchResult(
                memory=Memory(
                    id=str(hit.id),
                    text=hit.payload.get("text", ""),
                    memory_type=MemoryType(hit.payload.get("memory_type", "semantic")),
                    user_id=hit.payload.get("user_id", user_id),
                    metadata=hit.payload.get("metadata", {}),
                    source=hit.payload.get("source", "vector"),
                ),
                score=hit.score,
                source="vector",
            )
            for hit in results
            if hit.payload
        ]

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

    async def get_all(self, user_id: str, k: int = 50) -> list[MemorySearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        results = client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]),
            limit=k,
        )

        return [
            MemorySearchResult(
                memory=Memory(
                    id=str(point.id),
                    text=point.payload.get("text", ""),
                    memory_type=MemoryType(point.payload.get("memory_type", "semantic")),
                    user_id=point.payload.get("user_id", user_id),
                    metadata=point.payload.get("metadata", {}),
                    source=point.payload.get("source", "vector"),
                ),
                score=1.0,
                source="vector",
            )
            for point in results[0]
            if point.payload
        ]
