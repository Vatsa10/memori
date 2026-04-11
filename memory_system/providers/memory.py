from typing import Optional


class InMemoryProvider:
    """Simple in-memory knowledge base for testing. No external dependencies."""

    def __init__(self):
        self._documents: list[str] = []

    def add(self, text: str):
        self._documents.append(text)

    async def search(self, query: str, k: int = 2) -> list[str]:
        if not self._documents:
            return []

        # Simple keyword overlap scoring
        query_words = set(query.lower().split())
        scored = []
        for doc in self._documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]


class QdrantMemoryProvider:
    """Qdrant-backed memory provider for production use."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection: str = "bot_knowledge",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self._url = url
        self._api_key = api_key
        self._collection = collection
        self._embedding_model_name = embedding_model_name
        self._client = None
        self._embedding_model = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key if self._api_key else None,
            )
        return self._client

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    async def search(self, query: str, k: int = 2) -> list[str]:
        try:
            client = self._get_client()
            model = self._get_embedding_model()

            query_vector = model.encode(query).tolist()

            results = client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=k,
            )

            return [
                hit.payload.get("text", "")
                for hit in results
                if hit.payload and hit.payload.get("text")
            ]
        except Exception:
            return []
