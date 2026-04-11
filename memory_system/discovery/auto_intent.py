import re
import yaml
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DiscoveredIntent:
    name: str
    description: str
    keywords: list[str]
    instructions: str
    sample_messages: list[str]
    cluster_size: int


@dataclass
class DiscoveryResult:
    intents: list[DiscoveredIntent]
    unclustered_messages: list[str] = field(default_factory=list)
    silhouette_score: Optional[float] = None


class IntentDiscovery:
    """Automatically discover intents from conversation logs using clustering."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._model_name = embedding_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def discover(
        self,
        messages: list[str],
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 3,
    ) -> DiscoveryResult:
        """
        Discover intents from messages using embedding clustering + TF-IDF keywords.
        No LLM required.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics import silhouette_score
            import numpy as np
        except ImportError:
            raise ImportError(
                "Auto-intent discovery requires scikit-learn. "
                "Install with: pip install memory_system[discovery]"
            )

        if len(messages) < 6:
            return DiscoveryResult(
                intents=[],
                unclustered_messages=messages,
                silhouette_score=None,
            )

        # Embed all messages
        model = self._get_model()
        embeddings = model.encode(messages, normalize_embeddings=True)

        # Auto-detect optimal cluster count
        if n_clusters is None:
            n_clusters = self._find_optimal_k(embeddings, messages, min_k=2, max_k=min(10, len(messages) // 3))

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        score = None
        if n_clusters > 1:
            score = float(silhouette_score(embeddings, labels))

        # Extract keywords per cluster via TF-IDF
        tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf.fit(messages)

        intents = []
        unclustered = []

        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            cluster_messages = [messages[i] for i in cluster_indices]

            if len(cluster_messages) < min_cluster_size:
                unclustered.extend(cluster_messages)
                continue

            # Get top keywords for this cluster
            cluster_tfidf = tfidf.transform(cluster_messages)
            avg_scores = cluster_tfidf.mean(axis=0).A1
            feature_names = tfidf.get_feature_names_out()
            top_indices = avg_scores.argsort()[-7:][::-1]
            keywords = [feature_names[i] for i in top_indices if avg_scores[i] > 0]

            # Generate intent name from top keywords
            name = self._generate_name(keywords)
            description = f"User wants to {' or '.join(keywords[:3])}"
            instructions = f"Handle the user's request about {name.replace('_', ' ')}."

            intents.append(DiscoveredIntent(
                name=name,
                description=description,
                keywords=keywords[:5],
                instructions=instructions,
                sample_messages=cluster_messages[:5],
                cluster_size=len(cluster_messages),
            ))

        return DiscoveryResult(
            intents=intents,
            unclustered_messages=unclustered,
            silhouette_score=score,
        )

    async def discover_with_llm(
        self,
        messages: list[str],
        llm_fn: Optional[Callable] = None,
        model: str = "groq/llama-3.1-8b-instant",
        n_clusters: Optional[int] = None,
    ) -> DiscoveryResult:
        """Discover intents with LLM polish for names, descriptions, and instructions."""
        result = self.discover(messages, n_clusters=n_clusters)

        if not llm_fn or not result.intents:
            return result

        from memory_system.providers.llm import call_llm
        _llm_fn = llm_fn or call_llm

        polished_intents = []
        for intent in result.intents:
            samples = "\n".join(f"- {m}" for m in intent.sample_messages[:5])
            prompt = f"""Given these customer messages that belong to the same category:
{samples}

Keywords: {', '.join(intent.keywords)}

Generate:
1. A concise snake_case intent name (e.g., check_order_status)
2. A one-line description of what the user wants
3. A 2-3 sentence instruction for an AI agent handling this intent

Respond in this exact format:
NAME: <intent_name>
DESCRIPTION: <description>
INSTRUCTIONS: <instructions>"""

            response = await _llm_fn(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            # Parse LLM response
            name = intent.name
            description = intent.description
            instructions = intent.instructions

            for line in response.strip().split("\n"):
                if line.startswith("NAME:"):
                    name = re.sub(r"[^a-z0-9_]", "_", line.split(":", 1)[1].strip().lower())
                elif line.startswith("DESCRIPTION:"):
                    description = line.split(":", 1)[1].strip()
                elif line.startswith("INSTRUCTIONS:"):
                    instructions = line.split(":", 1)[1].strip()

            polished_intents.append(DiscoveredIntent(
                name=name,
                description=description,
                keywords=intent.keywords,
                instructions=instructions,
                sample_messages=intent.sample_messages,
                cluster_size=intent.cluster_size,
            ))

        return DiscoveryResult(
            intents=polished_intents,
            unclustered_messages=result.unclustered_messages,
            silhouette_score=result.silhouette_score,
        )

    def to_yaml(self, result: DiscoveryResult, bot_id: str, bot_name: str) -> str:
        """Convert discovery result to a ready-to-use bot YAML config."""
        config = {
            "bot_id": bot_id,
            "bot_name": bot_name,
            "base_instructions": f"You are {bot_name}, a helpful assistant. Be concise and helpful.",
            "intents": [],
            "fallback_instructions": "The request doesn't match a known category. Try to be helpful.",
        }

        for intent in result.intents:
            config["intents"].append({
                "name": intent.name,
                "description": intent.description,
                "keywords": intent.keywords,
                "instructions": intent.instructions,
                "max_history_turns": 2,
            })

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _find_optimal_k(self, embeddings, messages, min_k=2, max_k=8):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        max_k = min(max_k, len(messages) - 1)
        if max_k < min_k:
            return min_k

        best_k = min_k
        best_score = -1

        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def _generate_name(self, keywords: list[str]) -> str:
        if not keywords:
            return "unknown"
        name = "_".join(keywords[:2])
        return re.sub(r"[^a-z0-9_]", "", name.lower())
