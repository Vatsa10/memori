import re
import time
import numpy as np
from typing import Optional

from smartcontext.core.models import (
    BotConfig,
    ConversationTurn,
    IntentDefinition,
    IntentPrediction,
    PredictionMethod,
)


class IntentPredictor:
    """
    3-tier hybrid intent prediction:
      Tier 1: Keyword matching (0ms) — free, instant
      Tier 2: Embedding similarity (~10ms) — cheap, accurate
      Tier 3: LLM classification (~50ms) — expensive, handles ambiguity
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name
        self._intent_embeddings_cache: dict[str, dict[str, np.ndarray]] = {}

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def precompute_intent_embeddings(self, bot_config: BotConfig):
        """Pre-compute embeddings for all intent descriptions + keywords. Call on config load."""
        model = self._get_embedding_model()
        cache = {}
        for intent in bot_config.intents:
            text = f"{intent.name}: {intent.description}. Keywords: {', '.join(intent.keywords)}"
            cache[intent.name] = model.encode(text, normalize_embeddings=True)
        self._intent_embeddings_cache[bot_config.bot_id] = cache

    async def predict(
        self,
        bot_config: BotConfig,
        user_message: str,
        recent_history: list[ConversationTurn],
        llm_predict_fn=None,
    ) -> tuple[IntentPrediction, float]:
        """
        Returns (prediction, elapsed_ms).
        Tries each tier in order, stopping when confidence exceeds threshold.
        """
        start = time.perf_counter()

        # Tier 1: Keyword matching
        prediction = self._keyword_match(bot_config, user_message)
        if prediction and prediction.confidence >= bot_config.keyword_threshold:
            elapsed = (time.perf_counter() - start) * 1000
            return prediction, elapsed

        # Tier 2: Embedding similarity
        prediction = self._embedding_match(bot_config, user_message)
        if prediction and prediction.confidence >= bot_config.embedding_threshold:
            elapsed = (time.perf_counter() - start) * 1000
            return prediction, elapsed

        # Tier 3: LLM classification
        if llm_predict_fn:
            prediction = await self._llm_classify(
                bot_config, user_message, recent_history, llm_predict_fn
            )
            elapsed = (time.perf_counter() - start) * 1000
            return prediction, elapsed

        # Fallback: best embedding match regardless of threshold
        if prediction:
            prediction.method = PredictionMethod.FALLBACK
            elapsed = (time.perf_counter() - start) * 1000
            return prediction, elapsed

        # No intents defined
        elapsed = (time.perf_counter() - start) * 1000
        return IntentPrediction(
            intent_name="fallback",
            confidence=0.0,
            reasoning="No intents matched",
            method=PredictionMethod.FALLBACK,
        ), elapsed

    def _keyword_match(
        self, bot_config: BotConfig, user_message: str
    ) -> Optional[IntentPrediction]:
        message_lower = user_message.lower()
        message_words = set(re.findall(r"\w+", message_lower))

        best_intent: Optional[IntentDefinition] = None
        best_score = 0.0

        for intent in bot_config.intents:
            if not intent.keywords:
                continue

            best_kw_score = 0.0
            total_kw_score = 0.0
            matches = 0
            for keyword in intent.keywords:
                kw_lower = keyword.lower()
                # Exact phrase match scores highest
                if kw_lower in message_lower:
                    kw_score = 1.0
                else:
                    # Partial word overlap
                    kw_words = set(re.findall(r"\w+", kw_lower))
                    overlap = len(kw_words & message_words) / len(kw_words) if kw_words else 0
                    kw_score = overlap * 0.7 if overlap >= 0.5 else 0.0

                if kw_score > 0:
                    matches += 1
                    total_kw_score += kw_score
                    best_kw_score = max(best_kw_score, kw_score)

            if matches > 0:
                # Use best keyword score, boosted by additional matches
                score = best_kw_score + (matches - 1) * 0.1
                score = min(score, 1.0)
                if score > best_score:
                    best_score = score
                    best_intent = intent

        if best_intent and best_score > 0:
            return IntentPrediction(
                intent_name=best_intent.name,
                confidence=min(best_score, 1.0),
                reasoning=f"Keyword match score: {best_score:.2f}",
                method=PredictionMethod.KEYWORD,
            )
        return None

    def _embedding_match(
        self, bot_config: BotConfig, user_message: str
    ) -> Optional[IntentPrediction]:
        cache = self._intent_embeddings_cache.get(bot_config.bot_id)
        if not cache:
            return None

        model = self._get_embedding_model()
        msg_embedding = model.encode(user_message, normalize_embeddings=True)

        best_intent_name: Optional[str] = None
        best_similarity = -1.0

        for intent_name, intent_embedding in cache.items():
            similarity = float(np.dot(msg_embedding, intent_embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_intent_name = intent_name

        if best_intent_name:
            return IntentPrediction(
                intent_name=best_intent_name,
                confidence=max(0.0, min(best_similarity, 1.0)),
                reasoning=f"Embedding similarity: {best_similarity:.3f}",
                method=PredictionMethod.EMBEDDING,
            )
        return None

    async def _llm_classify(
        self,
        bot_config: BotConfig,
        user_message: str,
        recent_history: list[ConversationTurn],
        llm_predict_fn,
    ) -> IntentPrediction:
        intent_descriptions = "\n".join(
            f"- {intent.name}: {intent.description}"
            for intent in bot_config.intents
        )

        history_text = ""
        if recent_history:
            history_text = "\n".join(
                f"{turn.role}: {turn.content}" for turn in recent_history[-3:]
            )
            history_text = f"\nRecent conversation:\n{history_text}\n"

        classification_prompt = f"""Classify the user's intent into one of these categories:

{intent_descriptions}
- fallback: None of the above categories match
{history_text}
User message: {user_message}

Respond with the intent name that best matches. If unsure, use "fallback"."""

        prediction = await llm_predict_fn(
            model=bot_config.intent_model,
            prompt=classification_prompt,
        )
        prediction.method = PredictionMethod.LLM
        return prediction
