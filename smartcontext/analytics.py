from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from smartcontext.core.models import ChatResponse


@dataclass
class AnalyticsSnapshot:
    total_requests: int
    intent_distribution: dict[str, int]
    prediction_method_distribution: dict[str, int]
    avg_latency_ms: dict[str, float]
    avg_reduction_percent: float
    avg_smart_tokens: float
    avg_full_tokens: float
    cache_hit_rate: float
    recent_requests: list[dict]


class AnalyticsCollector:
    def __init__(self, max_history: int = 1000):
        self._intent_counts: Counter = Counter()
        self._method_counts: Counter = Counter()
        self._latency_sums: dict[str, float] = {}
        self._latency_counts: dict[str, int] = {}
        self._total_requests: int = 0
        self._total_reduction: float = 0.0
        self._total_smart_tokens: int = 0
        self._total_full_tokens: int = 0
        self._cache_hits: int = 0
        self._recent: deque = deque(maxlen=max_history)

    def record(self, response: ChatResponse, cache_hit: bool = False):
        self._total_requests += 1
        self._intent_counts[response.intent.intent_name] += 1
        self._method_counts[response.intent.method.value] += 1
        self._total_reduction += response.reduction_percent
        self._total_smart_tokens += response.token_estimate
        self._total_full_tokens += response.full_prompt_estimate
        if cache_hit:
            self._cache_hits += 1

        for stage, ms in response.latency_ms.items():
            self._latency_sums[stage] = self._latency_sums.get(stage, 0) + ms
            self._latency_counts[stage] = self._latency_counts.get(stage, 0) + 1

        self._recent.append({
            "intent": response.intent.intent_name,
            "method": response.intent.method.value,
            "confidence": response.intent.confidence,
            "token_estimate": response.token_estimate,
            "full_prompt_estimate": response.full_prompt_estimate,
            "reduction_percent": response.reduction_percent,
            "latency_ms": response.latency_ms,
            "cache_hit": cache_hit,
        })

    def snapshot(self) -> AnalyticsSnapshot:
        n = max(self._total_requests, 1)
        avg_latency = {
            stage: round(self._latency_sums[stage] / self._latency_counts[stage], 2)
            for stage in self._latency_sums
            if self._latency_counts.get(stage, 0) > 0
        }

        return AnalyticsSnapshot(
            total_requests=self._total_requests,
            intent_distribution=dict(self._intent_counts),
            prediction_method_distribution=dict(self._method_counts),
            avg_latency_ms=avg_latency,
            avg_reduction_percent=round(self._total_reduction / n, 1),
            avg_smart_tokens=round(self._total_smart_tokens / n, 1),
            avg_full_tokens=round(self._total_full_tokens / n, 1),
            cache_hit_rate=round(self._cache_hits / n, 3),
            recent_requests=list(self._recent),
        )

    def export(self) -> dict:
        snap = self.snapshot()
        return {
            "total_requests": snap.total_requests,
            "intent_distribution": snap.intent_distribution,
            "prediction_method_distribution": snap.prediction_method_distribution,
            "avg_latency_ms": snap.avg_latency_ms,
            "avg_reduction_percent": snap.avg_reduction_percent,
            "avg_smart_tokens": snap.avg_smart_tokens,
            "avg_full_tokens": snap.avg_full_tokens,
            "cache_hit_rate": snap.cache_hit_rate,
            "recent_requests": snap.recent_requests,
        }

    def reset(self):
        self._intent_counts.clear()
        self._method_counts.clear()
        self._latency_sums.clear()
        self._latency_counts.clear()
        self._total_requests = 0
        self._total_reduction = 0.0
        self._total_smart_tokens = 0
        self._total_full_tokens = 0
        self._cache_hits = 0
        self._recent.clear()
