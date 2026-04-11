import pytest
from memory_system.analytics import AnalyticsCollector
from memory_system.core.models import ChatResponse, IntentPrediction, PredictionMethod


def _make_response(intent_name="check_order", method=PredictionMethod.KEYWORD, reduction=70.0):
    return ChatResponse(
        response="test",
        intent=IntentPrediction(
            intent_name=intent_name,
            confidence=0.9,
            method=method,
        ),
        token_estimate=100,
        full_prompt_estimate=333,
        reduction_percent=reduction,
        latency_ms={
            "intent_prediction_ms": 1.0,
            "context_assembly_ms": 2.0,
            "generation_ms": 100.0,
            "total_ms": 103.0,
        },
    )


class TestAnalyticsCollector:
    def test_record_and_snapshot(self):
        collector = AnalyticsCollector()
        collector.record(_make_response())

        snap = collector.snapshot()
        assert snap.total_requests == 1
        assert snap.intent_distribution == {"check_order": 1}
        assert snap.prediction_method_distribution == {"keyword": 1}

    def test_multiple_records(self):
        collector = AnalyticsCollector()
        collector.record(_make_response("check_order"))
        collector.record(_make_response("return_item"))
        collector.record(_make_response("check_order"))

        snap = collector.snapshot()
        assert snap.total_requests == 3
        assert snap.intent_distribution["check_order"] == 2
        assert snap.intent_distribution["return_item"] == 1

    def test_avg_reduction(self):
        collector = AnalyticsCollector()
        collector.record(_make_response(reduction=60.0))
        collector.record(_make_response(reduction=80.0))

        snap = collector.snapshot()
        assert snap.avg_reduction_percent == 70.0

    def test_cache_hit_tracking(self):
        collector = AnalyticsCollector()
        collector.record(_make_response(), cache_hit=True)
        collector.record(_make_response(), cache_hit=False)

        snap = collector.snapshot()
        assert snap.cache_hit_rate == 0.5

    def test_latency_averages(self):
        collector = AnalyticsCollector()
        collector.record(_make_response())
        collector.record(_make_response())

        snap = collector.snapshot()
        assert snap.avg_latency_ms["intent_prediction_ms"] == 1.0
        assert snap.avg_latency_ms["generation_ms"] == 100.0

    def test_export(self):
        collector = AnalyticsCollector()
        collector.record(_make_response())

        data = collector.export()
        assert isinstance(data, dict)
        assert "total_requests" in data
        assert "recent_requests" in data

    def test_reset(self):
        collector = AnalyticsCollector()
        collector.record(_make_response())
        collector.reset()

        snap = collector.snapshot()
        assert snap.total_requests == 0

    def test_recent_requests_limit(self):
        collector = AnalyticsCollector(max_history=5)
        for i in range(10):
            collector.record(_make_response())

        snap = collector.snapshot()
        assert len(snap.recent_requests) == 5
