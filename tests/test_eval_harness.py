"""Smoke test for evals/ harness — mocked LLMs, no live API calls."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from evals.runner import Probe, Sample, Session, Turn


def test_metrics_em_and_f1():
    from evals.metrics import exact_match, token_f1

    assert exact_match("Seattle", "seattle") == 1.0
    assert exact_match("Seattle, WA", "Seattle") == 0.0
    assert token_f1("Alice lives in Seattle", "Alice lives Seattle") > 0.7
    assert token_f1("totally wrong", "right answer") < 0.3


def test_aggregate_handles_empty():
    from evals.metrics import aggregate

    out = aggregate([])
    assert out["n"] == 0
    assert out["em"] == 0.0


def test_aggregate_by_category():
    from evals.metrics import aggregate

    records = [
        {"em": 1.0, "f1": 1.0, "judge": True, "category": "temporal"},
        {"em": 0.0, "f1": 0.0, "judge": False, "category": "temporal"},
        {"em": 1.0, "f1": 1.0, "judge": True, "category": "multi-hop"},
    ]
    out = aggregate(records)
    assert out["n"] == 3
    assert out["judge_acc"] == pytest.approx(2 / 3)
    assert out["by_category"]["temporal"]["judge_acc"] == 0.5
    assert out["by_category"]["multi-hop"]["judge_acc"] == 1.0


def test_llm_factory_role_defaults_and_overrides(monkeypatch):
    from evals.llm_factory import get_model

    assert get_model("generation") == "deepseek/deepseek-v4-flash"
    assert get_model("eval_judge") == "openai/gpt-5.4-nano"
    assert get_model("summary_rollup") == "gemini/gemini-2.5-flash-lite"

    monkeypatch.setenv("EVAL_MODEL_GENERATION", "openai/gpt-5.4-mini")
    assert get_model("generation") == "openai/gpt-5.4-mini"


def test_llm_factory_unknown_role_raises():
    from evals.llm_factory import get_model

    with pytest.raises(ValueError):
        get_model("nonexistent")


def test_locomo_loader_parses_sample(tmp_path):
    from evals.run_locomo import _load_locomo

    fixture = [
        {
            "sample_id": "loc_001",
            "conversation": {
                "session_1": [
                    {"speaker": "user", "text": "I just moved to Seattle."},
                    {"speaker": "speaker_b", "text": "Got it, welcome to Seattle."},
                ],
            },
            "qa": [
                {"question": "Where does the user live?", "answer": "Seattle", "category": 1},
                {"question": "When did they move?", "answer": "today", "category": 3},
            ],
        }
    ]
    path = tmp_path / "loc.json"
    path.write_text(json.dumps(fixture))

    samples = _load_locomo(path)
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "loc_001"
    assert len(s.sessions) == 1
    assert s.sessions[0].turns[0].role == "user"
    assert s.sessions[0].turns[1].role == "assistant"
    assert len(s.probes) == 2
    assert s.probes[0].category == "single-hop"
    assert s.probes[1].category == "temporal"


def test_longmemeval_loader_parses_sample(tmp_path):
    from evals.run_longmemeval import _load_longmemeval

    fixture = [
        {
            "question_id": "lme_001",
            "question_type": "temporal-reasoning",
            "question": "When did user move?",
            "answer": "yesterday",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I moved yesterday."},
                    {"role": "assistant", "content": "Welcome."},
                ]
            ],
        }
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(fixture))

    samples = _load_longmemeval(path)
    assert len(samples) == 1
    assert samples[0].probes[0].category == "temporal"
    assert samples[0].metadata["question_type"] == "temporal-reasoning"


@pytest.mark.asyncio
async def test_run_ablation_end_to_end_with_mocked_llm(tmp_path, monkeypatch):
    """Run a full ablation cycle on a 1-sample fixture, all LLM calls mocked."""
    from evals.runner import run_ablation

    # Mock the llm_fn used inside MemorySystem (via litellm.acompletion)
    async def fake_acompletion(model, messages, **kwargs):
        class _Choice:
            class message:
                content = "Seattle"
        class _Resp:
            choices = [_Choice()]
        return _Resp()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    # Mock the structured-output judge to return correct
    from evals.metrics import JudgeVerdict

    async def fake_judge(question, pred, gold, **kwargs):
        return JudgeVerdict(correct=(pred.strip().lower() == gold.strip().lower()))

    monkeypatch.setattr("evals.runner.llm_judge", fake_judge)

    # Bypass API key requirement (we mocked everything)
    monkeypatch.setattr(
        "evals.runner.assert_keys_present", lambda roles: None
    )

    sample = Sample(
        sample_id="smoke_001",
        sessions=[
            Session(
                session_id="s1",
                turns=[
                    Turn(role="user", content="I just moved to Seattle from NYC."),
                    Turn(role="assistant", content="Got it."),
                ],
            )
        ],
        probes=[Probe(question="Where do I live?", gold_answer="Seattle", category="single-hop")],
    )

    # Use baseline config (no extras → no extra LLM call paths to mock)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "name: smoke\n"
        "description: test\n"
        "features:\n"
        "  enable_smart_ops: false\n"
        "  enable_summary_tree: false\n"
        "  use_hybrid_retriever: false\n"
        "  use_reranker: false\n"
        "  graph_max_hops: 1\n"
    )

    summary = await run_ablation(
        config_path=config_path,
        samples=[sample],
        output_dir=tmp_path / "out",
        concurrency=1,
    )

    assert summary["n"] == 1
    assert summary["judge_acc"] == 1.0
    assert (tmp_path / "out" / "smoke.jsonl").exists()
    assert (tmp_path / "out" / "smoke.summary.json").exists()
