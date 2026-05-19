"""Shared eval harness: build MemorySystem from a config, ingest sessions,
probe questions, score with metrics."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from evals.llm_factory import (
    assert_keys_present,
    get_llm_fn,
    get_model,
    summary as llm_summary,
)
from evals.metrics import aggregate, exact_match, llm_judge, token_f1
from memory_system import MemorySystem
from memory_system.providers.in_memory_stores import (
    InMemoryGraphStore,
    InMemoryMemoryStore,
)


# --- Data shape (benchmark-agnostic) ---

@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Session:
    session_id: str
    turns: list[Turn]
    started_at: Optional[str] = None  # ISO timestamp if available


@dataclass
class Probe:
    question: str
    gold_answer: str
    category: Optional[str] = None  # e.g. "temporal", "multi-hop"


@dataclass
class Sample:
    sample_id: str
    sessions: list[Session]
    probes: list[Probe]
    metadata: dict = field(default_factory=dict)


# --- Config loading ---

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_memory_system(features: dict) -> MemorySystem:
    """Build a MemorySystem with InMemory providers + features per ablation."""
    mem_store = InMemoryMemoryStore()
    graph_store = InMemoryGraphStore()

    llm = get_llm_fn()
    reasoning_llm = get_llm_fn(with_reasoning=True)

    kwargs: dict[str, Any] = {
        "instructions": "Answer using the user's memory and any retrieved context. Be concise and factual.",
        "model": get_model("generation"),
        "llm_fn": llm,
        "memory_store": mem_store,
        "graph_store": graph_store,
        "extraction_llm_fn": llm,
        "extraction_model": get_model("extraction"),
        "smart_ops_model": get_model("smart_ops_judge"),
        "enable_smart_ops": features.get("enable_smart_ops", False),
        "enable_summary_tree": features.get("enable_summary_tree", False),
    }

    if features.get("use_hybrid_retriever"):
        from memory_system.retrieval import BM25Retriever, HybridRetriever

        bm25 = BM25Retriever(mem_store)
        kwargs["retriever"] = HybridRetriever(
            mem_store,
            graph_store=graph_store,
            bm25=bm25,
            graph_max_hops=features.get("graph_max_hops", 2),
        )

    if features.get("use_reranker"):
        from memory_system.retrieval import CrossEncoderReranker

        kwargs["reranker"] = CrossEncoderReranker()

    return MemorySystem(**kwargs)


# --- Ingest + probe ---

async def _ingest_sessions(ms: MemorySystem, sample: Sample) -> None:
    """Push every turn pair through ms._memory.remember to populate memory."""
    user_id = sample.sample_id
    for session in sample.sessions:
        # Pair adjacent (user, assistant) turns
        pending_user: Optional[str] = None
        for turn in session.turns:
            if turn.role == "user":
                pending_user = turn.content
                continue
            if turn.role == "assistant" and pending_user is not None:
                await ms._memory.remember(
                    messages=[
                        {"role": "user", "content": pending_user},
                        {"role": "assistant", "content": turn.content},
                    ],
                    user_id=user_id,
                    session_id=session.session_id,
                )
                pending_user = None


async def _probe(ms: MemorySystem, sample: Sample, probe: Probe) -> dict:
    """Run one probe, return per-question metric record."""
    t0 = time.perf_counter()
    result = await ms.chat(probe.question, user_id=sample.sample_id)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    pred = result.response
    em = exact_match(pred, probe.gold_answer)
    f1 = token_f1(pred, probe.gold_answer)
    verdict = await llm_judge(probe.question, pred, probe.gold_answer)

    return {
        "sample_id": sample.sample_id,
        "question": probe.question,
        "gold": probe.gold_answer,
        "pred": pred,
        "category": probe.category,
        "em": em,
        "f1": f1,
        "judge": verdict.correct,
        "judge_rationale": verdict.rationale,
        "latency_ms": latency_ms,
        "memories_recalled": result.memories_recalled,
    }


async def run_ablation(
    *,
    config_path: str | Path,
    samples: list[Sample],
    output_dir: str | Path,
    limit: Optional[int] = None,
    concurrency: int = 4,
) -> dict:
    """Run one ablation config over a list of samples; write JSONL results."""
    config = load_config(config_path)
    features = config.get("features", {})
    name = config["name"]

    assert_keys_present(["generation", "extraction", "smart_ops_judge", "eval_judge"])

    if limit is not None:
        samples = samples[:limit]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.jsonl"

    all_records: list[dict] = []
    print(f"\n=== ablation: {name} | {len(samples)} samples ===")
    print(f"models: {json.dumps(llm_summary(), indent=2)}")

    for i, sample in enumerate(samples, 1):
        ms = _build_memory_system(features)
        print(f"[{i}/{len(samples)}] sample={sample.sample_id} ingest...", end="", flush=True)
        t0 = time.perf_counter()
        await _ingest_sessions(ms, sample)
        ingest_ms = round((time.perf_counter() - t0) * 1000)
        print(f" done in {ingest_ms}ms. probing {len(sample.probes)} questions...")

        # Probe with bounded concurrency
        sem = asyncio.Semaphore(concurrency)

        async def _bounded_probe(p):
            async with sem:
                return await _probe(ms, sample, p)

        records = await asyncio.gather(*[_bounded_probe(p) for p in sample.probes])
        all_records.extend(records)

        # Stream to disk so partial runs survive crashes
        with open(out_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        partial = aggregate(records)
        print(
            f"  sample score: judge={partial['judge_acc']:.2f} "
            f"f1={partial['f1']:.2f} em={partial['em']:.2f}"
        )

    summary_metrics = aggregate(all_records)
    summary_path = output_dir / f"{name}.summary.json"
    summary_metrics["config"] = config
    summary_metrics["models"] = llm_summary()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\n=== {name} done ===")
    print(json.dumps({k: v for k, v in summary_metrics.items() if k != "by_category"}, indent=2))
    return summary_metrics
