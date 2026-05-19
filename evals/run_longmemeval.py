"""LongMemEval benchmark runner.

LongMemEval dataset format (xiaowu0162/LongMemEval):
- ~500 Qs across 5 categories
- File: longmemeval_s.json (or _m.json / _oracle.json) — list of items, each:
  {question_id, question_type, question, answer, haystack_sessions: [[{role, content}, ...]]}

Usage:
    python -m evals.run_longmemeval --data evals/data/longmemeval_s.json --config evals/configs/full.yaml
    python -m evals.run_longmemeval --data ... --all-configs --limit 50
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from evals.runner import Probe, Sample, Session, Turn, run_ablation


# LongMemEval's 5 ability categories
_QTYPE_TO_CATEGORY = {
    "single-session-user": "info-extraction",
    "single-session-assistant": "info-extraction",
    "single-session-preference": "info-extraction",
    "temporal-reasoning": "temporal",
    "multi-session": "multi-session",
    "knowledge-update": "knowledge-update",
    "abstention": "abstention",
}


def _load_longmemeval(path: Path) -> list[Sample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: list[Sample] = []
    for entry in raw:
        qid = str(entry.get("question_id") or entry.get("id") or entry.get("uid"))
        haystack = entry.get("haystack_sessions") or entry.get("sessions") or []

        sessions: list[Session] = []
        for i, session_turns in enumerate(haystack):
            turns: list[Turn] = []
            for t in session_turns:
                role = t.get("role") or "user"
                content = t.get("content") or t.get("text") or ""
                if role not in ("user", "assistant"):
                    role = "user"
                turns.append(Turn(role=role, content=content))
            sessions.append(Session(session_id=f"{qid}-s{i}", turns=turns))

        qtype = entry.get("question_type", "")
        probe = Probe(
            question=entry.get("question", ""),
            gold_answer=str(entry.get("answer", "")),
            category=_QTYPE_TO_CATEGORY.get(qtype, qtype),
        )
        samples.append(
            Sample(
                sample_id=qid,
                sessions=sessions,
                probes=[probe],
                metadata={"question_type": qtype},
            )
        )
    return samples


def _all_configs() -> list[Path]:
    return sorted((Path(__file__).parent / "configs").glob("*.yaml"))


async def _main(args) -> None:
    samples = _load_longmemeval(Path(args.data))
    output_dir = Path(args.out)
    configs = _all_configs() if args.all_configs else [Path(args.config)]

    summaries: dict[str, dict] = {}
    for cfg in configs:
        summary = await run_ablation(
            config_path=cfg,
            samples=samples,
            output_dir=output_dir / "longmemeval",
            limit=args.limit,
            concurrency=args.concurrency,
        )
        summaries[cfg.stem] = summary

    # Comparison table
    print("\n=== LongMemEval comparison (overall + per-category judge_acc) ===")
    cats = sorted({c for s in summaries.values() for c in s.get("by_category", {})})
    header = f"{'config':<14} {'n':>5} {'judge':>7} {'F1':>6} " + " ".join(f"{c[:10]:>10}" for c in cats)
    print(header)
    for name, s in summaries.items():
        row = f"{name:<14} {s['n']:>5} {s['judge_acc']:>7.3f} {s['f1']:>6.3f} "
        for c in cats:
            cat_data = s.get("by_category", {}).get(c, {})
            row += f"{cat_data.get('judge_acc', 0):>10.3f} "
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to longmemeval_s.json (or _m / _oracle)")
    p.add_argument(
        "--config", default="evals/configs/full.yaml",
        help="Single ablation config to run",
    )
    p.add_argument("--all-configs", action="store_true")
    p.add_argument("--out", default="evals/results")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
