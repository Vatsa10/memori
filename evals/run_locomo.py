"""LoCoMo benchmark runner.

LoCoMo dataset format (snap-research/locomo):
- 10 conversations, each with multiple sessions and probe questions
- File: locomo10.json — list of {sample_id, conversation: {session_1: [...], ...}, qa: [...]}

Usage:
    python -m evals.run_locomo --data evals/data/locomo10.json --config evals/configs/full.yaml
    python -m evals.run_locomo --data ... --config ... --limit 1 --concurrency 2
    python -m evals.run_locomo --data ... --all-configs    # run every ablation
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from evals.runner import Probe, Sample, Session, Turn, run_ablation


_CATEGORY_MAP = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}


def _load_locomo(path: Path) -> list[Sample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]

    samples: list[Sample] = []
    for entry in raw:
        sid = str(entry.get("sample_id") or entry.get("id") or entry.get("conversation_id"))
        conv = entry.get("conversation") or {}
        sessions: list[Session] = []
        for key, turns in conv.items():
            if not key.startswith("session"):
                continue
            session_turns: list[Turn] = []
            for t in turns:
                speaker = t.get("speaker") or t.get("role") or "user"
                text = t.get("text") or t.get("content") or ""
                # LoCoMo uses arbitrary speaker names; map first → user, second → assistant
                role = "user" if str(speaker).strip().lower() in {"user", "speaker_a", "speakera"} else "assistant"
                session_turns.append(Turn(role=role, content=text))
            sessions.append(
                Session(
                    session_id=f"{sid}-{key}",
                    turns=session_turns,
                    started_at=conv.get(f"{key}_date_time"),
                )
            )

        probes: list[Probe] = []
        for q in entry.get("qa", []):
            cat_id = q.get("category")
            probes.append(
                Probe(
                    question=q.get("question", ""),
                    gold_answer=str(q.get("answer", "")),
                    category=_CATEGORY_MAP.get(cat_id, str(cat_id) if cat_id else None),
                )
            )

        samples.append(Sample(sample_id=sid, sessions=sessions, probes=probes))
    return samples


def _all_configs() -> list[Path]:
    return sorted((Path(__file__).parent / "configs").glob("*.yaml"))


async def _main(args) -> None:
    samples = _load_locomo(Path(args.data))
    output_dir = Path(args.out)
    if args.all_configs:
        configs = _all_configs()
    else:
        configs = [Path(args.config)]

    summaries: dict[str, dict] = {}
    for cfg in configs:
        summary = await run_ablation(
            config_path=cfg,
            samples=samples,
            output_dir=output_dir / "locomo",
            limit=args.limit,
            concurrency=args.concurrency,
        )
        summaries[cfg.stem] = {
            k: v for k, v in summary.items() if k not in ("config", "models")
        }

    # Print comparison table
    print("\n=== LoCoMo comparison ===")
    print(f"{'config':<14} {'n':>5} {'judge':>8} {'F1':>6} {'EM':>6}")
    for name, s in summaries.items():
        print(f"{name:<14} {s['n']:>5} {s['judge_acc']:>8.3f} {s['f1']:>6.3f} {s['em']:>6.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to locomo10.json")
    p.add_argument(
        "--config",
        default="evals/configs/full.yaml",
        help="Single ablation config to run",
    )
    p.add_argument(
        "--all-configs",
        action="store_true",
        help="Run every config in evals/configs/",
    )
    p.add_argument("--out", default="evals/results", help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
