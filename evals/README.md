# memory-system Eval Harness

Benchmarks the memory-system on **LoCoMo** and **LongMemEval** with 5 ablation
configs so you can table the lift of each feature.

## What you get

- `llm_factory.py` — role-based model routing (DeepSeek V4-Flash + Gemini Flash-Lite + GPT-5.4 nano judge)
- `metrics.py` — EM + token-F1 + LLM-judge (GPT-5.4 nano)
- `configs/` — 5 ablations: `baseline`, `smart_ops`, `hybrid`, `summaries`, `full`
- `runner.py` — shared harness (ingest → probe → score)
- `run_locomo.py`, `run_longmemeval.py` — benchmark-specific loaders

## Setup

```bash
# 1. Install eval extras
pip install -e ".[all,retrieval,rerank,ingestion]"

# 2. Confirm .env has these (in repo root)
#    OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY
#    Optional model overrides (defaults below already set in .env):
#    INTENT_MODEL, GENERATION_MODEL, EXTRACTION_MODEL,
#    SUMMARY_MODEL, VISION_MODEL, EVAL_JUDGE_MODEL

# 3. Get the datasets
mkdir -p evals/data
# LoCoMo
git clone https://github.com/snap-research/locomo evals/data/locomo_repo
cp evals/data/locomo_repo/data/locomo10.json evals/data/
# LongMemEval
git clone https://github.com/xiaowu0162/LongMemEval evals/data/longmemeval_repo
cp evals/data/longmemeval_repo/data/longmemeval_s.json evals/data/
```

## Verify setup (no API spend)

Runs the 8-test harness suite with mocked LLMs — confirms imports, dataset loaders,
metrics, and the end-to-end runner all wire correctly before you burn any tokens.

```bash
pytest tests/test_eval_harness.py -v
```

If all 8 pass, you're ready for live runs.

## Run

### Smoke test (1 sample, fastest config)

```bash
python -m evals.run_locomo \
  --data evals/data/locomo10.json \
  --config evals/configs/baseline.yaml \
  --limit 1 --concurrency 2
```

### Full LoCoMo, one ablation

```bash
python -m evals.run_locomo \
  --data evals/data/locomo10.json \
  --config evals/configs/full.yaml
```

### Full LoCoMo, every ablation (the headline comparison)

```bash
python -m evals.run_locomo \
  --data evals/data/locomo10.json \
  --all-configs
```

### LongMemEval

```bash
python -m evals.run_longmemeval \
  --data evals/data/longmemeval_s.json \
  --all-configs --limit 100
```

## Output

Per-run, two files per ablation in `evals/results/<benchmark>/`:

- `<config>.jsonl` — one line per probe with `pred`, `gold`, `em`, `f1`, `judge`, `latency_ms`
- `<config>.summary.json` — aggregated metrics + active model assignment

## Model assignment (current defaults)

| Role | Model | $/M in/out |
|---|---|---|
| Generation | `deepseek/deepseek-v4-flash` | $0.14 / $0.28 |
| Extraction | `deepseek/deepseek-v4-flash` | $0.14 / $0.28 |
| Smart_ops judge | `deepseek/deepseek-v4-flash` | $0.14 / $0.28 |
| Summary rollup | `gemini/gemini-2.5-flash-lite` | $0.10 / $0.40 |
| Vision (image/PDF) | `gemini/gemini-2.5-flash` | $0.30 / $2.50 |
| Eval judge | `openai/gpt-5.4-nano` | $0.20 / $1.25 |

Override any role via env: `EVAL_MODEL_GENERATION=...`, etc.

## Cost estimate (full LoCoMo, 10 convos × ~600 turns + ~200 probes)

Computed against current routing — DeepSeek V4-Flash for bulk, Gemini Flash-Lite
for summaries, GPT-5.4 nano for the judge.

| Role | Tokens (in/out) | Model | $ |
|---|---|---|---|
| Generation (probes only) | 200K / 50K | deepseek-v4-flash @ $0.14/$0.28 | $0.04 |
| Extraction (every turn) | 3M / 200K | deepseek-v4-flash | $0.48 |
| Smart_ops judge (~3× per turn) | 9M / 500K | deepseek-v4-flash | $1.40 |
| Summary tree (per turn + rollups) | 4M / 200K | gemini-2.5-flash-lite @ $0.10/$0.40 | $0.48 |
| Eval judge (200 probes) | 100K / 40K | gpt-5.4-nano @ $0.20/$1.25 | $0.07 |
| **Total per ablation (full config)** | — | — | **~$2.47** |

Baseline ablation (no smart_ops, no summaries) drops to ~$0.55/run.
5 ablations × (LoCoMo + LongMemEval) typically lands in the **$15-25 total** range.

**With DeepSeek prompt caching** (~70% hit rate on repeated judge/summary templates),
expect another ~50-60% reduction on those rows.

## Reading the results

Aggregated comparison printed at end of `--all-configs` run. **Numbers below are
illustrative placeholders** to show the expected shape — your actual scores depend
on dataset version, LLM model versions, and retrieval-quality randomness.

```
=== LoCoMo comparison ===
config         n   judge     F1     EM
baseline     200   0.412  0.380  0.220
smart_ops    200   0.488  0.421  0.235      ← + knowledge-update gains
hybrid       200   0.523  0.470  0.245      ← + multi-hop / rare-term gains
summaries    200   0.495  0.430  0.225      ← + cross-session gains
full         200   0.612  0.522  0.275      ← all features stacked
```

Each row uses the same probe set; the deltas show which features actually moved
the needle on your dataset.

### What each ablation isolates

| Config | Adds | Expected to help on |
|---|---|---|
| `baseline` | nothing | floor — pure dense recall, append-only memory |
| `smart_ops` | P0.1 LLM-judged ops | LongMemEval `knowledge-update`, LoCoMo `temporal` (contradiction handling) |
| `hybrid` | P1.4/1.5/1.6 BM25 + rerank + 2-hop graph | LoCoMo `multi-hop`, queries with rare names / IDs |
| `summaries` | P2.9 turn → session → day rollups | LoCoMo `open-domain`, LongMemEval `multi-session` |
| `full` | all of the above | aggregate ceiling |

## Per-question debugging

Each `.jsonl` line has the failure mode you need:

```jsonl
{"sample_id":"loc_001","question":"Where does Alice live now?","gold":"Seattle",
 "pred":"NYC","em":0.0,"f1":0.0,"judge":false,
 "judge_rationale":"Model returned outdated location; user moved to Seattle in session 3.",
 "latency_ms":847,"memories_recalled":5}
```

Grep these to find systematic failures (e.g., temporal drift = bi-temporal not working).

## Adding a new ablation

Drop a YAML into `evals/configs/`:

```yaml
# evals/configs/my_test.yaml
name: my_test
description: What you're isolating.
features:
  enable_smart_ops: true
  enable_summary_tree: false
  use_hybrid_retriever: true
  use_reranker: false
  graph_max_hops: 2
```

Then `--all-configs` automatically picks it up. No code changes needed.

## Notes

- Uses **InMemoryMemoryStore + InMemoryGraphStore** for evals (deterministic, fast, no Docker).
- Same `MemorySystem` constructor as production — no special eval-only code paths.
- Resumable: each ablation streams to JSONL; re-running appends. Delete the file to start fresh.
- Bounded concurrency (`--concurrency 4` default) protects you from rate-limits.
- LLM-judge falls back to token-F1 ≥ 0.5 if the judge call itself fails, so a flaky
  judge doesn't crash the run — but those rows have a `judge_rationale` indicating
  the fallback. Grep for `"judge failed"` in JSONL to find them.
