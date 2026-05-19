# Memory System

Production-ready memory + context layer for AI agents. Smart memory ops, bi-temporal
facts, hybrid retrieval, multi-modal ingestion, hierarchical summarization — every
P0/P1/P2 feature opt-in, every default backward-compatible.

**227 tests | DeepSeek V4-Flash + Gemini 2.5 + GPT-5.4 | Qdrant + Neo4j + Redis | LoCoMo / LongMemEval harness included**

## Why this exists

Most "memory layers" stop at store-and-search. This one:

- **Reasons about new facts** vs old ones (Mem0-parity smart ops: ADD/UPDATE/MERGE/DELETE/NOOP)
- **Tracks fact validity over time** (Graphiti-parity bi-temporal: `valid_from`/`valid_to`/`recorded_at`)
- **Stamps provenance** on every memory (source utterance, turn_id, extractor model, confidence)
- **Hybrid retrieval** (Supermemory-parity: BM25 + dense + multi-hop graph fused via RRF)
- **Cross-encoder reranking** (bge-reranker or Cohere)
- **Multi-modal ingestion** (PDF / URL / image / audio → chunks → memory)
- **Hierarchical summary tree** for compressive long-term recall
- **Built-in eval harness** for LoCoMo + LongMemEval with 5 ablation configs

## Three ways to use

### 1. Standalone Memory — plug into any agent

```python
from memory_system import StandaloneMemory
from memory_system.providers.in_memory_stores import InMemoryMemoryStore

memory = StandaloneMemory(store=InMemoryMemoryStore())

await memory.add("User prefers morning deliveries", user_id="u1", importance=0.8)
await memory.add("User lives in NYC", user_id="u1")

results = await memory.search("delivery preferences", user_id="u1")

# Auto-extract facts from a turn
await memory.remember(
    messages=[
        {"role": "user", "content": "I just moved to San Francisco"},
        {"role": "assistant", "content": "Welcome to SF!"},
    ],
    user_id="u1",
)

# Auto-built user profile
profile = await memory.get_user_profile("u1")

# Token-budgeted context window
context = await memory.get_context_window("u1", "delivery", token_budget=2000)
```

### 2. DB-grounded MemorySystem — full chat with memory

The primary API. Every response is grounded in:
1. **Knowledge base** (business docs, FAQs — searched in real-time)
2. **User memory** (facts about THIS user from past conversations)
3. **User profile** (auto-built from accumulated memories)
4. **Conversation history** (current session)

```python
from memory_system import MemorySystem
from memory_system.providers.in_memory_stores import InMemoryMemoryStore

ms = MemorySystem(
    instructions="You are a support agent for Acme Corp. Be concise and factual.",
    knowledge_store=InMemoryMemoryStore(),
    memory_store=InMemoryMemoryStore(),
    enable_smart_ops=True,         # P0: LLM-judged memory ops
    enable_summary_tree=True,      # P2: hierarchical rollups
)

await ms.add_knowledge("Returns accepted within 30 days of purchase.")
result = await ms.chat("How do I return something?", user_id="u1")

print(result.response)
print(result.memories_recalled)     # how many memories used
print(result.token_estimate)
print(result.latency_ms)             # per-stage breakdown
```

### 3. Intent-aware MemorySystem — optional legacy mode

For migrations from YAML-intent bots. Not the recommended path for new code.

```python
from memory_system import MemorySystem
ms = MemorySystem.from_yaml("configs/legacy_bot.yaml")  # if you already have YAML
result = await ms.chat("Where is my order?", user_id="u1")
print(result.intent)  # IntentPrediction(name="check_order", confidence=0.95)
```

## P0 — Memory intelligence

### Smart memory ops (Mem0 parity)

LLM judges what to do with each newly extracted fact vs existing similar memories:

```python
ms = MemorySystem(..., enable_smart_ops=True)

await ms.chat("I live in NYC", user_id="u1")
await ms.chat("Actually I just moved to SF", user_id="u1")
# → judge picks UPDATE: invalidates "lives in NYC", adds "lives in SF"
#   with metadata.supersedes pointing to the old fact
```

Actions: `ADD` / `UPDATE` / `MERGE` / `DELETE` / `NOOP`. Default off (preserves legacy behavior).

### Bi-temporal facts (Graphiti parity)

Every memory carries `valid_from`, `valid_to`, `recorded_at`, `superseded_by`:

```python
from datetime import datetime, timedelta, timezone

# Default search excludes invalidated facts
current = await memory.search("where do I live", user_id="u1")  # → SF

# Point-in-time query
past = await memory.recall_at(
    "where do I live",
    user_id="u1",
    as_of=datetime.now(timezone.utc) - timedelta(days=30),  # → NYC
)

# Full history (audit)
all_facts = await memory.store.search(
    "where do I live", user_id="u1", include_invalidated=True,
)
```

### Provenance

Every memory stores `source_text` (the utterance that triggered it), `turn_id`
(one per `remember()` call, shared across facts extracted from the same turn),
`confidence`, and `extractor_model`. Grep these for debugging.

## P1 — Retrieval quality

### Hybrid retrieval (BM25 + dense + graph, fused via RRF)

```python
from memory_system.retrieval import BM25Retriever, HybridRetriever

bm25 = BM25Retriever(mem_store)
retriever = HybridRetriever(
    mem_store,
    graph_store=graph_store,
    bm25=bm25,
    weights={"dense": 1.0, "bm25": 1.0, "graph": 0.5},
    graph_max_hops=2,
)
ms = MemorySystem(..., retriever=retriever)
```

BM25 cache auto-invalidates on memory writes via `HookManager` — no manual `bump()`.

### Cross-encoder reranking

```python
from memory_system.retrieval import CrossEncoderReranker, CohereReranker

ms = MemorySystem(
    ...,
    retriever=retriever,
    reranker=CrossEncoderReranker(model="BAAI/bge-reranker-base"),
    hybrid_top_n=20,
    rerank_top_k=5,
)
```

### Multi-hop graph traversal

`GraphStore.traverse(start_entity, user_id, max_hops=2, relation_filter=None)`
returns paths as `list[list[Relationship]]`. Memory.recall surfaces them as
`"Alice [manages] Bob -> Bob [works_at] Acme"` with hop-decayed scores.

## P2 — Ingestion breadth

### Multi-modal `ingest_document`

```python
# PDF → chunks → knowledge store
await ms.ingest_document("docs/policies.pdf", target="knowledge")

# URL → trafilatura main content → chunks
await ms.ingest_document("https://example.com/help", target="knowledge")

# Image → vision LLM description → memory
await ms.ingest_document("chart.png", target="memory", user_id="u1")

# Audio → Whisper transcription → chunks
await ms.ingest_document("call.mp3", target="memory", user_id="u1")

# Stream large docs (yields memories as each chunk persists)
async for mem in await ms.ingest_document("big.pdf", target="knowledge", stream=True):
    print("stored:", mem.id)
```

### Streaming ingest

```python
async def chunk_source():
    async for c in live_transcription_feed():
        yield c

async for mem in ms.ingest_stream(chunk_source(), target="memory", user_id="u1"):
    ...  # mem persists with batch_size concurrency
```

### Hierarchical summary tree

Turn → session → day → month rollups, stored as `EPISODIC` memories with
`metadata.summary_level`. Default `recall()` excludes them; `search_hierarchical`
opts in:

```python
ms = MemorySystem(..., enable_summary_tree=True)
# summarize_turn runs fire-and-forget after every chat() call

# Coarse-to-fine retrieval
results = await ms.search_hierarchical("life events last quarter", user_id="u1")
```

## Architecture

```
                          ┌─────────────────────────┐
        User Message ──▶  │  MemorySystem.chat()    │
                          └────────────┬────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
    ┌──────────────────┐   ┌────────────────────────┐   ┌─────────────────┐
    │ Knowledge store  │   │ Memory store +         │   │ User profile +  │
    │ (business docs)  │   │ Graph store +          │   │ history         │
    │                  │   │ Hybrid retriever +     │   │                 │
    │ dense search     │   │ BM25 + multi-hop +     │   │ auto-built      │
    │                  │   │ rerank (optional)      │   │                 │
    └────────┬─────────┘   └───────────┬────────────┘   └────────┬────────┘
             │                         │                         │
             └─────────┬───────────────┴─────────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ Grounded prompt assembly     │
        │ (token-budget aware)         │
        └──────────────┬───────────────┘
                       ▼
                ┌─────────────┐
                │ LLM         │ DeepSeek V4-Flash (default)
                └──────┬──────┘
                       ▼
        ┌──────────────────────────────────┐
        │ Extract facts (LLM)              │
        │  ↓                               │
        │ Smart_ops judge (optional)       │  ADD / UPDATE / MERGE / DELETE / NOOP
        │  ↓                               │
        │ Persist with bi-temporal +       │
        │ provenance fields                │
        │  ↓                               │
        │ Summary tree turn rollup (bg)    │
        └──────────────────────────────────┘
```

## Installation

```bash
pip install -e .                                 # core
pip install -e ".[retrieval,rerank]"             # + BM25 + cross-encoder
pip install -e ".[ingestion]"                    # + PDF/URL/audio ingestion
pip install -e ".[ingestion-pro,multimodal]"     # + pymupdf, Pillow
pip install -e ".[production]"                   # + Qdrant, Neo4j, Redis
pip install -e ".[all]"                          # everything
```

## Production setup

```bash
docker compose up -d   # Qdrant + Neo4j + Redis
```

```python
from memory_system import MemorySystem
from memory_system.providers.qdrant_store import QdrantMemoryStore
from memory_system.providers.neo4j_store import Neo4jGraphStore
from memory_system.providers.session import SessionStore  # or RedisSessionStore
from memory_system.retrieval import BM25Retriever, HybridRetriever, CrossEncoderReranker

knowledge = QdrantMemoryStore(collection="knowledge")
memory = QdrantMemoryStore(collection="user_memories")
graph = Neo4jGraphStore()
await knowledge.ensure_collection()
await memory.ensure_collection()
await graph.ensure_indexes()

ms = MemorySystem(
    instructions="...",
    knowledge_store=knowledge,
    memory_store=memory,
    graph_store=graph,
    enable_smart_ops=True,
    retriever=HybridRetriever(memory, graph, BM25Retriever(memory)),
    reranker=CrossEncoderReranker(),
    enable_summary_tree=True,
)
```

## Default model assignment

Set in `.env`. Override any role via env vars.

| Role | Model | $/M in/out |
|---|---|---|
| Generation | `deepseek/deepseek-v4-flash` | $0.14 / $0.28 |
| Extraction | `deepseek/deepseek-v4-flash` | $0.14 / $0.28 |
| Smart_ops judge | `deepseek/deepseek-v4-flash` | $0.14 / $0.28 |
| Summary rollup | `gemini/gemini-2.5-flash-lite` | $0.10 / $0.40 |
| Vision ingestion | `gemini/gemini-2.5-flash` | $0.30 / $2.50 |
| Eval judge | `openai/gpt-5.4-nano` | $0.20 / $1.25 |

Each provider plays to its strength: cheapest output (DeepSeek), cheapest input
+ long context (Gemini Flash-Lite), best vision/$ (Gemini Flash), best judge
precision/$ (GPT-5.4 nano).

## Memory features (Standalone API)

```python
from memory_system import MemoryType

# Memory types
await memory.add("User prefers tea", user_id="u1", memory_type=MemoryType.SEMANTIC)
await memory.add("Met on March 5th", user_id="u1", memory_type=MemoryType.EPISODIC)
await memory.add("Run with --verbose", user_id="u1", memory_type=MemoryType.PROCEDURAL)

# Importance / TTL / expiry
await memory.add("Critical: allergic to peanuts", user_id="u1", importance=1.0)
await memory.add("Promo: SAVE20", user_id="u1", ttl=86400)  # 24h

# Lifecycle
await memory.decay(user_id="u1")          # reduce importance over time
await memory.consolidate(user_id="u1")    # merge similar
await memory.cleanup(user_id="u1")        # drop expired + zero-importance

# Conversation summarization (flat)
summary = await memory.summarize_conversation(turns=[...], max_sentences=3)

# GDPR
await memory.forget(user_id="u1")
await memory.forget(user_id="u1", before=datetime(2024, 1, 1))
await memory.delete(memory_id="...")
```

## Framework integrations

```python
# LangChain
from memory_system.integrations.langchain import LangChainMiddleware
middleware = LangChainMiddleware(ms)

# CrewAI
from memory_system.integrations.crewai import CrewAIMemory
crew_memory = CrewAIMemory(memory, user_id="agent1")

# OpenAI Agents SDK
from memory_system.integrations.openai_agents import OpenAIAgentsMemory
agent_memory = OpenAIAgentsMemory(memory, user_id="u1")
tools = agent_memory.get_tool_definitions()
```

## Evaluation

`evals/` contains a full harness for **LoCoMo** and **LongMemEval** with 5 ablation
configs (`baseline`, `smart_ops`, `hybrid`, `summaries`, `full`). See
[evals/README.md](evals/README.md) for setup and run instructions.

```bash
# Verify everything wires together (no API spend, mocked LLMs)
pytest tests/test_eval_harness.py -v

# Live: full ablation matrix vs LoCoMo
python -m evals.run_locomo --data evals/data/locomo10.json --all-configs
```

Expect ~$15-25 total to run all five configs against both benchmarks.

## Event hooks

```python
from memory_system import EventType
ms.hooks.on(EventType.MEMORIES_RECALLED, lambda e: print(f"recalled {e.data['count']}"))
ms.hooks.on(EventType.MEMORIES_STORED,   lambda e: print(f"stored {e.data['count']}"))
```

`HookManager` auto-wires BM25 cache invalidation when a `HybridRetriever` with
`bm25` is attached to `MemorySystem` — no manual wiring needed.

## Database support

| Component | Production | Testing |
|---|---|---|
| Vector store | Qdrant | InMemoryMemoryStore |
| Graph store | Neo4j | InMemoryGraphStore |
| Session store | Redis | InMemorySessionStore |
| Cache | Redis | InMemoryCache |

## Running tests

```bash
pip install -e ".[dev,retrieval,rerank,ingestion]"
pytest tests/ -v
```

**227 tests** cover: standalone Memory, bi-temporal, provenance, smart_ops, graph
traversal, BM25 + RRF, reranking, multi-modal ingestion, streaming, summary tree,
end-to-end eval harness.

## Project structure

```
memory_system/
├── __init__.py                # Exports: MemorySystem, StandaloneMemory, ...
├── _client.py                 # MemorySystem (grounded chat + optional intent mode)
├── core/
│   ├── memory_models.py       # Memory (bi-temporal + provenance), MemoryType,
│   │                          # SummaryNode, SummaryLevel, MemorySearchResult
│   ├── models.py              # ChatResponse, ConversationTurn, BotConfig (legacy)
│   ├── protocols.py           # MemoryStore (search/invalidate/search_at),
│   │                          # GraphStore (traverse), KnowledgeSearcher
│   ├── pipeline.py            # Legacy 6-stage intent pipeline
│   ├── intent_predictor.py    # Legacy: keyword → embedding → LLM
│   ├── context_assembler.py
│   └── prompt_builder.py
├── memory/
│   ├── memory.py              # StandaloneMemory + recall_at + multi-hop graph
│   ├── manager.py             # MemoryManager (used by legacy Pipeline)
│   ├── extractor.py           # LLM fact extraction + confidence + provenance
│   ├── smart_ops.py           # ADD/UPDATE/MERGE/DELETE/NOOP judge + executor
│   ├── summary_tree.py        # Hierarchical turn/session/day/month rollups
│   ├── lifecycle.py           # Decay / consolidate / cleanup
│   ├── profiles.py            # Auto-build user profiles
│   └── context.py             # Token-budget context window
├── retrieval/
│   ├── bm25.py                # BM25Retriever + per-user cache + bump
│   ├── fusion.py              # Reciprocal Rank Fusion
│   ├── hybrid.py              # HybridRetriever (dense + BM25 + graph)
│   └── rerank.py              # Reranker protocol + CrossEncoder + Cohere
├── ingestion/
│   ├── chunker.py             # SemanticChunker (tiktoken w/ fallback)
│   ├── detect.py              # detect_source_type
│   ├── pdf.py                 # pypdf / pymupdf
│   ├── url.py                 # httpx + trafilatura
│   ├── image.py               # vision LLM
│   └── audio.py               # litellm transcription
├── providers/
│   ├── qdrant_store.py        # Production vector store (bi-temporal aware)
│   ├── neo4j_store.py         # Production graph store (multi-hop traverse)
│   ├── redis_store.py         # Sessions + cache
│   ├── in_memory_stores.py    # Testing stores
│   ├── session.py
│   └── memory.py              # Legacy KnowledgeSearcher impl
├── integrations/
│   ├── langchain.py
│   ├── crewai.py
│   ├── openai_agents.py
│   └── litellm_wrapper.py
├── discovery/
│   └── auto_intent.py         # KMeans + TF-IDF intent discovery
├── dashboard/
│   └── app.py                 # Analytics web UI
├── server/
│   └── app.py                 # Optional REST API
├── config/
│   ├── settings.py
│   ├── loader.py
│   └── factory.py
├── hooks.py
└── analytics.py

evals/
├── llm_factory.py             # Role → model routing
├── metrics.py                 # EM + token-F1 + LLM-judge
├── runner.py                  # Shared ingest→probe→score harness
├── run_locomo.py
├── run_longmemeval.py
├── configs/                   # 5 ablations
└── README.md
```
