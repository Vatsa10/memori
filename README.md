# Memory System

Production-ready memory + intent-aware context management for AI agents. Remember what users say, recall only what's relevant, respond smarter.

**118 tests | Qdrant + Neo4j + Redis | Works with any agent framework**

## Two Ways to Use

### 1. Standalone Memory (plug into any agent)

No intents, no YAML, no pipeline. Just memory.

```python
from memory_system import StandaloneMemory
from memory_system.providers.in_memory_stores import InMemoryMemoryStore

memory = StandaloneMemory(store=InMemoryMemoryStore())

# Add memories
await memory.add("User prefers morning deliveries", user_id="user1")
await memory.add("User lives in NYC", user_id="user1", importance=0.9)

# Search
results = await memory.search("delivery preferences", user_id="user1")

# Auto-extract from conversations
await memory.remember(
    messages=[
        {"role": "user", "content": "I just moved to San Francisco"},
        {"role": "assistant", "content": "Welcome to SF!"},
    ],
    user_id="user1",
)

# Get user profile (auto-built from memories)
profile = await memory.get_user_profile("user1")

# Build context window with token budget
context = await memory.get_context_window("user1", "delivery", token_budget=2000)
```

### 2. Full Pipeline (intent-aware context + memory)

Based on the [Haptik intent-aware context architecture](https://www.haptik.ai/): predict intent first, then retrieve only relevant context.

```python
from memory_system import MemorySystem

ms = MemorySystem.from_yaml("configs/my_bot.yaml")
result = await ms.chat("Where is my order?", user_id="user1")

print(result.response)
print(result.intent)              # IntentPrediction(name="check_order", confidence=0.95)
print(result.reduction_percent)   # 75% smaller prompts
print(result.memories_recalled)   # 3 memories used
```

## Architecture

```
User Message
    |
    v
+---------------------------+      +---------------------------+
| Intent Prediction (0-50ms)|      | Standalone Memory API     |
| keyword -> embedding -> LLM|      | .add() .search() .recall()|
+----------+----------------+      | .remember() .forget()     |
           |                       | .get_user_profile()       |
           v                       | .get_context_window()     |
+---------------------------+      +---------------------------+
| Context Assembly          |                |
| intent-specific only      |                |
+----------+----------------+                |
           |                                 |
           v                                 |
+---------------------------+                |
| RECALL User Memories  <---+----------------+
| Qdrant: semantic search   |
| Neo4j: entity graph       |
+----------+----------------+
           |
           v
+---------------------------+
| Smart Prompt Builder      |
| profile + memories +      |
| instructions + history    |
+----------+----------------+
           |
           v
+---------------------------+
| LLM Generation            |
+----------+----------------+
           |
           v
+---------------------------+
| REMEMBER New Facts        |
| LLM extracts facts ->    |
| dedup -> store in         |
| Qdrant + Neo4j            |
+---------------------------+
```

## Installation

```bash
pip install -e .                          # Core only
pip install -e ".[embeddings]"            # + sentence-transformers
pip install -e ".[production]"            # + Qdrant, Neo4j, Redis
pip install -e ".[all]"                   # Everything
```

## Production Setup

```bash
docker-compose up -d   # Starts Qdrant, Neo4j, Redis
```

```python
from memory_system import StandaloneMemory
from memory_system.providers.qdrant_store import QdrantMemoryStore
from memory_system.providers.neo4j_store import Neo4jGraphStore

memory = StandaloneMemory(
    store=QdrantMemoryStore(url="http://localhost:6333"),
    graph=Neo4jGraphStore(uri="bolt://localhost:7687", password="password"),
)
```

## Memory Features

### Memory Types

```python
from memory_system import MemoryType

await memory.add("User prefers tea", user_id="u1", memory_type=MemoryType.SEMANTIC)    # Facts
await memory.add("Met on March 5th",  user_id="u1", memory_type=MemoryType.EPISODIC)   # Events
await memory.add("Run with --verbose", user_id="u1", memory_type=MemoryType.PROCEDURAL) # How-tos
```

### Importance Scoring

```python
await memory.add("Critical: allergic to peanuts", user_id="u1", importance=1.0)
await memory.add("Mentioned liking blue", user_id="u1", importance=0.3)
```

### TTL / Expiry

```python
await memory.add("Promo code: SAVE20", user_id="u1", ttl=86400)  # Expires in 24h
```

### Memory Lifecycle

```python
# Decay old memories (reduce importance over time)
await memory.decay(user_id="u1")

# Merge similar memories into stronger ones
await memory.consolidate(user_id="u1")

# Remove expired + zero-importance memories
await memory.cleanup(user_id="u1")
```

### User Profiles

```python
profile = await memory.get_user_profile("u1")
print(profile.summary)        # "Prefers morning deliveries. Lives in NYC."
print(profile.properties)     # {"location": "NYC", "preferences": ["morning deliveries"]}
print(profile.memory_count)   # 15
```

### Context Window Builder

```python
# Fits profile + memories into a token budget
context = await memory.get_context_window("u1", "delivery", token_budget=2000)
# Returns formatted string ready for prompt injection
```

### Conversation Summarization

```python
summary = await memory.summarize_conversation(
    turns=[
        {"role": "user", "content": "I need to return my order"},
        {"role": "assistant", "content": "I can help with that..."},
    ],
    max_sentences=3,
)
```

### GDPR Compliance

```python
# Delete all memories for a user
await memory.forget(user_id="u1")

# Delete memories before a date
from datetime import datetime
await memory.forget(user_id="u1", before=datetime(2024, 1, 1))

# Delete single memory
await memory.delete(memory_id="...")
```

## Framework Integrations

### LangChain

```python
from memory_system.integrations.langchain import LangChainMiddleware

middleware = LangChainMiddleware(ms)
result = await middleware.ainvoke("Where is my order?")
```

### CrewAI

```python
from memory_system.integrations.crewai import CrewAIMemory

crew_memory = CrewAIMemory(memory, user_id="agent1")
context = await crew_memory.get_context("project requirements")
results = await crew_memory.search("deadlines", k=5)
```

### OpenAI Agents SDK

```python
from memory_system.integrations.openai_agents import OpenAIAgentsMemory

agent_memory = OpenAIAgentsMemory(memory, user_id="user1")

# Get tool definitions for the agent
tools = agent_memory.get_tool_definitions()

# Handle tool calls
result = await agent_memory.handle_tool_call("search_memory", {"query": "preferences"})
```

### Any Custom Agent

```python
# Before LLM call: recall relevant memories
memories = await memory.recall("delivery question", user_id="u1", k=5)
context = memory.format_memories(memories)

# Inject into your prompt
prompt = f"User Memories:\n{context}\n\nUser: {user_message}"

# After LLM call: remember new facts
await memory.remember(
    messages=[
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ],
    user_id="u1",
)
```

## Intent-Aware Pipeline (MemorySystem)

For chatbot/voice agent use cases where you want intent routing + context optimization.

### Bot Config (YAML)

```yaml
# configs/support_bot.yaml
bot_id: support
bot_name: SupportBot
base_instructions: "You are a helpful support agent."

intents:
  - name: check_order
    description: "Customer wants order status"
    keywords: ["order", "tracking", "shipment"]
    instructions: "Ask for order ID, then check status."
    tools: ["CheckOrderStatus"]
    max_history_turns: 2

  - name: return_refund
    description: "Customer wants to return/refund"
    keywords: ["return", "refund", "send back"]
    instructions: "Verify order, check eligibility, initiate return."
    tools: ["CheckOrderStatus", "CreateReturnLabel"]
    max_history_turns: 3
```

### Usage

```python
from memory_system import MemorySystem
from memory_system.providers.in_memory_stores import InMemoryMemoryStore

ms = MemorySystem.from_yaml(
    "configs/support_bot.yaml",
    memory_store=InMemoryMemoryStore(),  # Enable persistent memory
)

# Chat with memory
result = await ms.chat("Where is my order?", user_id="customer_123")

# Access standalone memory API directly
await ms.memory.add("VIP customer", user_id="customer_123", importance=0.9)
profile = await ms.memory.get_user_profile("customer_123")
```

### Auto-Discover Intents

Don't know what intents to define? Feed conversation logs:

```python
from memory_system.discovery.auto_intent import IntentDiscovery

discovery = IntentDiscovery()
result = discovery.discover([
    "Where is my order?",
    "Can you track my package?",
    "I want to return this",
    "How do I get a refund?",
    # ... hundreds of real messages
])

# Generate ready-to-use YAML config
yaml_config = discovery.to_yaml(result, "my_bot", "MyBot")
```

## Analytics Dashboard

```python
from memory_system.dashboard.app import create_dashboard

app = create_dashboard(ms)
# Run: uvicorn module:app --port 8080
# Visit: http://localhost:8080/dashboard/
```

Shows: intent distribution, prompt size reduction, latency breakdown, cache hit rate.

## Event Hooks

```python
from memory_system import EventType

ms.hooks.on(EventType.MEMORIES_RECALLED, lambda e: print(f"Recalled {e.data['count']} memories"))
ms.hooks.on(EventType.MEMORIES_STORED, lambda e: print(f"Stored {e.data['count']} new facts"))
ms.hooks.on(EventType.INTENT_PREDICTED, lambda e: print(f"Intent: {e.data}"))
```

## Database Support

| Component | Options |
|-----------|---------|
| **Vector Store** | Qdrant (production), InMemory (testing) |
| **Graph Store** | Neo4j (production), InMemory (testing) |
| **Session Store** | Redis (production), InMemory (testing) |
| **Cache** | Redis (shared), InMemory (single process) |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
memory_system/
  __init__.py                 # Exports: MemorySystem, StandaloneMemory, models
  _client.py                  # MemorySystem (full pipeline)
  core/
    models.py                 # BotConfig, IntentDefinition, ChatResponse
    memory_models.py          # Memory, MemoryType, UserProfile, MemoryStats
    protocols.py              # MemoryStore, GraphStore protocols
    pipeline.py               # 6-stage pipeline (intent + context + recall + prompt + LLM + remember)
    intent_predictor.py       # 3-tier hybrid: keyword -> embedding -> LLM
    context_assembler.py      # Intent-specific context assembly
    prompt_builder.py         # Token-optimized prompt construction
  memory/
    memory.py                 # StandaloneMemory (plug-and-play API)
    manager.py                # MemoryManager (pipeline integration)
    extractor.py              # LLM-based fact extraction
    lifecycle.py              # Decay, consolidate, cleanup
    profiles.py               # Auto-build user profiles
    context.py                # Token-budget context window builder
  providers/
    qdrant_store.py           # Production vector store
    neo4j_store.py            # Production graph store
    redis_store.py            # Production sessions + cache
    in_memory_stores.py       # Testing stores (no external deps)
  integrations/
    langchain.py              # LangChain adapter
    crewai.py                 # CrewAI adapter
    openai_agents.py          # OpenAI Agents SDK adapter
  discovery/
    auto_intent.py            # Auto-discover intents from logs
  dashboard/
    app.py                    # Analytics web UI
  server/
    app.py                    # Optional REST API server
```
