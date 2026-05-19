"""Multi-hop graph traversal in InMemoryGraphStore (BFS)."""

import pytest

from memory_system.core.memory_models import Entity, Relationship
from memory_system.providers.in_memory_stores import InMemoryGraphStore


async def _seed(store, user_id="u1"):
    for name in ["Alice", "Bob", "Carol", "Dave"]:
        await store.add_entity(Entity(name=name, entity_type="person", user_id=user_id))

    await store.add_relationship(Relationship(
        source_entity="Alice", target_entity="Bob",
        relation_type="manages", user_id=user_id,
    ))
    await store.add_relationship(Relationship(
        source_entity="Bob", target_entity="Carol",
        relation_type="works_with", user_id=user_id,
    ))
    await store.add_relationship(Relationship(
        source_entity="Carol", target_entity="Dave",
        relation_type="manages", user_id=user_id,
    ))


@pytest.mark.asyncio
async def test_traverse_one_hop():
    store = InMemoryGraphStore()
    await _seed(store)

    paths = await store.traverse("Alice", "u1", max_hops=1)
    assert len(paths) == 1
    assert paths[0][0].source_entity == "Alice"
    assert paths[0][0].target_entity == "Bob"


@pytest.mark.asyncio
async def test_traverse_two_hop_returns_partial_and_full_paths():
    store = InMemoryGraphStore()
    await _seed(store)

    paths = await store.traverse("Alice", "u1", max_hops=2)
    # Alice -> Bob (1-hop), Alice -> Bob -> Carol (2-hop) — 2 paths total
    assert len(paths) == 2
    lengths = sorted(len(p) for p in paths)
    assert lengths == [1, 2]


@pytest.mark.asyncio
async def test_traverse_max_hops_cutoff():
    store = InMemoryGraphStore()
    await _seed(store)

    paths_2 = await store.traverse("Alice", "u1", max_hops=2)
    paths_3 = await store.traverse("Alice", "u1", max_hops=3)
    # 3-hop reaches Dave; 2-hop does not
    assert len(paths_3) > len(paths_2)
    assert any(p[-1].target_entity == "Dave" for p in paths_3)
    assert not any(p[-1].target_entity == "Dave" for p in paths_2)


@pytest.mark.asyncio
async def test_traverse_relation_filter():
    store = InMemoryGraphStore()
    await _seed(store)

    paths = await store.traverse(
        "Alice", "u1", max_hops=3, relation_filter=["manages"]
    )
    # Only "manages" edges → Alice -> Bob, and dead end (Bob doesn't manage anyone)
    assert len(paths) == 1
    assert paths[0][0].relation_type == "manages"


@pytest.mark.asyncio
async def test_traverse_isolates_by_user():
    store = InMemoryGraphStore()
    await _seed(store, user_id="u1")
    await store.add_relationship(Relationship(
        source_entity="Alice", target_entity="Eve",
        relation_type="manages", user_id="u2",
    ))
    paths = await store.traverse("Alice", "u1", max_hops=1)
    assert all(p[0].user_id == "u1" for p in paths)
    assert not any(p[0].target_entity == "Eve" for p in paths)


@pytest.mark.asyncio
async def test_traverse_handles_cycles():
    store = InMemoryGraphStore()
    await store.add_relationship(Relationship(
        source_entity="A", target_entity="B",
        relation_type="r", user_id="u1",
    ))
    await store.add_relationship(Relationship(
        source_entity="B", target_entity="A",
        relation_type="r", user_id="u1",
    ))
    paths = await store.traverse("A", "u1", max_hops=5)
    # Cycle must not loop forever; bounded by visited-set
    assert all(len(p) <= 5 for p in paths)


@pytest.mark.asyncio
async def test_traverse_max_hops_zero_returns_empty():
    store = InMemoryGraphStore()
    await _seed(store)
    paths = await store.traverse("Alice", "u1", max_hops=0)
    assert paths == []


@pytest.mark.asyncio
async def test_recall_uses_multi_hop_when_graph_max_hops_gt_1():
    from memory_system.memory.memory import Memory as MemoryAPI
    from memory_system.providers.in_memory_stores import InMemoryMemoryStore

    graph = InMemoryGraphStore()
    await _seed(graph)
    # search_entities matches case-insensitively against entity name OR entity_type;
    # "Alice" is in the name → use that as the query
    api = MemoryAPI(store=InMemoryMemoryStore(), graph=graph)
    results = await api.recall("Alice", user_id="u1", k=10, graph_max_hops=2)
    texts = [r.memory.text for r in results]
    # Multi-hop arrow notation should appear for the 2-hop path
    assert any("->" in t for t in texts)
