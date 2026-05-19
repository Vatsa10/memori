"""LLM-based fact extraction from conversation turns."""

from typing import Callable, Optional

from pydantic import BaseModel

from memory_system.core.memory_models import (
    Entity,
    Memory,
    MemoryExtractionResult,
    MemoryType,
    Relationship,
)


class ExtractedFact(BaseModel):
    text: str
    memory_type: str = "semantic"  # semantic, episodic, procedural
    importance: float = 0.5  # 0.0-1.0
    confidence: float = 0.8  # extractor's confidence in this fact


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relation: str


class ExtractionOutput(BaseModel):
    facts: list[ExtractedFact] = []
    entities: list[ExtractedEntity] = []
    relationships: list[ExtractedRelationship] = []


EXTRACTION_PROMPT = """Extract important facts, entities, and relationships from this conversation turn.

User: {user_message}
Assistant: {assistant_response}

Extract:
1. FACTS: Key information worth remembering (user preferences, decisions, important details).
   For each fact, also output `confidence` ∈ [0,1] reflecting how certain you are the fact is true
   and worth storing (0.9+ = explicit statement; 0.5-0.7 = inferred; <0.5 = speculative).
2. ENTITIES: Named things (people, products, places, concepts)
3. RELATIONSHIPS: How entities relate (prefers, bought, lives_in, works_at)

Only extract genuinely useful information. Skip greetings, filler, and generic statements.
If there's nothing worth remembering, return empty lists."""


async def extract_memories(
    user_message: str,
    assistant_response: str,
    user_id: str,
    llm_fn: Optional[Callable] = None,
    model: str = "groq/llama-3.1-8b-instant",
    custom_prompt: Optional[str] = None,
    source_text: Optional[str] = None,
    turn_id: Optional[str] = None,
) -> MemoryExtractionResult:
    """Extract facts, entities, and relationships from a conversation turn."""

    if not llm_fn:
        return MemoryExtractionResult()

    try:
        import instructor
        from litellm import acompletion

        client = instructor.from_litellm(acompletion)

        template = custom_prompt or EXTRACTION_PROMPT
        prompt = template.format(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        output = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=ExtractionOutput,
            temperature=0.0,
            max_retries=1,
        )
    except Exception:
        return MemoryExtractionResult()

    # Convert to domain models
    memories = []
    for fact in output.facts:
        mem_type = MemoryType.SEMANTIC
        if fact.memory_type == "episodic":
            mem_type = MemoryType.EPISODIC
        elif fact.memory_type == "procedural":
            mem_type = MemoryType.PROCEDURAL

        memories.append(Memory(
            text=fact.text,
            memory_type=mem_type,
            user_id=user_id,
            source="chat",
            importance=max(0.0, min(1.0, fact.importance)),
            source_text=source_text if source_text is not None else user_message,
            turn_id=turn_id,
            confidence=max(0.0, min(1.0, fact.confidence)),
            extractor_model=model,
        ))

    entities = [
        Entity(name=e.name, entity_type=e.entity_type, user_id=user_id)
        for e in output.entities
    ]

    relationships = [
        Relationship(
            source_entity=r.source,
            target_entity=r.target,
            relation_type=r.relation,
            user_id=user_id,
        )
        for r in output.relationships
    ]

    return MemoryExtractionResult(
        memories=memories,
        entities=entities,
        relationships=relationships,
    )
