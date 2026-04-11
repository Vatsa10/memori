from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: Optional[str] = None


class IntentDefinition(BaseModel):
    name: str
    description: str
    keywords: list[str] = []
    instructions: str
    example: Optional[str] = None
    tools: list[str] = []
    max_history_turns: int = 2
    retrieval_query_prefix: Optional[str] = None  # Custom prefix for Qdrant queries


class BotConfig(BaseModel):
    bot_id: str
    bot_name: str
    base_instructions: str  # Always included (identity, tone, constraints)
    intents: list[IntentDefinition]
    fallback_instructions: str = ""
    intent_model: str = "groq/llama-3.1-8b-instant"
    generation_model: str = "groq/llama-3.3-70b-versatile"
    confidence_threshold: float = 0.6
    keyword_threshold: float = 0.4  # Min keyword overlap score to skip LLM
    embedding_threshold: float = 0.7  # Min cosine similarity to skip LLM


class PredictionMethod(str, Enum):
    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    LLM = "llm"
    FALLBACK = "fallback"


class IntentPrediction(BaseModel):
    intent_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    method: PredictionMethod = PredictionMethod.LLM


class AssembledContext(BaseModel):
    instructions: str  # base_instructions + intent-specific instructions
    history: list[ConversationTurn]  # Trimmed to last N turns
    retrieved_context: Optional[str] = None  # Top Qdrant results
    example: Optional[str] = None
    tools: list[dict] = []  # Filtered tool schemas


class SmartPrompt(BaseModel):
    system_message: str
    history: list[ConversationTurn]
    tools: list[dict] = []
    token_estimate: int = 0
    full_prompt_estimate: int = 0  # What it would have been without filtering


class ChatRequest(BaseModel):
    bot_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    knowledge_used: int = 0
    memories_recalled: int = 0
    memories_stored: int = 0
    token_estimate: int = 0
    latency_ms: dict[str, float] = {}
    # Legacy fields (populated when using intent-based Pipeline)
    intent: Optional[IntentPrediction] = None
    full_prompt_estimate: int = 0
    reduction_percent: float = 0.0


class PipelineResult(BaseModel):
    intent: IntentPrediction
    smart_prompt: SmartPrompt
    response: str
    latency_ms: dict[str, float] = {}
    memories_recalled: int = 0
    memories_stored: int = 0
