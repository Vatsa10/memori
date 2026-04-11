from fastapi import APIRouter, HTTPException

from smartcontext.core.models import ChatRequest, ChatResponse, ConversationTurn
from smartcontext.core.pipeline import Pipeline
from smartcontext.core.intent_predictor import IntentPredictor
from smartcontext.config.registry import registry
from smartcontext.providers.session import session_store
from smartcontext.providers.memory import InMemoryProvider

router = APIRouter(prefix="/api", tags=["chat"])

# Shared pipeline instance (initialized in main.py lifespan)
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        predictor = IntentPredictor()
        memory = InMemoryProvider()
        _pipeline = Pipeline(intent_predictor=predictor, memory_provider=memory)
    return _pipeline


def set_pipeline(pipeline: Pipeline):
    global _pipeline
    _pipeline = pipeline


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    bot_config = registry.get(request.bot_id)
    if not bot_config:
        raise HTTPException(status_code=404, detail=f"Bot '{request.bot_id}' not found")

    pipeline = get_pipeline()

    # Get conversation history
    history = session_store.get_history(request.session_id)

    # Run the intent-aware pipeline
    result = await pipeline.run(
        bot_config=bot_config,
        user_message=request.message,
        conversation_history=history,
    )

    # Store the turns
    session_store.add_turn(
        request.session_id,
        ConversationTurn(role="user", content=request.message),
    )
    session_store.add_turn(
        request.session_id,
        ConversationTurn(role="assistant", content=result.response),
    )

    # Calculate reduction
    smart_tokens = result.smart_prompt.token_estimate
    full_tokens = result.smart_prompt.full_prompt_estimate
    reduction = (
        ((full_tokens - smart_tokens) / full_tokens * 100) if full_tokens > 0 else 0.0
    )

    return ChatResponse(
        response=result.response,
        intent=result.intent,
        token_estimate=smart_tokens,
        full_prompt_estimate=full_tokens,
        reduction_percent=round(reduction, 1),
        latency_ms=result.latency_ms,
    )
