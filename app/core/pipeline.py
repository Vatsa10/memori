import time

from app.core.models import (
    BotConfig,
    ConversationTurn,
    IntentPrediction,
    PipelineResult,
    SmartPrompt,
)
from app.core.intent_predictor import IntentPredictor
from app.core.context_assembler import ContextAssembler, MemorySearcher
from app.core.prompt_builder import (
    build_smart_prompt,
    build_full_prompt_estimate,
    smart_prompt_to_messages,
)
from app.providers.llm import call_llm, predict_intent_llm
from typing import Optional


class Pipeline:
    def __init__(
        self,
        intent_predictor: IntentPredictor,
        memory_provider: Optional[MemorySearcher] = None,
    ):
        self.intent_predictor = intent_predictor
        self.context_assembler = ContextAssembler(memory_provider)

    async def run(
        self,
        bot_config: BotConfig,
        user_message: str,
        conversation_history: list[ConversationTurn],
    ) -> PipelineResult:
        latency = {}

        # Stage 1: Predict intent
        prediction, intent_ms = await self.intent_predictor.predict(
            bot_config=bot_config,
            user_message=user_message,
            recent_history=conversation_history[-3:],
            llm_predict_fn=predict_intent_llm,
        )
        latency["intent_prediction_ms"] = round(intent_ms, 2)

        # Stage 2: Assemble context
        t0 = time.perf_counter()
        context = await self.context_assembler.assemble(
            bot_config=bot_config,
            intent=prediction,
            user_message=user_message,
            full_history=conversation_history,
        )
        latency["context_assembly_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 3: Build smart prompt
        t0 = time.perf_counter()
        smart_prompt = build_smart_prompt(context)
        smart_prompt.full_prompt_estimate = build_full_prompt_estimate(
            bot_config, conversation_history
        )
        latency["prompt_build_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 4: Call LLM with minimal prompt
        t0 = time.perf_counter()
        messages = smart_prompt_to_messages(smart_prompt)
        # Add current user message
        messages.append({"role": "user", "content": user_message})

        response = await call_llm(
            model=bot_config.generation_model,
            messages=messages,
            tools=smart_prompt.tools if smart_prompt.tools else None,
        )
        latency["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        latency["total_ms"] = round(sum(latency.values()), 2)

        return PipelineResult(
            intent=prediction,
            smart_prompt=smart_prompt,
            response=response,
            latency_ms=latency,
        )
