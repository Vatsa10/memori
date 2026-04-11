import time
from typing import Callable, Optional

from memory_system.core.models import (
    BotConfig,
    ConversationTurn,
    IntentPrediction,
    PipelineResult,
)
from memory_system.core.intent_predictor import IntentPredictor
from memory_system.core.context_assembler import ContextAssembler, MemorySearcher
from memory_system.core.prompt_builder import (
    build_smart_prompt,
    build_full_prompt_estimate,
    smart_prompt_to_messages,
)
from memory_system.providers.llm import call_llm as default_call_llm
from memory_system.providers.llm import predict_intent_llm as default_predict_intent_llm


class Pipeline:
    def __init__(
        self,
        intent_predictor: IntentPredictor,
        memory_provider: Optional[MemorySearcher] = None,
        llm_fn: Optional[Callable] = None,
        intent_llm_fn: Optional[Callable] = None,
    ):
        self.intent_predictor = intent_predictor
        self.context_assembler = ContextAssembler(memory_provider)
        self._llm_fn = llm_fn or default_call_llm
        self._intent_llm_fn = intent_llm_fn or default_predict_intent_llm

    async def run(
        self,
        bot_config: BotConfig,
        user_message: str,
        conversation_history: list[ConversationTurn],
        cached_intent: Optional[IntentPrediction] = None,
    ) -> PipelineResult:
        latency = {}

        # Stage 1: Predict intent (or use cached)
        if cached_intent:
            prediction = cached_intent
            latency["intent_prediction_ms"] = 0.0
        else:
            prediction, intent_ms = await self.intent_predictor.predict(
                bot_config=bot_config,
                user_message=user_message,
                recent_history=conversation_history[-3:],
                llm_predict_fn=self._intent_llm_fn,
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
        messages.append({"role": "user", "content": user_message})

        response = await self._llm_fn(
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
