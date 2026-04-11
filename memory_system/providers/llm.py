from litellm import acompletion
import instructor

from memory_system.core.models import IntentPrediction


# Instructor client for structured output
_instructor_client = None


def _get_instructor_client():
    global _instructor_client
    if _instructor_client is None:
        _instructor_client = instructor.from_litellm(acompletion)
    return _instructor_client


async def call_llm(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.1,
) -> str:
    """Call LLM via LiteLLM and return the response text."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools

    response = await acompletion(**kwargs)
    return response.choices[0].message.content or ""


async def predict_intent_llm(
    model: str,
    prompt: str,
) -> IntentPrediction:
    """Use instructor to get structured IntentPrediction from LLM."""
    client = _get_instructor_client()

    prediction = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_model=IntentPrediction,
        temperature=0.0,
        max_retries=1,
    )
    return prediction
