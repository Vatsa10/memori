"""Eval metrics: F1, exact-match, and LLM-as-judge."""

import re
import string
from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field

from evals.llm_factory import get_llm_fn, get_model


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punct/articles/whitespace."""
    text = (text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


JUDGE_PROMPT = """You are grading whether a model's answer is correct given the gold answer.

Question: {question}
Gold answer: {gold}
Model answer: {pred}

Is the model answer correct? It is correct if it conveys the same information
as the gold answer, even with different wording. Partial answers count as
incorrect. Off-topic, evasive, or hedged answers count as incorrect.

Respond with JSON matching JudgeVerdict."""


class JudgeVerdict(BaseModel):
    correct: bool
    rationale: str = Field(default="", description="One short sentence.")


async def llm_judge(
    question: str,
    pred: str,
    gold: str,
    *,
    model: Optional[str] = None,
) -> JudgeVerdict:
    """Use the configured eval judge model to grade a prediction."""
    judge_model = model or get_model("eval_judge")
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)

    try:
        import instructor
        from litellm import acompletion

        client = instructor.from_litellm(acompletion)
        verdict = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_model=JudgeVerdict,
            temperature=0.0,
            max_retries=1,
        )
        return verdict
    except Exception as e:
        # Fall back to token F1 ≥ 0.5 as a structural backup
        score = token_f1(pred, gold)
        return JudgeVerdict(
            correct=score >= 0.5,
            rationale=f"judge failed ({e}); fell back to token-F1={score:.2f}",
        )


def aggregate(records: list[dict]) -> dict:
    """Aggregate per-question scores into summary metrics.

    Each record must have keys: 'em', 'f1', 'judge' (bool), optionally 'category'.
    """
    if not records:
        return {"n": 0, "em": 0.0, "f1": 0.0, "judge_acc": 0.0, "by_category": {}}

    n = len(records)
    em = sum(r["em"] for r in records) / n
    f1 = sum(r["f1"] for r in records) / n
    judge_acc = sum(1 for r in records if r.get("judge")) / n

    by_category: dict[str, dict] = {}
    for r in records:
        cat = r.get("category")
        if not cat:
            continue
        b = by_category.setdefault(cat, {"n": 0, "em": 0, "f1": 0, "judge": 0})
        b["n"] += 1
        b["em"] += r["em"]
        b["f1"] += r["f1"]
        b["judge"] += int(bool(r.get("judge")))
    for cat, b in by_category.items():
        if b["n"] > 0:
            b["em"] /= b["n"]
            b["f1"] /= b["n"]
            b["judge_acc"] = b["judge"] / b["n"]
            b.pop("judge")

    return {
        "n": n,
        "em": em,
        "f1": f1,
        "judge_acc": judge_acc,
        "by_category": by_category,
    }
