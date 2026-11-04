"""Small LLM judge for faithfulness and answer relevance.

I used this instead of Ragas: Ragas adds a heavy dependency tree and its own prompts, and
its default judge is OpenAI. This judge is two short prompts on the same Anthropic model, so
cost is logged by the same wrapper. Limitation: the judge is the same model family that wrote
the answers, which tends to flatter them; see Known limits in the README.
"""

import json
import re

from answer import llm
from retrieval import Hit

FAITH_SYSTEM = """You check whether an answer is supported by the given context passages.
Split the answer into its distinct factual claims. For each claim decide if the context
directly supports it. Ignore citation markers like [1]. Return JSON only:
{"total": <number of claims>, "supported": <number supported by the context>}"""

REL_SYSTEM = """You rate how well an answer addresses a question, ignoring whether it is true.
5 = fully and directly answers it, 3 = partly answers it, 1 = does not answer it.
Return JSON only: {"score": <1-5>}"""


def _json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0)) if m else {}


def faithfulness(answer: str, hits: list[Hit]) -> tuple[float | None, float]:
    ctx = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1))
    raw, meta = llm.complete(
        FAITH_SYSTEM, f"Context:\n{ctx}\n\nAnswer:\n{answer}", "judge_faith", max_tokens=100
    )
    try:
        d = _json(raw)
        total, supported = int(d["total"]), int(d["supported"])
    except (KeyError, ValueError, TypeError):
        return None, meta.get("cost_usd", 0.0)
    return (min(supported, total) / total if total > 0 else None), meta["cost_usd"]


def relevance(question: str, answer: str) -> tuple[float | None, float]:
    raw, meta = llm.complete(
        REL_SYSTEM, f"Question: {question}\n\nAnswer: {answer}", "judge_rel", max_tokens=50
    )
    try:
        score = int(_json(raw)["score"])
    except (KeyError, ValueError, TypeError):
        return None, meta.get("cost_usd", 0.0)
    return (min(max(score, 1), 5) - 1) / 4, meta["cost_usd"]
