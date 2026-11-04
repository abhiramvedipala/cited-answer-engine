"""Grounded answering: numbered context in, inline [n] citations out, or NOT_IN_CORPUS."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from answer import llm
from retrieval import Hit

REFUSAL = "NOT_IN_CORPUS"

SYSTEM = f"""You answer questions using ONLY the numbered context passages provided.
Rules:
- End every factual sentence with the passage number(s) it came from, like [1] or [2][3].
- Use only numbers that appear in the context. Never cite anything else.
- Do not add facts that are not in the passages, even if you know them.
- If the passages do not contain the answer, reply with exactly {REFUSAL} and nothing else.
- Be concise: at most 5 sentences."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{i}] ({h.doc}, page {h.page_start})\n{h.text}" for i, h in enumerate(hits, 1)]
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)  # cited numbers not in the retrieved set
    refused: bool = False
    uncited: bool = False  # non-refusal answer with zero valid citations
    meta: dict = field(default_factory=dict)


CITE = re.compile(r"\[(\d+)\]")


def postprocess(text: str, hits: list[Hit]) -> Answer:
    text = text.strip()
    if text == REFUSAL or text.startswith(REFUSAL):
        return Answer(REFUSAL, refused=True)
    cited, dropped = [], []
    for n in dict.fromkeys(int(m) for m in CITE.findall(text)):
        (cited if 1 <= n <= len(hits) else dropped).append(n)
    for n in dropped:  # strip invalid markers so the reader never sees a dangling [9]
        text = text.replace(f"[{n}]", "")
    citations = [
        {
            "n": n,
            "doc": hits[n - 1].doc,
            "page": hits[n - 1].page_start,
            "chunk_id": hits[n - 1].chunk_id,
            "source": f"{hits[n - 1].doc}:{hits[n - 1].page_start}",
        }
        for n in cited
    ]
    return Answer(text, citations, dropped, uncited=not citations)


def answer_stream(question: str, hits: list[Hit]) -> Iterator[str | Answer]:
    """Yields text deltas, then the post-processed Answer."""
    if not hits:
        yield REFUSAL
        yield Answer(REFUSAL, refused=True)
        return
    parts, meta = [], {}
    for x in llm.stream(SYSTEM, build_prompt(question, hits), purpose="answer", max_tokens=600):
        if isinstance(x, dict):
            meta = x
        else:
            parts.append(x)
            yield x
    a = postprocess("".join(parts), hits)
    a.meta = meta
    yield a


def answer(question: str, hits: list[Hit]) -> Answer:
    *_, final = answer_stream(question, hits)
    return final
