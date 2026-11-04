"""Script-written half of the eval set: questions generated from section headings.

1. Find headings with PyMuPDF font sizes (spans >= 1.15x the document's median size).
2. Sample pages that have a heading and enough body text (seeded, so reproducible).
3. Ask the LLM for one factual question about that section plus a verbatim evidence quote.
4. Keep a question only if the quote really appears on the page; gold = that doc:page.

Run: uv run python -m eval.gen_questions --n 75 > eval/script_questions.jsonl
"""

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

from answer import llm

SYSTEM = """You write evaluation questions for a document search system.
Given a section heading and the page text, write ONE specific factual question that this page
answers, phrased the way a user would ask it (do not mention "this page" or "the section").
Return JSON only: {"question": "...", "evidence": "<a sentence copied verbatim from the page>"}"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def headings_by_page(path: Path) -> list[tuple[int, str, str]]:
    out = []
    with pymupdf.open(path) as pdf:
        pages = [p.get_text("dict") for p in pdf]
        sizes = [
            s["size"]
            for d in pages
            for b in d["blocks"]
            for ln in b.get("lines", [])
            for s in ln["spans"]
            if s["text"].strip()
        ]
        if not sizes:
            return out
        cut = statistics.median(sizes) * 1.15
        for i, d in enumerate(pages):
            heads = [
                s["text"].strip()
                for b in d["blocks"]
                for ln in b.get("lines", [])
                for s in ln["spans"]
                if s["size"] >= cut
                and 4 <= len(s["text"].strip()) <= 80
                and re.search(r"[A-Za-z]{3}", s["text"])
            ]
            text = pdf[i].get_text()
            if len(text) > 800:
                # PDFs without detectable heading fonts (NIST 800-207) fall back to the first line
                first = next((ln.strip() for ln in text.splitlines() if len(ln.strip()) > 6), "")
                out.append((i + 1, heads[0] if heads else first[:80], text))
    return out


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Stratified so the eval is not 90% Python docs: shares of the n questions per source.
    shares = {
        "python": 0.48,
        "nasa": 0.2,
        "nist-sp-800-53r5": 0.12,
        "nist-sp-800-61r2": 0.1,
        "nist-sp-800-207": 0.1,
    }
    groups: dict[str, list] = {g: [] for g in shares}
    for path in sorted(Path("data/corpus").glob("*.pdf")):
        g = "python" if path.stem.startswith("python-") else path.stem.split("-se")[0]
        groups[g] += [(path.stem, *h) for h in headings_by_page(path)]
    rng = random.Random(args.seed)
    candidates = []
    for g, share in shares.items():
        rng.shuffle(groups[g])
        candidates.append(groups[g][: round(args.n * share) + 8])  # +8 spare for rejected ones
    quota = {g: round(args.n * share) for g, share in shares.items()}
    got = dict.fromkeys(shares, 0)

    kept = 0
    for g, group in zip(shares, candidates, strict=True):
        for doc, page, heading, text in group:
            if got[g] >= quota[g]:
                break
            kept = _emit(doc, page, heading, text, kept, got, g)


def _emit(doc, page, heading, text, kept, got, g):
    """Generate and verify one question; returns the updated kept count."""
    raw, _ = llm.complete(
        SYSTEM,
        f"Heading: {heading}\n\nPage text:\n{text[:6000]}",
        purpose="gen_question",
        max_tokens=300,
    )
    try:
        q = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    except ValueError:
        return kept
    if len(norm(q.get("evidence", ""))) < 20 or norm(q["evidence"]) not in norm(text):
        print(f"skip {doc}:{page} evidence not on page", file=sys.stderr)
        return kept
    kept += 1
    got[g] += 1
    print(
        json.dumps(
            {
                "id": f"s{kept:03d}",
                "question": q["question"],
                "gold": [{"doc": doc, "page": page}],
                "answerable": True,
                "source": "script",
                "heading": heading,
                "evidence": q["evidence"],
            }
        )
    )
    return kept


if __name__ == "__main__":
    main()
