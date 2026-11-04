"""Resolve hand-written questions to gold pages and merge with script questions.

Each hand question carries an anchor: a phrase that must appear on exactly one page of
its document. That page is the gold page, so gold labels are checked by search, not memory.

Run: uv run python -m eval.build_questions
"""

import json
import sys
from pathlib import Path

from eval.pages import find, load

ROOT = Path("eval")


def main() -> int:
    pages = load()
    out, bad = [], 0
    for i, line in enumerate(ROOT.joinpath("hand_questions.tsv").read_text().splitlines(), 1):
        doc, anchor, question = line.split("\t")
        hits = find(
            pages, anchor
        )  # corpus-wide: boilerplate repeated across PDFs is all valid gold
        if not 1 <= len(hits) <= 6 or doc not in {d for d, _ in hits}:
            print(f"line {i}: {len(hits)} pages for {anchor[:60]!r} in {doc}: {hits[:5]}")
            bad += 1
            continue
        out.append(
            {
                "id": f"h{len(out) + 1:03d}",
                "question": question,
                "gold": [{"doc": doc, "page": pg} for _, pg in hits],
                "answerable": True,
                "source": "hand",
                "evidence": anchor,
            }
        )
    script = [
        json.loads(x) for x in ROOT.joinpath("script_questions.jsonl").read_text().splitlines()
    ]
    unans = [
        json.loads(x) for x in ROOT.joinpath("unanswerable.jsonl").read_text().splitlines() if x
    ]
    import random

    keep = 150 - len(
        out
    )  # answerable set is exactly 150: all hand questions + seeded script sample
    if len(script) > keep:
        script = sorted(random.Random(7).sample(script, keep), key=lambda q: q["id"])
    allq = script + out + unans
    ROOT.joinpath("questions.jsonl").write_text("\n".join(json.dumps(q) for q in allq) + "\n")
    print(f"script={len(script)} hand={len(out)} unanswerable={len(unans)} bad_anchor={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
