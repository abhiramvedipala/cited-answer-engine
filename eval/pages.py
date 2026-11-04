"""Cached page text plus a phrase locator, used to pin gold pages for hand-written questions."""

import json
import re
from pathlib import Path

from ingest.parse import parse_dir

CACHE = Path("data/cache/pages.jsonl")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("­", "")).strip().lower()


def load() -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("w") as f:
            for p in parse_dir(Path("data/corpus")):
                f.write(json.dumps({"doc": p.doc, "page": p.page, "text": norm(p.text)}) + "\n")
    return [json.loads(line) for line in CACHE.open()]


def find(pages: list[dict], phrase: str, doc: str | None = None) -> list[tuple[str, int]]:
    ph = norm(phrase)
    return [
        (p["doc"], p["page"]) for p in pages if ph in p["text"] and (not doc or p["doc"] == doc)
    ]
