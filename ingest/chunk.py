"""Token-window chunker that remembers which pages each chunk came from.

Windows run across page boundaries inside one document, so a 1024-token chunk can
span two pages. Each chunk keeps page_start and page_end; a citation points at
page_start and recall counts a hit if the gold page is anywhere in the span.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ingest.parse import Page

# text -> list of (char_start, char_end) per token
OffsetFn = Callable[[str], list[tuple[int, int]]]


@dataclass(frozen=True)
class Chunk:
    doc: str
    page_start: int
    page_end: int
    chunk_idx: int
    text: str


def config_name(size: int, overlap: int) -> str:
    return f"s{size}_o{overlap}"


def chunk_id(config: str, c: Chunk) -> str:
    return f"{config}:{c.doc}:{c.chunk_idx}"


def chunk_doc(pages: Sequence[Page], size: int, overlap: int, offsets: OffsetFn) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    tokens: list[tuple[int, int, int]] = []  # (index into pages, start, end)
    for pi, p in enumerate(pages):
        tokens.extend((pi, s, e) for s, e in offsets(p.text))
    if not tokens:
        return []

    chunks: list[Chunk] = []
    stride = size - overlap
    i = 0
    while True:
        window = tokens[i : i + size]
        parts, by_page = [], {}
        for pi, s, e in window:
            lo, hi = by_page.get(pi, (s, e))
            by_page[pi] = (min(lo, s), max(hi, e))
        for pi in sorted(by_page):
            lo, hi = by_page[pi]
            parts.append(pages[pi].text[lo:hi].strip())
        chunks.append(
            Chunk(
                doc=pages[0].doc,
                page_start=pages[window[0][0]].page,
                page_end=pages[window[-1][0]].page,
                chunk_idx=len(chunks),
                text="\n".join(x for x in parts if x),
            )
        )
        if i + size >= len(tokens):
            return chunks
        i += stride


def chunk_pages(pages: Sequence[Page], size: int, overlap: int, offsets: OffsetFn) -> list[Chunk]:
    by_doc: dict[str, list[Page]] = {}
    for p in pages:
        by_doc.setdefault(p.doc, []).append(p)
    out: list[Chunk] = []
    for doc_pages in by_doc.values():
        out.extend(chunk_doc(doc_pages, size, overlap, offsets))
    return out
