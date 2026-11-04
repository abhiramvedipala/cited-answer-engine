"""Three retrievers behind one interface: search(query, k) -> list[Hit].

dense  : pgvector cosine distance over bge-small embeddings
bm25   : rank_bm25 Okapi over lowercase word tokens, built in memory per config
hybrid : reciprocal rank fusion of dense and bm25
Any of them can be wrapped in a cross-encoder reranker.
"""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RRF_K = 60  # standard constant from Cormack et al. 2009


@dataclass
class Hit:
    chunk_id: str
    doc: str
    page_start: int
    page_end: int
    text: str
    score: float

    def covers(self, doc: str, page: int) -> bool:
        return self.doc == doc and self.page_start <= page <= self.page_end


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


COLS = "id, doc, page_start, page_end, text"


class Dense:
    def __init__(self, conn, config: str):
        self.conn, self.config = conn, config

    def search(self, query: str, k: int) -> list[Hit]:
        from ingest.embed import embed_query

        rows = self.conn.execute(
            f"SELECT {COLS}, 1 - (embedding <=> %s) FROM chunks WHERE config = %s"
            " ORDER BY embedding <=> %s LIMIT %s",
            (q := embed_query(query), self.config, q, k),
        ).fetchall()
        return [Hit(*r) for r in rows]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=8)
def _bm25_index(conn, config: str):
    from rank_bm25 import BM25Okapi

    rows = conn.execute(
        f"SELECT {COLS} FROM chunks WHERE config = %s ORDER BY id", (config,)
    ).fetchall()
    if not rows:
        raise ValueError(f"no chunks for config {config}; run ingest first")
    return BM25Okapi([tokenize(r[4]) for r in rows]), rows


class BM25:
    def __init__(self, conn, config: str):
        self.index, self.rows = _bm25_index(conn, config)

    def search(self, query: str, k: int) -> list[Hit]:
        import numpy as np

        scores = self.index.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [Hit(*self.rows[i], float(scores[i])) for i in top if scores[i] > 0]


def rrf(ranked_lists: list[list[Hit]], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    """Fuse by sum of 1/(rrf_k + rank). Rank-based, so dense cosine and BM25 scores
    never need to be put on the same scale."""
    fused: dict[str, float] = {}
    first: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(h.chunk_id, h)
    order = sorted(fused, key=lambda cid: -fused[cid])[:k]
    return [Hit(**{**first[c].__dict__, "score": fused[c]}) for c in order]


class Hybrid:
    def __init__(self, conn, config: str, depth: int = 50):
        self.dense, self.bm25, self.depth = Dense(conn, config), BM25(conn, config), depth

    def search(self, query: str, k: int) -> list[Hit]:
        return rrf([self.dense.search(query, self.depth), self.bm25.search(query, self.depth)], k)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, max_length=512)


class Reranked:
    def __init__(self, base: Retriever, candidates: int = 20):
        self.base, self.candidates = base, candidates

    def search(self, query: str, k: int) -> list[Hit]:
        hits = self.base.search(query, self.candidates)
        if not hits:
            return []
        scores = _cross_encoder().predict([(query, h.text) for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.score = float(s)
        return sorted(hits, key=lambda h: -h.score)[:k]


def get_retriever(name: str, conn, config: str, rerank: bool = False) -> Retriever:
    base = {"dense": Dense, "bm25": BM25, "hybrid": Hybrid}[name](conn, config)
    return Reranked(base) if rerank else base
