"""Pure metric functions, kept free of I/O so they are unit-tested."""

import math

from retrieval import Hit


def gold_rank(hits: list[Hit], gold: list[dict]) -> int | None:
    """1-based rank of the first hit whose page span covers any gold doc:page."""
    for i, h in enumerate(hits, 1):
        if any(h.covers(g["doc"], g["page"]) for g in gold):
            return i
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks)


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0-100)."""
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")
