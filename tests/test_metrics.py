import pytest

from eval.metrics import gold_rank, mean, mrr, percentile, recall_at
from retrieval import Hit

GOLD = [{"doc": "d", "page": 5}]


def hit(doc, a, b=None):
    return Hit(f"{doc}{a}", doc, a, b or a, "t", 0)


def test_gold_rank_matches_page_inside_span():
    hits = [hit("x", 1), hit("d", 4, 6), hit("d", 9)]
    assert gold_rank(hits, GOLD) == 2


def test_gold_rank_none_when_missing_or_wrong_doc():
    assert gold_rank([hit("x", 5), hit("d", 6)], GOLD) is None


def test_gold_rank_accepts_either_of_two_gold_pages():
    assert gold_rank([hit("d", 7)], GOLD + [{"doc": "d", "page": 7}]) == 1


def test_recall_and_mrr():
    ranks = [1, 3, None, 8]
    assert recall_at(ranks, 5) == 0.5
    assert recall_at(ranks, 10) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 1 / 8) / 4)


def test_percentile_nearest_rank():
    xs = list(range(1, 101))
    assert percentile(xs, 50) == 50 and percentile(xs, 95) == 95
    assert percentile([3.0], 95) == 3.0


def test_mean_empty_is_nan():
    assert mean([]) != mean([])
