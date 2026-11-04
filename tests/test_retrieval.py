from retrieval import Hit, rrf


def h(cid, doc="d", p=1):
    return Hit(cid, doc, p, p, "t", 0.0)


def test_rrf_rewards_agreement():
    fused = rrf([[h("a"), h("b"), h("c")], [h("c"), h("a"), h("d")]], k=4)
    assert [x.chunk_id for x in fused] == ["a", "c", "b", "d"]


def test_rrf_truncates_and_handles_empty_lists():
    assert [x.chunk_id for x in rrf([[h("a"), h("b")], []], k=1)] == ["a"]
    assert rrf([[], []], k=5) == []


def test_rrf_score_is_reciprocal_rank_sum():
    [top] = rrf([[h("a")], [h("a")]], k=1, rrf_k=60)
    assert abs(top.score - 2 / 61) < 1e-12


def test_hit_covers_page_span():
    hit = Hit("x", "doc", 3, 4, "t", 0)
    assert hit.covers("doc", 3) and hit.covers("doc", 4)
    assert not hit.covers("doc", 5) and not hit.covers("other", 3)
