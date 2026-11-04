from answer.answer import REFUSAL, build_prompt, postprocess
from answer.llm import cost
from retrieval import Hit

HITS = [Hit("c:a:0", "a", 3, 3, "alpha", 1), Hit("c:b:5", "b", 7, 8, "beta", 0.5)]


def test_citations_map_to_doc_page():
    a = postprocess("Alpha is first [1]. Beta spans pages [2][1].", HITS)
    assert [c["source"] for c in a.citations] == ["a:3", "b:7"]
    assert not a.uncited and not a.refused and a.dropped == []


def test_out_of_range_citation_is_dropped_and_removed_from_text():
    a = postprocess("Claim [1]. Made up [4].", HITS)
    assert a.dropped == [4] and "[4]" not in a.text
    assert [c["n"] for c in a.citations] == [1]


def test_zero_citations_is_flagged():
    a = postprocess("An answer with no sources.", HITS)
    assert a.uncited and a.citations == []


def test_only_invalid_citations_is_flagged_uncited():
    assert postprocess("Bad [0] and [3].", HITS).uncited


def test_refusal_detected():
    a = postprocess(f"  {REFUSAL}\n", HITS)
    assert a.refused and a.text == REFUSAL and not a.uncited


def test_prompt_numbers_passages_with_source():
    p = build_prompt("q?", HITS)
    assert "[1] (a, page 3)\nalpha" in p and "[2] (b, page 7)" in p and p.endswith("Question: q?")


def test_cost_uses_per_million_prices():
    assert abs(cost("claude-haiku-5-5", 1_000_000, 1_000_000) - 0.60) < 1e-9
    assert cost("unknown-model", 10, 10) == 0.0
