import re

import pytest

from ingest.chunk import chunk_doc, chunk_id, chunk_pages, config_name
from ingest.parse import Page


def words(text):  # whitespace tokenizer stands in for the HF tokenizer
    return [m.span() for m in re.finditer(r"\S+", text)]


def doc(*texts, name="d"):
    return [Page(name, i + 1, t) for i, t in enumerate(texts)]


def test_empty_doc_gives_no_chunks():
    assert chunk_doc(doc("", "   "), 4, 1, words) == []


def test_text_shorter_than_chunk_is_one_chunk():
    [c] = chunk_doc(doc("a b c"), 10, 2, words)
    assert c.text == "a b c" and (c.page_start, c.page_end) == (1, 1)


def test_exact_multiple_has_no_tiny_tail():
    chunks = chunk_doc(doc("a b c d e f"), 3, 0, words)
    assert [c.text for c in chunks] == ["a b c", "d e f"]


def test_overlap_repeats_tokens():
    chunks = chunk_doc(doc("a b c d e"), 3, 1, words)
    assert [c.text for c in chunks] == ["a b c", "c d e"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (4, 4), (4, 5), (4, -1)])
def test_bad_params_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_doc(doc("a b"), size, overlap, words)


def test_chunk_spanning_pages_keeps_both_page_numbers():
    chunks = chunk_doc(doc("a b", "c d", "e f"), 3, 0, words)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 2), (2, 3)]
    assert chunks[0].text == "a b\nc"


def test_blank_page_is_skipped_but_numbering_preserved():
    [c] = chunk_doc(doc("a", "", "b"), 5, 0, words)
    assert (c.page_start, c.page_end) == (1, 3)


def test_chunks_never_cross_documents():
    pages = doc("a b", name="x") + doc("c d", name="y")
    chunks = chunk_pages(pages, 10, 0, words)
    assert [(c.doc, c.text) for c in chunks] == [("x", "a b"), ("y", "c d")]


def test_ids_are_deterministic():
    [c] = chunk_doc(doc("a"), 4, 0, words)
    assert chunk_id(config_name(4, 0), c) == "s4_o0:d:0"
