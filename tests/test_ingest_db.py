"""Integration: needs the docker-compose Postgres. Uses a fake embedder, so no model download."""

import re

import numpy as np
import pytest

from ingest.chunk import chunk_pages
from ingest.parse import Page

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.integration


@pytest.fixture
def conn():
    from ingest.db import connect, init_schema

    try:
        c = connect()
    except psycopg.OperationalError:
        pytest.skip("Postgres not running")
    init_schema(c)
    c.execute("DELETE FROM chunks WHERE config = 'test'")
    yield c
    c.execute("DELETE FROM chunks WHERE config = 'test'")


def fake_embed(texts):
    return np.ones((len(texts), 384), dtype=np.float32) / np.sqrt(384)


def test_ingest_twice_is_idempotent_and_keeps_pages(conn):
    from ingest.run import ingest

    def words(t):
        return [m.span() for m in re.finditer(r"\S+", t)]

    pages = [Page("doc", 1, "alpha beta gamma"), Page("doc", 2, "delta epsilon")]
    chunks = chunk_pages(pages, 3, 1, words)
    assert ingest(conn, chunks, "test", fake_embed) == len(chunks)
    assert ingest(conn, chunks, "test", fake_embed) == 0
    rows = conn.execute(
        "SELECT page_start, page_end, text FROM chunks WHERE config='test' ORDER BY chunk_idx"
    ).fetchall()
    assert rows == [(1, 1, "alpha beta gamma"), (1, 2, "gamma\ndelta epsilon")]
