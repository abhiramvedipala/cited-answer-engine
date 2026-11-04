"""Ingest data/corpus/*.pdf into Postgres for one chunking config.

Run: uv run python -m ingest.run --size 512 --overlap 64
Idempotent: chunk ids are deterministic (config:doc:idx) and existing ids are skipped,
so a re-run embeds nothing new and leaves row counts unchanged.
"""

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from ingest.chunk import Chunk, chunk_id, chunk_pages, config_name
from ingest.db import connect, init_schema
from ingest.parse import parse_dir


def ingest(conn, chunks: list[Chunk], config: str, embed) -> int:
    existing = {r[0] for r in conn.execute("SELECT id FROM chunks WHERE config = %s", (config,))}
    new = [c for c in chunks if chunk_id(config, c) not in existing]
    if not new:
        return 0
    vectors = embed([c.text for c in new])
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, config, doc, page_start, page_end, chunk_idx, text, embedding)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
            [
                (
                    chunk_id(config, c),
                    config,
                    c.doc,
                    c.page_start,
                    c.page_end,
                    c.chunk_idx,
                    c.text,
                    v,
                )
                for c, v in zip(new, vectors, strict=True)
            ],
        )
    return len(new)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    args = ap.parse_args()

    from ingest.embed import embed_passages, hf_offsets

    t0 = time.perf_counter()
    pages = parse_dir(args.corpus)
    config = config_name(args.size, args.overlap)
    chunks = chunk_pages(pages, args.size, args.overlap, hf_offsets)
    conn = connect()
    init_schema(conn)
    added = ingest(conn, chunks, config, embed_passages)
    total = conn.execute("SELECT count(*) FROM chunks WHERE config=%s", (config,)).fetchone()[0]
    print(
        f"{config}: {len(pages)} pages, {len(chunks)} chunks, {added} new, {total} in db, "
        f"{time.perf_counter() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
