# Changelog

The project was built directly on `main` in seven stages, all on 2026-11-04, and has one merged pull request (#1). Each section lists the commit range (oldest first) and what changed, grouped from the commit subjects.

## Phase 1: corpus and ingestion (2026-11-04)

Commits `d287f01`..`54892ea`.

- Scaffold: `.gitignore` and `.env.example`, `pyproject.toml` (Python 3.12, uv), and a docker-compose file for pgvector Postgres 16 on port 5433.
- Corpus download script with pinned sha256 checksums for the Python 3.12 docs and four public-domain US government PDFs; `library.pdf` and `c-api.pdf` excluded.
- PyMuPDF parser giving one record per page with 1-based page numbers.
- Page-aware token-window chunker that records `page_start` and `page_end`, with edge-case tests.
- Local `bge-small-en-v1.5` embeddings, a pgvector `chunks` table keyed by a `config` column, and an idempotent `ingest.run` CLI.
- Idempotency test against real Postgres using a fake embedder.

## Phase 2: retrieval (2026-11-04)

Commits `d19711a`..`0239f54`.

- Dense (pgvector cosine), BM25 (rank_bm25, in memory) and hybrid (reciprocal rank fusion, k=60) retrievers behind one `search(query, k)` interface.
- Tests for rank fusion (agreement, truncation, empty lists, exact score sum) and for `Hit.covers` on multi-page spans.

## Phase 3: answering (2026-11-04)

Commits `74f255e`..`7d9c114`.

- Anthropic SDK wrapper that streams and logs model, tokens, latency and cost per call, with thinking off.
- Numbered-context prompt requiring `[n]` citations or the exact string `NOT_IN_CORPUS`.
- Post-processing that maps numbers to `doc:page`, strips numbers outside the retrieved set, and flags uncited answers; an empty retrieval refuses without a model call.
- Tests for citation mapping, dropping, refusal detection, prompt format and cost math.

## Phase 4: evaluation (2026-11-04)

Commits `23e391b`..`f8e2365`.

- Fix: NUL bytes are stripped in `parse_pdf` so the first full ingest no longer fails in Postgres.
- Question tooling: heading-based question generator with verified evidence and per-source quotas, a cached page text and phrase locator, 68 hand-written and 20 unanswerable questions, and a build step that fails on a bad gold anchor (final set: 150 answerable + 20 unanswerable).
- Pure metric functions (recall@k, MRR, nearest-rank percentiles) with tests.
- LLM judge for faithfulness and relevance, and an eval runner for the seven setups with per-question rows.
- `DECISIONS.md` records the design choices and rejected alternatives.

## Phase 5: service and Docker (2026-11-04)

Commits `c987320`..`c695ea7`.

- Pytest `testpaths` fix so `scripts/load_test.py` is not collected as a test.
- `POST /ask` streaming SSE endpoint with a per-IP rate limit and a request log, plus `GET /health`; API tests for event order, validation and 429.
- `scripts/check.sh` for lint, tests and eval-set integrity, and an asyncio load test.
- Dockerfile with baked-in models and a compose app service that waits on the database healthcheck.

## Fixes and measurement (2026-11-04)

Commits `47f5929`..`8422b38`.

- Citations carry `page_end` and print page spans such as `doc:38-39`.
- Eval correction: every page holding the evidence counts as gold, duplicates are rejected, and all results were re-run; a new "cites gold" metric measures citation accuracy directly.
- Report tooling: results tables with a 95% interval on recall@5, the recall-vs-latency chart, failure candidates, and the raw results for all seven setups committed.
- Docker image: CPU-only torch on Linux, with torch made a direct dependency so the pin applies.
- Service robustness: mid-stream failures are sent as an SSE `error` event, the load test counts error events as failures, and compose forwards the API key and model from the host shell and passes `RATE_LIMIT` through.
- CI workflow running ruff and pytest on push and pull request; one style fix for a long line.

## Rerank switch and written docs (2026-11-04)

Commits `5ed5a9e`..`8289dfe`.

- `RERANK` switch; compose defaults it to off after the load test showed p95 29.9 s with rerank on.
- `RESULTS.md` with every table and its command, and `DECISIONS.md` updated with the measured trade-offs; a correction to the Python PDF count.
- README with problem, demo, architecture, results, limits and structure.
- `docs/INTERVIEW-NOTES.md` for all five phases, and `PROGRESS.md` with open items and next steps.

## Pull request #1: docs(progress): mark the GitHub push as done (merged 2026-11-11)

Commits `572399b`, merge commit `1f499b4`.

- `PROGRESS.md` now records that `main` is pushed to GitHub and that the first CI run passed.
