# Study guide: cited-answer-engine

This is my map of the repository for interview preparation: what to say in one minute, where each step lives in the code, and which commits and files to open, in order.

## In one minute

A retrieval-augmented answer service over 34 PDFs (2,019 pages). It cites a document and page for every claim and replies `NOT_IN_CORPUS` when the corpus cannot answer. I also measured how often it gets that right, on 150 answerable and 20 unanswerable questions.

Stack: PyMuPDF, `bge-small-en-v1.5` (local embeddings), PostgreSQL + pgvector, rank_bm25, `ms-marco-MiniLM-L-6-v2` (optional reranker), the Anthropic SDK with Claude Haiku 5.5 for answers and the judge, FastAPI + slowapi, Docker Compose.

Three measured results worth quoting, copied from `RESULTS.md`:

1. "BM25 ties hybrid+rerank on recall@5 (0.887 each)."
   Produced by `uv run python -m eval.run --config all` then `uv run python -m eval.report`.
2. "Smaller chunks did best: 256 gave 0.933 and 1024 gave 0.720."
   Produced by the same two commands (`--config abl_size256` and `--config abl_size1024` for the single runs).
3. The Docker load-test row with rerank off, and the same container with rerank on:

   ```
   | Docker (4-CPU VM), default | off | 0 | 10.0 req/s | 3.05 s | 3.55 s | $0.000465 |
   | Docker (4-CPU VM) | on | 0 | 0.78 req/s | 29.4 s | 29.9 s | $0.000468 |
   ```

   Produced by `RATE_LIMIT=100000/minute docker compose up -d app` then `uv run python -m scripts.load_test --users 20 --per-user 5`. Columns are Deployment, Rerank, Errors, Throughput, TTFT p95, Total p95, Cost/request.

## How the system works, in five steps

1. **Parse.** `ingest/parse.py`, `parse_pdf` / `parse_dir`: PyMuPDF gives one `Page(doc, page, text)` per PDF page, 1-based, with NUL bytes removed.
2. **Chunk.** `ingest/chunk.py`, `chunk_doc` / `chunk_pages`: fixed token windows (default 512 with 64 overlap) that may cross a page break inside one document, so each `Chunk` stores `page_start` and `page_end`. Token offsets come from the embedding model's own tokenizer (`hf_offsets` in `ingest/embed.py`).
3. **Embed and store.** `ingest/embed.py` `embed_passages` (bge-small, 384 dims) and `ingest/run.py` `ingest()`: rows go into one `chunks` table (schema in `ingest/db.py`, `init_schema`) keyed by a `config` column. Ids are `config:doc:idx`, existing ids are skipped, so a re-run adds nothing.
4. **Retrieve.** `retrieval/retrievers.py`: `Dense` (pgvector cosine) and `BM25` (rank_bm25, in memory) are fused by `rrf` (reciprocal rank fusion, k=60) inside `Hybrid`. `Reranked` optionally re-scores the top 20 with the cross-encoder. `get_retriever` builds the chosen combination; every retriever exposes `search(query, k)` and returns `Hit`s.
5. **Answer, check, stream.** `answer/answer.py`: `answer_stream` sends the top 5 chunks as numbered passages (`build_prompt`) through `answer/llm.py` `stream`; `postprocess` maps `[n]` to `doc:page`, drops numbers outside the retrieved set, flags answers with no valid citation, and detects the `NOT_IN_CORPUS` sentinel. `service/api.py` `create_app` / `ask` streams this as SSE events `token`, `citations`, `done` (or `error`).

## Read the history in this order

The whole project was built directly on `main`: 53 commits, all dated 2026-11-04 except the last two. Only one pull request has been merged (#1, a docs change), so I group the commits into phases by reading the log. The phases follow `PROGRESS.md`, with the fix commits that came out of running things kept in their own rows. Ranges are first and last short hash in log order (oldest first).

| order | pull request or phase | commit range | what was built | what to notice |
|---|---|---|---|---|
| 1 | Phase 1: corpus and ingestion | `d287f01`..`54892ea` (11 commits) | Pinned-checksum corpus download, PDF parser, page-aware chunker, local embeddings, pgvector schema, idempotent ingest CLI, tests | Every chunk carries a page span; ingest ids are deterministic; chunker tests use a whitespace tokenizer so they need no model |
| 2 | Phase 2: retrieval | `d19711a`..`0239f54` (2 commits) | Dense, BM25 and hybrid retrievers behind one `search(query, k)` interface; RRF tests | RRF fuses ranks so no score calibration; BM25 is the real algorithm, not `ts_rank` |
| 3 | Phase 3: answering | `74f255e`..`7d9c114` (3 commits) | Anthropic wrapper that logs cost per call; numbered-context prompt; citation and refusal post-processing; tests | `NOT_IN_CORPUS` is a fixed string; invalid citation numbers are stripped, not trusted |
| 4 | Phase 4: evaluation | `23e391b`..`f8e2365` (9 commits) | NUL-byte fix, question generator with verified evidence, page cache and phrase locator, 68 hand + 20 unanswerable questions, final 170-question set, metrics, LLM judge, runner, `DECISIONS.md` | Gold labels are checked by phrase search, not memory; judge is a small prompt pair instead of Ragas |
| 5 | Phase 5: service and Docker | `c987320`..`c695ea7` (6 commits) | pytest collection fix, `/ask` SSE endpoint with rate limit, API tests, `scripts/check.sh`, asyncio load test, Dockerfile and compose | App factory takes an injected retriever and stream function, so API tests need no database or key |
| 6 | Fix and measure | `47f5929`..`8422b38` (14 commits) | Page-span citations, corrected gold labels, "cites gold" metric, CPU-only torch, report/chart, CI, SSE error event, load-test and key-forwarding fixes, raw eval results for all seven setups | Most of the real learning: the gold-label bug, the torch CUDA image, and the 200-with-broken-stream bug were all found by running, not by reading |
| 7 | Rerank switch and written docs | `5ed5a9e`..`8289dfe` (6 commits) | `RERANK` switch (off in Docker), `RESULTS.md`, README, `docs/INTERVIEW-NOTES.md`, `PROGRESS.md` | A measured trade-off drove a default: rerank cost 8x throughput in Docker for no recall gain |
| 8 | Pull request #1: "docs(progress): mark the GitHub push as done" | `572399b`, merge `1f499b4` (2026-11-11) | `PROGRESS.md` now records that `main` is pushed and CI passed | The only merged PR; a small docs change (2 insertions, 3 deletions) |

## The 10 commits that matter most

| short hash | subject | why it matters | file and function to open |
|---|---|---|---|
| `58926fe` | feat(ingest): add page-aware token window chunker | Design decision behind every citation: windows cross page breaks and record `page_start`/`page_end`; text is cut from original strings via tokenizer offsets to keep casing | `ingest/chunk.py`, `chunk_doc` |
| `23e391b` | fix(ingest): strip NUL bytes from extracted PDF text | Hardest ingest bug: Postgres rejects `0x00`. Fixed once at the single place text enters, so chunker, BM25 and the database see the same string | `ingest/parse.py`, `parse_pdf` |
| `d19711a` | feat(retrieval): add dense, bm25 and hybrid retrievers behind one interface | The retrieval design: one `search(query, k) -> list[Hit]` interface, RRF with k=60 so scores never need calibrating | `retrieval/retrievers.py`, `rrf` and `Hybrid.search` |
| `eb9a6e8` | feat(answer): answer with inline citations or NOT_IN_CORPUS | The core product behaviour: numbered passages, citation or sentinel refusal, invalid numbers stripped, uncited answers flagged, empty retrieval refuses with no model call | `answer/answer.py`, `postprocess` and `answer_stream` |
| `7d9c114` | test(answer): cover citation mapping, dropping and refusal | The tests that pin the headline claim (cites `doc:page`, drops fabricated numbers, refuses with the sentinel). Note the unit tests prove the mechanism; the 0.887 figure comes from the eval, not from pytest | `tests/test_answer.py`, `test_out_of_range_citation_is_dropped_and_removed_from_text` |
| `47f5929` | fix(answer): cite the page span of multi-page chunks | A correct answer from page 39 was cited as page 38 because only `page_start` was shown. Citations now carry `page_end` and print `doc:38-39` | `answer/answer.py`, `source` and `postprocess` |
| `2d2569f` | fix(eval): count every page that holds the evidence as gold | Hardest eval bug: 22 of 150 questions had evidence repeated on other pages, so recall was undercounted. The first run was discarded and everything re-run | `eval/gen_questions.py`, `_emit`; `eval/build_questions.py`, `main` |
| `575800d` | feat(eval): report how often citations point at a gold page | Measures the citation claim directly (cited chunk span covers a gold page) instead of inferring it from recall and faithfulness | `eval/run.py`, `cites_gold` |
| `8422b38` | docs(eval): commit raw eval results for all seven setups | The measurements: per-setup summary JSON and per-question JSONL, so every README number and failure case traces to a row | `eval/results/hybrid_rerank.jsonl`, `eval/results/*.json` |
| `5ed5a9e` | feat(service): add RERANK switch and default it off in Docker | A measurement turned into a default: p95 29.9 s at 0.78 req/s with rerank on, versus about 3.4 s at 10 req/s off (the commit text says 3.4 s; `RESULTS.md` says 3.55 s) | `service/api.py`, `default_retriever`; `docker-compose.yml` |

## 15-minute path

1. `README.md`: what it does, the results table, known limits. Question: what is the claim and what did I measure?
2. `RESULTS.md`, section "What the numbers say, and what they do not". Question: which differences are real at n=150?
3. `ingest/chunk.py`, `chunk_doc`. Question: how does a chunk know which pages it covers?
4. `retrieval/retrievers.py`, `rrf` and `Hybrid`. Question: how are dense and BM25 combined without calibrating scores?
5. `answer/answer.py`, `postprocess`. Question: how are fabricated citations and refusals handled?
6. `service/api.py`, `ask` / `events`. Question: how do failures reach the client after the 200 header is sent?
7. `DECISIONS.md`. Question: what did I reject, and why?

## 60-minute path

1. `README.md` and `DECISIONS.md`. Question: what is the design and what are the trade-offs?
2. `ingest/parse.py`, `ingest/chunk.py`, `ingest/embed.py`. Question: why count chunk size in the embedder's own tokens?
3. `ingest/run.py` and `ingest/db.py`. Question: why is ingest idempotent, and why no ANN index?
4. `tests/test_chunk.py` and `tests/test_ingest_db.py`. Question: which chunker edge cases are covered, and what skips without Postgres?
5. `retrieval/retrievers.py` and `tests/test_retrieval.py`. Question: what exactly does RRF compute, and what does `Hit.covers` mean for recall?
6. `answer/llm.py` and `answer/answer.py`. Question: how is cost computed per call, and why is thinking off?
7. `tests/test_answer.py`. Question: which behaviours are pinned by tests?
8. `eval/gen_questions.py`, `eval/pages.py`, `eval/build_questions.py`. Question: how do I know the gold labels are right?
9. `eval/metrics.py` and `eval/judge.py`. Question: how are recall@k, MRR, and faithfulness defined, and how is the judge biased?
10. `eval/run.py`. Question: how is latency measured so concurrency does not inflate p95?
11. `RESULTS.md` and `docs/failure_candidates.md`. Question: what do the numbers show, and which rows fail?
12. `service/api.py`, `tests/test_api.py`, `scripts/load_test.py`. Question: how is streaming, rate limiting and the error event tested?
13. `Dockerfile`, `docker-compose.yml`, `pyproject.toml`, `.github/workflows/ci.yml`. Question: why CPU-only torch, why rerank off, and what does CI run?
14. `docs/INTERVIEW-NOTES.md`. Question: what are the prepared questions for each phase?

## Numbers to remember

All from `RESULTS.md`; single run per setup, 150 answerable + 20 unanswerable questions, model `claude-haiku-5-5`. Re-run with `uv run python -m eval.run --config all` then `uv run python -m eval.report`.

| Number | What it is | How it was measured |
|---|---|---|
| 2,019 pages across 34 PDFs | Corpus size | `uv run python scripts/download_corpus.py` then `uv run python -m eval.report` |
| 2,831 chunks (512/64) | Chunks at the default config; 6,578 (256/64), 1,331 (1024/64), 2,484 (512/0) | `uv run python -m ingest.run --size N --overlap M` |
| recall@5 0.780 / 0.887 / 0.887 | dense / bm25 / hybrid+rerank (base), each with a ±0.066 / ±0.051 / ±0.051 interval | Gold page falls inside the page span of one of the top 5 chunks; 95% normal approximation |
| MRR 0.653 / 0.758 / 0.774 | dense / bm25 / hybrid+rerank | Reciprocal rank of the first chunk covering a gold page |
| recall@5 0.933 ±0.040 | chunk 256 | Ablation `abl_size256` |
| recall@5 0.720 ±0.072 | chunk 1024 | Ablation `abl_size1024`; embeds only the first 512 tokens of each chunk |
| recall@5 0.873 ±0.053 | overlap 0, and hybrid with no rerank | Ablations `abl_overlap0` and `abl_norerank` |
| Faithfulness 0.949 / 0.977 / 0.965 | dense / bm25 / hybrid+rerank | Supported claims / total claims, scored by an LLM judge on the 5 passages the answerer saw |
| Refusal accuracy 1.00 on 20 | Every setup except chunk 1024 (0.95) | Share of the 20 unanswerable questions that got exactly `NOT_IN_CORPUS` |
| Over-refusal 0.033 | hybrid+rerank (dense 0.073, bm25 0.053, chunk 1024 0.093) | Share of the 150 answerable questions that got `NOT_IN_CORPUS` |
| Uncited rate 0.0 | All seven runs | Answers left with zero valid citations |
| $0.00047 per query | hybrid+rerank (chunk 256 is $0.00028, chunk 1024 is $0.00085) | Answer call only; judge cost excluded |
| Judge cost $0.477 | Across all 7 runs | Listed separately from $/query |
| p95 1.76 s | hybrid+rerank end to end (retrieval p95 0.225 s) | Retrieval time plus that answer call's streaming time |
| 3.55 s vs 29.9 s | Docker p95 at 20 users, rerank off vs on (native macOS with rerank on: 4.80 s) | `scripts.load_test --users 20 --per-user 5`, 100 requests, 0 errors |
| 10.0 req/s vs 0.78 req/s | Docker throughput, rerank off vs on | Same load test |
| 7,012 calls, $2.32 | Total logged LLM calls and spend over the whole build | Includes one discarded eval run, question generation and load tests |
| 0.827 to 0.887 | hybrid+rerank recall@5 before and after the gold-label fix | From `docs/INTERVIEW-NOTES.md` (Phase 4), not from `RESULTS.md` |

## Weak spots an interviewer may probe

- **recall@5 is not above BM25.** Hybrid+rerank ties BM25 at 0.887 and the gaps are inside the ±0.05 interval. Answer: the corpus is full of exact identifiers like `PYTHONSTARTUP` and `AC-2`, so lexical matching is strong, and I reported the tie instead of tuning until the hybrid won.
- **Over-refusal on near-misses (`h012`).** The right page ranked 7th, outside the top 5, so the model correctly refused with no evidence. Answer: it is a retrieval miss, not model timidity; next step is retrieving 8 to 10 chunks.
- **NIST control questions (`h052`, `h055`).** Control IDs like `AC-2` and `RA-5` split into weak tokens and there are hundreds of near-identical control pages. Answer: an exact-ID boost for control names is the planned fix.
- **Wrong citation (`s014`).** The answer cites five chunks and says pages 12 and 1 "do not mention it", though page 12 does. Answer: the real answer lives in index pages that were never retrieved, and it is a weak meta question; script-generated questions need a filter for those.
- **Unsupported detail (`s070`).** The answer states `SC-7(18)` but the cited page never contains that string. Answer: true but not grounded; next step is a post-check that every identifier in the answer appears in a cited chunk.
- **Citation to claim support is not checked.** The model can cite a real passage that does not support the claim. Answer: faithfulness is judged per claim, but a citation-to-claim entailment check has not been built.
- **Judge shares a model family with the answerer.** Faithfulness is probably optimistic, and one manual check found a judge false negative (`s041`). Answer: I chose a small judge over Ragas for dependencies and cost logging; a second judge model would test the bias.
- **82 of 150 answerable questions are generated by the same model family** that answers and judges them. Answer: each has a verbatim evidence sentence that must appear on the gold page, but it can still flatter results.
- **Hand-written questions need human review.** Answer: I wrote them from corpus passages with anchor-checked gold pages; a sample still needs a spot-check (`PROGRESS.md`).
- **One run per setup.** Sampling noise is not measured. Answer: I only call the 1024 gap clear, the 256 vs 512 gap (+0.047) suggestive, and the rest noise; seeds and paired tests would tighten it.
- **The 1024-token config embeds only the first 512 tokens per chunk.** Answer: that likely contributes to its drop; BM25 and the reranker still see the whole chunk, and I report this next to the result.
- **Latency tails.** `overlap 0` (p95 3.33 s) and `chunk 1024` (2.86 s) come from API tail latency, not config; their retrieval p95 is about 0.2 s like the others.
- **Rerank is off by default in Docker.** Answer: on CPU in the Docker VM it took p95 from 3.55 s to 29.9 s, a lock did not help so I removed it, and the eval shows only +0.036 MRR; `RERANK=1` turns it on.
- **Single shared Postgres connection and in-process BM25.** Answer: fine at about 7k rows, not at millions of chunks; it would need a pool, HNSW and a real search engine.
- **Load-test figures differ between files.** `DECISIONS.md` and commit `5ed5a9e` say about 3.4 s for rerank off; `README.md` and `RESULTS.md` say 3.55 s. Answer: I quote `RESULTS.md`.

The full prepared questions and answers, phase by phase, are in `docs/INTERVIEW-NOTES.md`.

## Run it yourself

From the README quick start (prerequisites: Docker, uv, Python 3.12, and an `ANTHROPIC_API_KEY` in your shell):

```bash
cp .env.example .env
docker compose up -d db
uv sync && uv run python scripts/download_corpus.py
uv run python -m ingest.run --size 512 --overlap 64      # about 1.5 minutes
ANTHROPIC_API_KEY=... docker compose up -d app            # serves on :8000, rerank off by default
curl -N -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What does PYTHONSTARTUP do?"}'
```

Run the checks with `scripts/check.sh` (add `--eval` for a paid 13-question smoke run).
