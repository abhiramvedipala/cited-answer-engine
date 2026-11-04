# Interview notes

One section per phase: what it built, the decisions and the "why not X", the hardest problem, five likely questions, and the files to open.

## Phase 1: Corpus and ingestion

**Built**
- A checksummed download script for 34 PDFs (2,019 pages): Python 3.12 docs plus NASA SE Handbook and three NIST publications.
- Page-aware token chunker: windows of N tokens with overlap that remember `page_start` and `page_end`.
- Local embeddings (bge-small) into one pgvector table keyed by chunking config.
- Idempotent ingest CLI; tests for chunker edge cases, page mapping and re-ingest.

**Decisions**
- One table with a `config` column so the chunk-size ablation is a parameter. *Why not a table per config?* More schema, no benefit.
- No ANN index. *Why not HNSW?* An exact scan over about 7k rows takes milliseconds, and an index with a config filter adds recall caveats I would have to measure.
- bge-small over MiniLM: 512-token input matches 512-token chunks.

**Hardest problem.** The first full ingest crashed with `PostgreSQL text fields cannot contain NUL (0x00) bytes`; some PDF text layers emit them. I fixed it at the single place text enters, `parse_pdf`. A second trap: ingest is not atomic, so a crash leaves partial rows; deterministic ids and skip-existing make a re-run safe.

**Questions**
1. *How do you keep citations accurate when a chunk crosses a page?* Each chunk stores its page span. Citations print the span and recall counts a hit if the gold page is anywhere in it.
2. *Why is ingest idempotent and how?* Chunk ids are `config:doc:idx`; existing ids are skipped before embedding, so a rerun embeds nothing and row counts stay the same.
3. *Why count chunk size in model tokens, not characters?* The embedder truncates at 512 tokens; sizing with its tokenizer means a "512" chunk really is 512 tokens it sees.
4. *What goes wrong with the 1024-token config?* The embedding model reads only the first 512 tokens, so the second half of each chunk is invisible to dense retrieval. BM25 and the reranker still see it.
5. *How would this scale to millions of chunks?* Add an HNSW index, move BM25 to Postgres full-text or a search engine, and use a connection pool.

**Open:** `ingest/chunk.py`, `ingest/run.py`, `tests/test_chunk.py`, `scripts/download_corpus.py`.

## Phase 2: Retrieval

**Built**: dense (pgvector cosine), BM25 (rank_bm25), hybrid (reciprocal rank fusion), and an optional cross-encoder reranker over the top 20, all behind `search(query, k) -> list[Hit]`. Every hit carries doc, page span and chunk id.

**Decisions**
- RRF with k=60 because it fuses ranks, so cosine and BM25 scores need no calibration. *Why not a weighted score sum?* It needs a tuned weight and normalisation per query.
- rank_bm25 over Postgres `ts_rank`, because `ts_rank` is not BM25 and the baseline should be the algorithm people mean.

**Hardest problem.** Interpreting the result, not building it: BM25 tied hybrid+rerank on recall@5 (0.887 each). On a corpus full of exact identifiers lexical search is strong, and I report that instead of tuning until the hybrid won.

**Questions**
1. *Why did hybrid not beat BM25?* The questions and documents share exact terms (function names, control IDs); dense adds recall for paraphrases but the corpus has few. The gap is also inside the ±0.05 interval.
2. *What does RRF do exactly?* Score = sum over lists of 1/(60 + rank). A chunk ranked well in both lists beats one ranked first in only one.
3. *What does the reranker cost?* About 0.2 s p95 natively, but it dominated service latency in Docker: p95 29.9 s versus 3.55 s at 20 users.
4. *How is a retrieval hit defined?* A chunk whose page span covers any gold doc:page.
5. *Why rerank only 20 candidates?* Cross-encoder cost grows linearly with candidates; 20 balances recall gain and CPU time.

**Open:** `retrieval/retrievers.py`, `tests/test_retrieval.py`.

## Phase 3: Answering with citations and refusal

**Built**: numbered-context prompt that requires `[n]` citations or exactly `NOT_IN_CORPUS`; post-processing that maps numbers to `doc:page`, drops numbers outside the retrieved set, and flags answers with no valid citation; an SDK wrapper that logs tokens, latency and cost per call.

**Decisions**
- Refusal is a sentinel string, so accuracy is exact. *Why not free-text refusals?* They need a judge and add noise.
- Thinking is off: short extractive answers do not benefit and it adds cost and latency.
- Empty retrieval refuses without calling the model.

**Hardest problem.** Citation page spans. Early output cited only a chunk's first page, so a correct answer from page 39 showed page 38. Adding `page_end` and printing spans fixed it, and a new "cites gold" metric measures it.

**Questions**
1. *How do you stop fabricated citations?* Only numbers 1..k are valid; others are removed from the text and recorded in `dropped`.
2. *Can the model cite a real passage that does not support the claim?* Yes, and I do see it (`s014`). Faithfulness is judged per claim, but a citation-to-claim entailment check is a next step.
3. *Why Haiku?* Cheapest model that follows the citation rules; about $0.0005 per query.
4. *How is cost computed?* Per-call tokens times the per-million price, written to `logs/llm_calls.jsonl`.
5. *What if the model refuses wrongly?* That is over-refusal (3.3% of answerable questions for hybrid+rerank); it is mostly a retrieval miss, not model timidity.

**Open:** `answer/answer.py`, `answer/llm.py`, `tests/test_answer.py`.

## Phase 4: Evaluation

**Built**: 150 answerable + 20 unanswerable questions (82 generated from headings with verified evidence sentences, 68 hand-written with anchor-checked gold pages); runner reporting recall@5/10, MRR, faithfulness, relevance, refusal accuracy, over-refusal, cost, p50/p95; four ablations; report with intervals and a chart.

**Decisions**
- Small LLM judge instead of Ragas (dependencies, default OpenAI judge). Cost: same-family judge.
- Gold is doc:page, checked by search, not by memory.
- Retrieval timed serially, answer calls in threads each timed alone, so concurrency does not inflate p95.

**Hardest problem.** My first full run gave plausible numbers that were wrong. Failure review showed 22 of 150 questions had evidence repeated on other pages (Python license and glossary exist in every PDF; one header string on 492 pages) and two were identical. Recall was undercounted. I discarded the run, required evidence on at most 2 pages with all of them as gold, deduplicated, and re-ran. Recall@5 for hybrid+rerank moved from 0.827 to 0.887.

**Questions**
1. *How do you know your eval is trustworthy?* Gold labels are verified by phrase search, unanswerable terms are checked against the corpus, and I read failure rows. That last step is what caught the duplicate-content bug.
2. *Is a 0.054 gap between setups real?* Not at n=150: the 95% interval is about ±0.05. I call 1024 vs the rest clear, 256 vs 512 suggestive, and the rest noise.
3. *What are the eval's biases?* Same model family generates, answers and judges; hand questions written by one person; single run; page-level, not passage-level, matching.
4. *Why report over-refusal?* Perfect refusal accuracy is easy to get by refusing everything. Over-refusal on answerable questions keeps it honest.
5. *What would you add?* Paired bootstrap tests, a second judge model, a human-reviewed sample, and repeated runs.

**Open:** `eval/run.py`, `eval/metrics.py`, `eval/judge.py`, `eval/build_questions.py`, `RESULTS.md`.

## Phase 5: Service

**Built**: FastAPI `POST /ask` streaming SSE (tokens, then citations, then cost and latency), `GET /health`, slowapi per-IP limit, request log with cost, Dockerfile and compose, an asyncio load test.

**Decisions**
- App factory with injected retriever and stream function, so API tests need no database or key.
- Failures after the 200 header are sent as an `error` event.
- Rerank off by default in Docker after measuring it.

**Hardest problem.** The Docker work. The first image build failed (disk) because Linux torch pulled CUDA and 3.3 GB of nvidia libraries (10.8 GB). My first fix did nothing because uv only applies `[tool.uv.sources]` to direct dependencies. Then the container started with no API key; the load test showed 100 errors while curl showed "200", which is why the stream now reports errors. Finally the load test showed rerank cost 8x in throughput, and a lock did not help, so I removed it and made rerank a switch.

**Questions**
1. *Why SSE and not WebSockets?* One-way token stream over plain HTTP, simple for clients and proxies.
2. *How does rate limiting work and how did you test it?* slowapi keys on client IP; a unit test sends 3 requests against a 2/minute limit and expects 200, 200, 429, and I saw the same on the container.
3. *Why was rerank so slow in Docker?* CPU torch in a 4-CPU VM, about 1.3 s per rerank, serialised by contention; throughput fell from 10 to 0.78 req/s.
4. *How would you make it faster with rerank on?* Batch rerank across concurrent requests, fewer candidates, ONNX or quantised model, or a GPU.
5. *How do you keep the API key out of the image?* Compose forwards it from the host shell at run time; it is not in `.env`, the image or git.

**Open:** `service/api.py`, `scripts/load_test.py`, `Dockerfile`, `docker-compose.yml`, `tests/test_api.py`.
