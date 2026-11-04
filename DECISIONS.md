# Design decisions

Each entry: what I chose, what I rejected, and why.

## Corpus: Python 3.12 docs (minus library and C API) + NASA SE Handbook + three NIST SPs
- 2,019 pages across 34 PDFs. `library.pdf` (2,393 pages) alone would have exceeded the whole
  target, and a corpus that is 60% one API reference makes retrieval too easy to game.
- Mixing prose-heavy government PDFs with code-heavy docs makes BM25 and dense retrieval fail in
  different ways, which is what makes the three-setup comparison informative.
- Every file is pinned by sha256 in `scripts/download_corpus.py`; a changed upstream file aborts.

## Embeddings: BAAI/bge-small-en-v1.5, local
- Zero API cost for ingest. 512-token input limit, so 512-token chunks embed whole.
  all-MiniLM-L6-v2 was rejected because its 256-token limit would silently truncate chunks.
- Known effect: the 1024-token ablation embeds only the first 512 tokens of each chunk.
  BM25 and the reranker still see more text. I report this next to the result.

## Storage: one `chunks` table, a `config` column per chunking setup
- All four chunking configs live side by side so ablations are a parameter, not a re-deploy.
- No ANN index: an exact cosine scan over about 7k rows takes milliseconds, and HNSW with a
  filter on `config` adds recall caveats I would then have to measure too.

## BM25: rank_bm25 in memory, built from Postgres rows
- Postgres full-text search was the alternative. It uses ts_rank, not BM25, so the baseline would
  not be the algorithm people mean by "BM25".

## Fusion: reciprocal rank fusion, k=60
- Rank-based, so cosine scores and BM25 scores never need calibrating.

## Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 over the top 20 candidates
- Candidate depth 20 balances recall gain against rerank latency on CPU.

## Answering: Claude Haiku 5.5, thinking off, max 600 output tokens
- Cheapest model that follows citation rules. Thinking is off because the task is extractive
  and short; it would add latency and output cost.
- `NOT_IN_CORPUS` is a sentinel string, not a free-text refusal, so refusal accuracy is exact.
- Citations the model invents (numbers outside the retrieved set) are removed, and an answer left
  with zero valid citations is flagged `uncited`.

## Evaluation: small LLM judge instead of Ragas
- Ragas pulls a large dependency tree and defaults to an OpenAI judge. Two short prompts on the
  same SDK keep cost logging uniform. The cost is that the judge shares a model family with the
  answerer, so faithfulness is probably optimistic. Stated in the README limits.
- Faithfulness = supported claims / total claims, judged on the exact 5 passages the answerer saw.
- Recall@5 counts a hit when any of the top 5 chunks has a page span covering a gold page.

## Eval set: 82 script + 68 hand-written answerable, 20 unanswerable
- Script questions come from detected headings, with a verbatim evidence sentence that must
  appear on the gold page. Sampling is stratified by source (a first unstratified run was 93%
  Python).
- Hand questions are anchored to a phrase that must match exactly one page (two if an appendix
  repeats the text), so a wrong gold label fails the build.
- Several unanswerable questions are near misses: the corpus mentions the term but cannot answer.

## Latency measurement
- Retrieval runs serially and is timed alone. Answer calls run in a thread pool, each timed on
  its own. End-to-end latency = retrieval + that call. Judge calls are excluded from latency and
  per-query cost; their total is reported separately.

## Gold labels: every page that holds the evidence counts
- The first full eval run undercounted recall: 22 of 150 questions had evidence repeated on other
  pages (the Python license and glossary appear in every PDF, one header string on 492 pages) and
  two questions were identical. I discarded that run, required script-question evidence to appear
  on at most 2 corpus pages (all of them gold), rejected duplicate text, and re-ran everything.
- Hand questions take gold from a corpus-wide anchor search for the same reason.

## Citations show the chunk's page span
- A chunk can cross a page break, so citing only its first page pointed readers at the wrong page.
  Citations carry `page_end` and print as `doc:38-39`. "Cites gold" counts a hit when a cited
  chunk's span covers a gold page.

## Service: rerank is off by default in Docker, on by default in code
- At 20 concurrent users in the Docker VM (4 CPUs, CPU torch) the cross-encoder gave p95 29.9s at
  0.78 req/s; without it p95 was 3.4s at 10 req/s. Natively on macOS it was p95 4.8s. A lock around
  inference did not help (27.9s without, 29.9s with), so I removed it.
- The eval shows rerank adds 0.036 MRR and no measurable recall@5 gain on this corpus, so the
  trade is poor for a CPU deployment. `RERANK=1` turns it on.
- Mid-stream failures are sent as an SSE `error` event, because the 200 header is already out.

## Docker image: CPU-only torch on Linux
- PyPI's Linux torch pulls CUDA and 3.3 GB of nvidia libraries (10.8 GB image). Pointing Linux at
  the PyTorch CPU index gives 2.9 GB. torch must be a direct dependency for `[tool.uv.sources]`
  to apply; my first attempt did nothing because it was only transitive.

## Repository
- The code is committed locally on `main`. No remote has been created or pushed to yet; see
  PROGRESS.md.
