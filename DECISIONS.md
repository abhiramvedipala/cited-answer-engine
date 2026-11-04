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

## Repository workflow
- PROMPT.md was updated mid-build to ask for phase branches, PRs and a GitHub repo created by
  the assistant. I kept building on `main` locally and did not create or push a remote, because
  that is outward-facing and was not asked for in chat. See PROGRESS.md.
