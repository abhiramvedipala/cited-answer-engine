# Progress

## Done
- Phase 1: corpus (2,019 pages, checksummed), chunking with page spans, local embeddings, pgvector ingest, tests.
- Phase 2: dense, BM25, hybrid (RRF) and cross-encoder rerank retrievers.
- Phase 3: cited answers, `NOT_IN_CORPUS` refusal, citation post-processing, cost logging.
- Phase 4: 150 + 20 question eval set, runner, judge, seven setups run, report and chart, RESULTS.md.
- Phase 5: streaming FastAPI service, rate limit, request log, Docker image, load test, README.

`scripts/check.sh` passes: lint, 36 tests, eval-set integrity.

## Needs you
1. **Remote repository.** Nothing has been created on GitHub or pushed. The README CI badge points at
   `github.com/abhiramvedipala/cited-answer-engine`. When you want it: `gh repo create
   abhiramvedipala/cited-answer-engine --private --source=. --remote=origin`, then push `main`.
2. **Review a sample of the 68 hand-written questions** in `eval/hand_questions.tsv`.
3. **Update the resume** to the measured numbers in RESULTS.md (recall@5 is 0.887, not a target).
4. **Docker disk.** The Docker VM was nearly full while building (other projects' images).

## Notes
- Built on `main` locally. PROMPT.md was updated mid-build to ask for per-phase branches and PRs;
  I did not rewrite history to create them.
- The first eval run was discarded after a gold-label bug; see DECISIONS.md.

## Next
- Retrieve 8 to 10 chunks and add an exact-ID boost for control names (over-refusals).
- Citation-to-claim entailment check and an identifier-grounding post-check.
- A second judge model and paired significance tests.
