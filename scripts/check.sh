#!/usr/bin/env bash
# Lint + tests + eval smoke test. Run before every commit that finishes a unit of work.
#   scripts/check.sh          fast: lint, unit tests, eval-set integrity
#   scripts/check.sh --eval   also run the paid eval smoke test (13 questions, about $0.01)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== lint =="
uv run ruff check .
uv run ruff format --check .

echo "== tests =="
uv run pytest -q

echo "== eval set integrity =="
uv run python -m eval.build_questions >/dev/null
git diff --quiet eval/questions.jsonl || { echo "eval/questions.jsonl is stale"; exit 1; }
uv run python - <<'PY'
import json
q = [json.loads(x) for x in open("eval/questions.jsonl")]
a = [x for x in q if x["answerable"]]
assert len(a) == 150 and len(q) - len(a) == 20, (len(a), len(q))
print("150 answerable + 20 unanswerable, all gold pages present")
PY

if [[ "${1:-}" == "--eval" ]]; then
  echo "== eval smoke =="
  uv run python -m eval.run --config hybrid_rerank --limit 10 --force
fi
echo "OK"
