"""Evaluate retrieval setups end to end on eval/questions.jsonl.

  uv run python -m eval.run --config main        # dense, bm25, hybrid+rerank at 512/64
  uv run python -m eval.run --config ablations   # chunk size, overlap, rerank on/off
  uv run python -m eval.run --config all
  uv run python -m eval.run --config dense       # a single setup by name

Per setup it records recall@5, recall@10, MRR, faithfulness, answer relevance, refusal accuracy,
over-refusal, answer cost per query and p50/p95 end-to-end latency. Per-question rows go to
eval/results/<name>.jsonl (used for the failure cases in the README).

Latency note: retrieval runs one query at a time and is timed alone. Answer calls run in a
thread pool for speed, but each call's latency is its own streaming wall time, so end-to-end
latency = retrieval time + that call's time. Judge calls are excluded from latency and cost.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from answer.answer import answer
from eval import judge
from eval.metrics import gold_rank, mean, mrr, percentile, recall_at
from ingest.db import connect
from retrieval import get_retriever

RESULTS = Path("eval/results")
K_ANSWER, K_RETRIEVE = 5, 10


@dataclass(frozen=True)
class Setup:
    name: str
    retriever: str
    config: str  # chunking config id, e.g. s512_o64
    rerank: bool


SETUPS = {
    s.name: s
    for s in [
        Setup("dense", "dense", "s512_o64", False),
        Setup("bm25", "bm25", "s512_o64", False),
        Setup("hybrid_rerank", "hybrid", "s512_o64", True),
        # ablations: each changes one thing from hybrid_rerank
        Setup("abl_size256", "hybrid", "s256_o64", True),
        Setup("abl_size1024", "hybrid", "s1024_o64", True),
        Setup("abl_overlap0", "hybrid", "s512_o0", True),
        Setup("abl_norerank", "hybrid", "s512_o64", False),
    ]
}
GROUPS = {
    "main": ["dense", "bm25", "hybrid_rerank"],
    "ablations": ["abl_size256", "abl_size1024", "abl_overlap0", "abl_norerank"],
}
GROUPS["all"] = GROUPS["main"] + GROUPS["ablations"]


def load_questions() -> list[dict]:
    return [json.loads(x) for x in Path("eval/questions.jsonl").read_text().splitlines() if x]


def run_setup(setup: Setup, questions: list[dict], workers: int = 8) -> tuple[dict, list[dict]]:
    conn = connect()
    retriever = get_retriever(setup.retriever, conn, setup.config, setup.rerank)
    retriever.search("warm up", 1)  # load models and BM25 index outside the timed loop

    retrieved = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], K_RETRIEVE)
        retrieved.append((hits, time.perf_counter() - t0))

    def work(item):
        q, (hits, ret_s) = item
        a = answer(q["question"], hits[:K_ANSWER])
        row = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "gold": q["gold"],
            "answer": a.text,
            "refused": a.refused,
            "uncited": a.uncited,
            "dropped_citations": a.dropped,
            "citations": a.citations,
            "retrieval_s": ret_s,
            "answer_s": a.meta.get("latency_s", 0.0),
            "answer_cost": a.meta.get("cost_usd", 0.0),
            "judge_cost": 0.0,
            "top": [
                {"chunk": h.chunk_id, "doc": h.doc, "p": h.page_start, "pe": h.page_end}
                for h in hits[:K_RETRIEVE]
            ],
            "rank": gold_rank(hits, q["gold"]) if q["answerable"] else None,
            "faith": None,
            "rel": None,
        }
        if not a.refused:
            row["faith"], c1 = judge.faithfulness(a.text, hits[:K_ANSWER])
            if q["answerable"]:
                row["rel"], c2 = judge.relevance(q["question"], a.text)
                row["judge_cost"] = c1 + c2
            else:
                row["judge_cost"] = c1
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(work, zip(questions, retrieved, strict=True)))
    return summarize(setup, rows), rows


def summarize(setup: Setup, rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    ranks = [r["rank"] for r in ans]
    answered = [r for r in ans if not r["refused"]]
    lat = [r["retrieval_s"] + r["answer_s"] for r in rows]
    return {
        "setup": setup.name,
        "retriever": setup.retriever,
        "chunking": setup.config,
        "rerank": setup.rerank,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "recall@5": recall_at(ranks, 5),
        "recall@10": recall_at(ranks, 10),
        "mrr": mrr(ranks),
        "faithfulness": mean([r["faith"] for r in rows if r["faith"] is not None]),
        "answer_relevance": mean([r["rel"] for r in ans if r["rel"] is not None]),
        "refusal_accuracy": mean([float(r["refused"]) for r in unans]),
        "over_refusal": mean([float(r["refused"]) for r in ans]),
        "uncited_rate": mean([float(r["uncited"]) for r in answered]),
        "cost_per_query_usd": mean([r["answer_cost"] for r in rows]),
        "judge_cost_total_usd": sum(r["judge_cost"] for r in rows),
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "retrieval_p95_s": percentile([r["retrieval_s"] for r in rows], 95),
    }


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main", help="main | ablations | all | <setup name>")
    ap.add_argument("--limit", type=int, help="first N answerable + 3 unanswerable (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    args = ap.parse_args()

    names = GROUPS.get(args.config) or [args.config]
    questions = load_questions()
    tag = ""
    if args.limit:
        questions = [q for q in questions if q["answerable"]][: args.limit] + [
            q for q in questions if not q["answerable"]
        ][:3]
        tag = "_smoke"
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in names:
        out = RESULTS / f"{name}{tag}.json"
        if out.exists() and not args.force:
            print(f"skip {name} (exists; use --force)")
            continue
        t0 = time.perf_counter()
        summary, rows = run_setup(SETUPS[name], questions)
        out.write_text(json.dumps(summary, indent=2))
        out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(
            f"{name}: "
            + " ".join(f"{k}={v:.3f}" for k, v in summary.items() if isinstance(v, float))
            + f" ({time.perf_counter() - t0:.0f}s)"
        )


if __name__ == "__main__":
    main()
