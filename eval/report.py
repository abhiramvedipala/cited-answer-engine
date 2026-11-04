"""Build RESULTS.md numbers, the ablation chart and failure-case candidates from eval/results.

uv run python -m eval.report            # writes docs/recall_vs_latency.png, docs/results_tables.md,
                                        # docs/failure_candidates.md
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.pages import load as load_pages
from eval.run import GROUPS, SETUPS, cites_gold

RES = Path("eval/results")
DOCS = Path("docs")
LABEL = {
    "dense": "dense",
    "bm25": "bm25",
    "hybrid_rerank": "hybrid+rerank (base)",
    "abl_size256": "chunk 256",
    "abl_size1024": "chunk 1024",
    "abl_overlap0": "overlap 0",
    "abl_norerank": "hybrid, no rerank",
}


def load_results() -> dict[str, dict]:
    return {n: json.loads((RES / f"{n}.json").read_text()) for n in GROUPS["all"]}


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n)


def table(results: dict, names: list[str]) -> str:
    head = (
        "| Setup | Chunk/overlap | Rerank | recall@5 (±95% CI) | recall@10 | MRR | Faithfulness "
        "| Relevance | Cites gold | Refusal acc. (20) | Over-refusal | $/query | p50 s | p95 s | retrieval p95 s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for n in names:
        r = results[n]
        rows.append(
            f"| {LABEL[n]} | {r['chunking'].replace('s', '').replace('_o', '/')} "
            f"| {'yes' if r['rerank'] else 'no'} "
            f"| {r['recall@5']:.3f} ±{ci95(r['recall@5'], r['n_answerable']):.3f} "
            f"| {r['recall@10']:.3f} | {r['mrr']:.3f} | {r['faithfulness']:.3f} "
            f"| {r['answer_relevance']:.3f} | {r['citation_hits_gold']:.3f} "
            f"| {r['refusal_accuracy']:.2f} "
            f"| {r['over_refusal']:.3f} | ${r['cost_per_query_usd']:.5f} "
            f"| {r['latency_p50_s']:.2f} | {r['latency_p95_s']:.2f} | {r['retrieval_p95_s']:.3f} |"
        )
    return head + "\n".join(rows) + "\n"


def chart(results: dict) -> None:
    DOCS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, r in results.items():
        ax.scatter(
            r["latency_p95_s"],
            r["recall@5"],
            s=60,
            color="tab:red" if n == "hybrid_rerank" else "tab:blue",
        )
        ax.annotate(
            LABEL[n],
            (r["latency_p95_s"], r["recall@5"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("p95 end-to-end latency (s): retrieval + answer")
    ax.set_ylabel("recall@5 (150 answerable questions)")
    ax.set_title("Recall@5 vs p95 latency by configuration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "recall_vs_latency.png", dpi=150)


def failures(name: str = "hybrid_rerank") -> str:
    rows = [json.loads(x) for x in (RES / f"{name}.jsonl").read_text().splitlines()]

    cited_ok = cites_gold

    over = [r for r in rows if r["answerable"] and r["refused"]]
    wrong = [
        r
        for r in rows
        if r["answerable"] and not r["refused"] and r["citations"] and not cited_ok(r)
    ]
    unfaith = sorted((r for r in rows if r["faith"] is not None), key=lambda r: r["faith"])[:8]
    bad_unans = [r for r in rows if not r["answerable"] and not r["refused"]]
    out = [f"# Failure candidates ({name})\n"]
    for title, group in [
        ("Over-refusal", over),
        ("Cited a page other than gold", wrong[:10]),
        ("Lowest faithfulness", unfaith),
        ("Answered an unanswerable", bad_unans),
    ]:
        out.append(f"\n## {title} ({len(group)})\n")
        for r in group:
            out.append(
                f"- **{r['id']}** {r['question']}\n  - gold: {r['gold']} rank: {r['rank']}\n"
                f"  - cited: {[c['source'] for c in r['citations']]} faith: {r['faith']}\n"
                f"  - answer: {r['answer'][:300]!r}\n"
            )
    return "".join(out)


def main() -> None:
    results = load_results()
    DOCS.mkdir(exist_ok=True)
    pages = load_pages()
    meta = f"Corpus: {len(pages)} pages, {len({p['doc'] for p in pages})} PDFs.\n\n"
    judge = sum(r["judge_cost_total_usd"] for r in results.values())
    text = (
        meta
        + "### Retrieval setups (chunk 512 / overlap 64)\n\n"
        + table(results, GROUPS["main"])
        + "\n### Ablations (each row changes one thing from hybrid+rerank)\n\n"
        + table(results, ["hybrid_rerank"] + GROUPS["ablations"])
        + f"\nJudge cost across all 7 runs: ${judge:.3f} (not in $/query).\n"
    )
    (DOCS / "results_tables.md").write_text(text)
    chart(results)
    (DOCS / "failure_candidates.md").write_text(failures())
    print(text)


if __name__ == "__main__":
    assert set(SETUPS) >= set(GROUPS["all"])
    main()
