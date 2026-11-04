Corpus: 2019 pages, 34 PDFs.

### Retrieval setups (chunk 512 / overlap 64)

| Setup | Chunk/overlap | Rerank | recall@5 (±95% CI) | recall@10 | MRR | Faithfulness | Relevance | Cites gold | Refusal acc. (20) | Over-refusal | $/query | p50 s | p95 s | retrieval p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 512/64 | no | 0.780 ±0.066 | 0.867 | 0.653 | 0.949 | 0.885 | 0.784 | 1.00 | 0.073 | $0.00047 | 1.07 | 1.54 | 0.035 |
| bm25 | 512/64 | no | 0.887 ±0.051 | 0.940 | 0.758 | 0.977 | 0.887 | 0.887 | 1.00 | 0.053 | $0.00046 | 1.01 | 1.62 | 0.006 |
| hybrid+rerank (base) | 512/64 | yes | 0.887 ±0.051 | 0.920 | 0.774 | 0.965 | 0.917 | 0.890 | 1.00 | 0.033 | $0.00047 | 1.25 | 1.76 | 0.225 |

### Ablations (each row changes one thing from hybrid+rerank)

| Setup | Chunk/overlap | Rerank | recall@5 (±95% CI) | recall@10 | MRR | Faithfulness | Relevance | Cites gold | Refusal acc. (20) | Over-refusal | $/query | p50 s | p95 s | retrieval p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid+rerank (base) | 512/64 | yes | 0.887 ±0.051 | 0.920 | 0.774 | 0.965 | 0.917 | 0.890 | 1.00 | 0.033 | $0.00047 | 1.25 | 1.76 | 0.225 |
| chunk 256 | 256/64 | yes | 0.933 ±0.040 | 0.967 | 0.832 | 0.968 | 0.897 | 0.932 | 1.00 | 0.020 | $0.00028 | 1.19 | 1.72 | 0.353 |
| chunk 1024 | 1024/64 | yes | 0.720 ±0.072 | 0.807 | 0.549 | 0.947 | 0.853 | 0.728 | 0.95 | 0.093 | $0.00085 | 1.24 | 2.86 | 0.186 |
| overlap 0 | 512/0 | yes | 0.873 ±0.053 | 0.927 | 0.740 | 0.965 | 0.929 | 0.901 | 1.00 | 0.060 | $0.00047 | 1.29 | 3.33 | 0.221 |
| hybrid, no rerank | 512/64 | no | 0.873 ±0.053 | 0.913 | 0.738 | 0.970 | 0.900 | 0.847 | 1.00 | 0.040 | $0.00047 | 1.03 | 1.55 | 0.042 |

Judge cost across all 7 runs: $0.477 (not in $/query).
