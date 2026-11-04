"""Local embedding model (no API cost). bge-small: 384 dims, 512-token max input."""

import os
from functools import lru_cache

import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DIM = 384


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def hf_offsets(text: str) -> list[tuple[int, int]]:
    """Token char offsets from the embedding model's own tokenizer, so chunk sizes are
    measured in the same tokens the embedder sees."""
    enc = model().tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )
    return [tuple(o) for o in enc["offset_mapping"]]


def embed_passages(texts: list[str]) -> np.ndarray:
    # ponytail: chunks over 512 tokens are truncated for the dense vector only;
    # BM25 and the reranker still see the full text. Noted in the ablation results.
    return model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 500
    )


def embed_query(q: str) -> np.ndarray:
    return model().encode(QUERY_PREFIX + q, normalize_embeddings=True)
