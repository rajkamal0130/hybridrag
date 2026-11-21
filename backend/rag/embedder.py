"""Embed: text -> normalized 384-dim vectors (fastembed, runs on CPU, no torch)."""
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from rag.config import EMBED_MODEL, MODEL_CACHE


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE)


def _normalize(vecs) -> np.ndarray:
    arr = np.asarray(list(vecs), dtype="float32")
    if arr.size == 0:
        return arr.reshape(0, dimension())
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def embed(texts: list[str]) -> np.ndarray:
    """Passages / entity names."""
    return _normalize(get_model().embed(texts, batch_size=64))


def embed_query(text: str) -> np.ndarray:
    """Questions (bge models use a query prefix; fastembed adds it)."""
    return _normalize(get_model().query_embed(text))


@lru_cache(maxsize=1)
def dimension() -> int:
    return len(next(iter(get_model().embed(["dimension probe"]))))
