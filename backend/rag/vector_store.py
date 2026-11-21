"""FAISS index + parallel metadata list, persisted to disk."""
import json
import threading
from collections import Counter

import faiss
import numpy as np

from rag.config import FAISS_FILE, META_FILE


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.lock = threading.RLock()
        if FAISS_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(FAISS_FILE))
            self.metadata: list[dict] = json.loads(META_FILE.read_text(encoding="utf-8"))
        else:
            self.index = faiss.IndexFlatIP(dim)  # cosine similarity on normalized vectors
            self.metadata = []
        self._by_id = {m["chunk_id"]: m for m in self.metadata}

    def _save(self):
        faiss.write_index(self.index, str(FAISS_FILE))
        META_FILE.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")
        self._by_id = {m["chunk_id"]: m for m in self.metadata}

    def add(self, vectors: np.ndarray, chunks: list[dict]):
        with self.lock:
            self.index.add(vectors)
            self.metadata.extend(chunks)
            self._save()

    def remove_source(self, source: str):
        with self.lock:
            keep = [i for i, m in enumerate(self.metadata) if m["source"] != source]
            if len(keep) == len(self.metadata):
                return
            new_index = faiss.IndexFlatIP(self.dim)
            if keep:
                new_index.add(self.index.reconstruct_n(0, self.index.ntotal)[keep])
            self.index = new_index
            self.metadata = [self.metadata[i] for i in keep]
            self._save()

    def search(self, qvec: np.ndarray, top_k: int) -> list[dict]:
        with self.lock:
            if self.index.ntotal == 0:
                return []
            scores, ids = self.index.search(qvec, min(top_k, self.index.ntotal))
            return [
                {**self.metadata[i], "score": float(s)}
                for s, i in zip(scores[0], ids[0])
                if i != -1
            ]

    def get(self, chunk_id: str) -> dict | None:
        return self._by_id.get(chunk_id)

    def file_hashes(self) -> dict[str, str]:
        return {m["source"]: m["file_hash"] for m in self.metadata}

    def chunk_counts(self) -> Counter:
        return Counter(m["source"] for m in self.metadata)

    def chunks_for(self, source: str) -> list[dict]:
        return [m for m in self.metadata if m["source"] == source]

    def reset(self):
        with self.lock:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []
            self._save()

    @property
    def total(self) -> int:
        return self.index.ntotal
