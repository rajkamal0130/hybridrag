"""The three retrieval modes.

vector : plain RAG - FAISS top-k chunks for the question.
graph  : GraphRAG - link question to entities, walk the graph, use the triples plus the
         chunks those triples were extracted from.
hybrid : both, chunk lists merged with Reciprocal Rank Fusion, triples added as extra context.
"""
from collections import Counter

from rag import embedder
from rag.config import GRAPH_HOPS, GRAPH_MAX_FACTS, RRF_K
from rag.graph_store import GraphStore
from rag.vector_store import VectorStore

MODES = ("vector", "graph", "hybrid")


def vector_search(vs: VectorStore, question: str, top_k: int) -> list[dict]:
    hits = vs.search(embedder.embed_query(question), top_k)
    return [{**h, "via": "vector"} for h in hits]


def graph_search(vs: VectorStore, gs: GraphStore, question: str, top_k: int):
    entities = gs.match_entities(question)
    facts = gs.neighborhood([e["id"] for e in entities], GRAPH_HOPS, GRAPH_MAX_FACTS)
    # chunks ranked by how many retrieved facts came from them
    counts = Counter(f["chunk_id"] for f in facts)
    chunks = []
    for cid, n in counts.most_common(top_k):
        c = vs.get(cid)
        if c:
            chunks.append({**c, "score": float(n), "via": "graph"})
    return entities, facts, chunks


def rrf(*ranked_lists: list[dict], top_k: int) -> list[dict]:
    scores, items, via = {}, {}, {}
    for lst in ranked_lists:
        for rank, c in enumerate(lst):
            cid = c["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            items.setdefault(cid, c)
            via.setdefault(cid, set()).add(c["via"])
    best = sorted(scores, key=lambda k: -scores[k])[:top_k]
    return [
        {**items[k], "score": round(scores[k], 5), "via": "+".join(sorted(via[k]))} for k in best
    ]


def retrieve(vs: VectorStore, gs: GraphStore, question: str, mode: str, top_k: int) -> dict:
    entities, facts, chunks = [], [], []
    if mode == "vector":
        chunks = vector_search(vs, question, top_k)
    elif mode == "graph":
        entities, facts, chunks = graph_search(vs, gs, question, top_k)
    elif mode == "hybrid":
        vec = vector_search(vs, question, top_k)
        entities, facts, gchunks = graph_search(vs, gs, question, top_k)
        chunks = rrf(vec, gchunks, top_k=top_k)
    else:
        raise ValueError(f"mode must be one of {MODES}")
    return {"entities": entities, "facts": facts, "chunks": chunks}
