"""Knowledge graph: NetworkX MultiDiGraph of (entity) -[relation]-> (entity).

NetworkX is the in-memory working graph (entity linking, traversal, export). It is persisted
to Neo4j when NEO4J_URI is set (see neo4j_store.py), and always snapshotted to graph.json.

Every node and edge remembers the chunk_ids it came from, so graph facts can always be
traced back to a page in a PDF.
"""
import json
import logging
import threading

import networkx as nx
import numpy as np

from rag import embedder
from rag.config import ENTITY_MATCH_THRESHOLD, ENTITY_TOP_K, GRAPH_FILE, GRAPH_QUERY_ENGINE, NEO4J_URI

log = logging.getLogger(__name__)


def norm(name: str) -> str:
    return " ".join(name.lower().strip().split())


class GraphStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.g = nx.MultiDiGraph()
        self._keys: list[str] = []
        self._vecs: np.ndarray | None = None
        self._dirty: set[str] = set()  # sources with edges not yet pushed to Neo4j
        self.neo4j = None
        self.neo4j_error: str | None = None
        if NEO4J_URI:
            try:
                from rag.neo4j_store import Neo4jStore
                self.neo4j = Neo4jStore()
            except Exception as e:  # noqa: BLE001 - keep serving from JSON if Neo4j is down
                self.neo4j_error = f"{type(e).__name__}: {e}"
                log.warning("Neo4j unavailable, using graph.json only: %s", self.neo4j_error)

        if self.neo4j and not self.neo4j.is_empty():
            self._load(self.neo4j.load())
        elif GRAPH_FILE.exists():
            self._load(json.loads(GRAPH_FILE.read_text(encoding="utf-8")))
            if self.neo4j and self.g.number_of_nodes():  # first run with Neo4j: migrate graph.json
                self._dirty = self.sources()
                self._push()

    @property
    def backend(self) -> str:
        return "neo4j" if self.neo4j else "json"

    # ---------- persistence ----------
    def _load(self, data: dict):
        for n in data.get("nodes", []):
            self.g.add_node(n["id"], name=n["name"], type=n["type"], chunk_ids=set(n["chunk_ids"]))
        for e in data.get("edges", []):
            self.g.add_edge(e["s"], e["t"], **{k: v for k, v in e.items() if k not in ("s", "t")})

    def _save(self):
        data = {
            "nodes": [
                {"id": n, "name": d["name"], "type": d["type"], "chunk_ids": sorted(d["chunk_ids"])}
                for n, d in self.g.nodes(data=True)
            ],
            "edges": [{"s": u, "t": v, **d} for u, v, d in self.g.edges(data=True)],
        }
        GRAPH_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self._vecs = None  # node-name embeddings are rebuilt lazily

    def _push(self):
        """Write the relations of dirty sources (and every node they touch) to Neo4j."""
        if not self.neo4j or not self._dirty:
            self._dirty.clear()
            return
        sources = sorted(self._dirty)
        edges = [{"s": u, "t": v, **d} for u, v, d in self.g.edges(data=True) if d["source"] in self._dirty]
        prefixes = tuple(f"{s}::" for s in sources)
        keys = {e["s"] for e in edges} | {e["t"] for e in edges} | {
            n for n, d in self.g.nodes(data=True) if any(c.startswith(prefixes) for c in d["chunk_ids"])
        }
        nodes = [
            {"id": n, "name": self.g.nodes[n]["name"], "type": self.g.nodes[n]["type"],
             "chunk_ids": sorted(self.g.nodes[n]["chunk_ids"])}
            for n in keys
        ]
        self.neo4j.replace_sources(sources, nodes, edges)
        self._dirty.clear()

    # ---------- writes ----------
    def _node(self, name: str, etype: str, chunk_id: str) -> str | None:
        key = norm(name)
        if not key:
            return None
        if key in self.g:
            d = self.g.nodes[key]
            d["chunk_ids"].add(chunk_id)
            if d["type"] == "Concept" and etype != "Concept":
                d["type"] = etype
        else:
            self.g.add_node(key, name=name.strip(), type=etype or "Concept", chunk_ids={chunk_id})
        return key

    def add_extraction(self, chunk: dict, kg: dict):
        """kg = {"entities": [{name, type}], "relations": [{subject, predicate, object}]}"""
        with self.lock:
            self._dirty.add(chunk["source"])
            types = {norm(e["name"]): e.get("type", "Concept") for e in kg.get("entities", [])}
            for e in kg.get("entities", []):
                self._node(e["name"], e.get("type", "Concept"), chunk["chunk_id"])
            for r in kg.get("relations", []):
                s = self._node(r["subject"], types.get(norm(r["subject"]), "Concept"), chunk["chunk_id"])
                o = self._node(r["object"], types.get(norm(r["object"]), "Concept"), chunk["chunk_id"])
                if s and o and s != o and r.get("predicate", "").strip():
                    self.g.add_edge(
                        s, o,
                        relation=r["predicate"].strip().lower(),
                        chunk_id=chunk["chunk_id"], source=chunk["source"], page=chunk["page"],
                    )

    def save(self):
        with self.lock:
            self._save()
            self._push()

    def remove_source(self, source: str):
        with self.lock:
            prefix = f"{source}::"
            drop = [(u, v, k) for u, v, k, d in self.g.edges(keys=True, data=True) if d["source"] == source]
            self.g.remove_edges_from(drop)
            for n, d in list(self.g.nodes(data=True)):
                d["chunk_ids"] = {c for c in d["chunk_ids"] if not c.startswith(prefix)}
                if not d["chunk_ids"]:
                    self.g.remove_node(n)
            self._save()
            if self.neo4j:
                self.neo4j.remove_source(source)

    def reset(self):
        with self.lock:
            self.g = nx.MultiDiGraph()
            self._dirty.clear()
            self._save()
            if self.neo4j:
                self.neo4j.reset()

    # ---------- reads ----------
    def sources(self) -> set[str]:
        return {d["source"] for _, _, d in self.g.edges(data=True)} | {
            c.split("::")[0] for _, d in self.g.nodes(data=True) for c in d["chunk_ids"]
        }

    def _ensure_vecs(self):
        if self._vecs is None:
            self._keys = list(self.g.nodes)
            names = [self.g.nodes[k]["name"] for k in self._keys]
            self._vecs = embedder.embed(names) if names else None

    def match_entities(self, question: str) -> list[dict]:
        """Link the question to graph nodes: exact name mentions first, then embedding similarity."""
        with self.lock:
            if self.g.number_of_nodes() == 0:
                return []
            q = f" {norm(question)} "
            hits: dict[str, float] = {}
            for k in self.g.nodes:
                if len(k) >= 3 and f" {k} " in q:
                    hits[k] = 1.0
            self._ensure_vecs()
            if self._vecs is not None:
                sims = self._vecs @ embedder.embed([question])[0]
                for i in np.argsort(-sims)[: ENTITY_TOP_K * 2]:
                    if sims[i] >= ENTITY_MATCH_THRESHOLD:
                        hits.setdefault(self._keys[i], float(sims[i]))
            ranked = sorted(hits.items(), key=lambda kv: -kv[1])[:ENTITY_TOP_K]
            return [
                {"id": k, "name": self.g.nodes[k]["name"], "type": self.g.nodes[k]["type"], "score": round(s, 4)}
                for k, s in ranked
            ]

    def neighborhood(self, seeds: list[str], hops: int, max_facts: int) -> list[dict]:
        """Walk outward (both edge directions) from the seed entities and collect triples."""
        if self.neo4j and GRAPH_QUERY_ENGINE == "neo4j":
            return self.neo4j.neighborhood(seeds, hops, max_facts)
        with self.lock:
            facts, seen_edges = [], set()
            frontier, visited = list(seeds), set(seeds)
            for _ in range(max(1, hops)):
                nxt = []
                for n in frontier:
                    edges = list(self.g.out_edges(n, keys=True, data=True)) + list(
                        self.g.in_edges(n, keys=True, data=True)
                    )
                    for u, v, k, d in edges:
                        if (u, v, k) in seen_edges:
                            continue
                        seen_edges.add((u, v, k))
                        facts.append({
                            "subject": self.g.nodes[u]["name"], "relation": d["relation"],
                            "object": self.g.nodes[v]["name"], "chunk_id": d["chunk_id"],
                            "source": d["source"], "page": d["page"],
                        })
                        if len(facts) >= max_facts:
                            return facts
                        for m in (u, v):
                            if m not in visited:
                                visited.add(m)
                                nxt.append(m)
                frontier = nxt
            return facts

    def export(self, limit: int = 300, focus: list[str] | None = None) -> dict:
        """Nodes + edges for visualisation (optionally only around some entities)."""
        if self.neo4j and GRAPH_QUERY_ENGINE == "neo4j":
            return self.neo4j.export(limit, focus)
        with self.lock:
            g = self.g
            if focus:
                keep = set(focus)
                for f in focus:
                    if f in g:
                        keep |= set(nx.all_neighbors(g, f))
                g = g.subgraph(keep)
            nodes = sorted(g.nodes, key=lambda n: -g.degree(n))[:limit]
            ns = set(nodes)
            return {
                "nodes": [
                    {"id": n, "name": g.nodes[n]["name"], "type": g.nodes[n]["type"], "degree": g.degree(n)}
                    for n in nodes
                ],
                "edges": [
                    {"source": u, "target": v, "relation": d["relation"], "page": d["page"], "file": d["source"]}
                    for u, v, d in g.edges(data=True) if u in ns and v in ns
                ],
            }

    @property
    def stats(self) -> dict:
        s = {
            "entities": self.g.number_of_nodes(),
            "relations": self.g.number_of_edges(),
            "store": self.backend,
            "query_engine": GRAPH_QUERY_ENGINE if self.neo4j else "networkx",
        }
        if self.neo4j:
            try:  # live counts straight from the database
                s["neo4j"] = self.neo4j.counts()
            except Exception as e:  # noqa: BLE001
                s["neo4j_error"] = f"{type(e).__name__}: {e}"
        if self.neo4j_error:
            s["neo4j_error"] = self.neo4j_error
        return s
