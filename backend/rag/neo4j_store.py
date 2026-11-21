"""Neo4j persistence for the knowledge graph.

Model:  (:Entity {key, name, type, chunk_ids})-[:RELATES_TO {relation, chunk_id, source, page}]->(:Entity)

NetworkX (graph_store.py) stays the in-memory working copy; this class mirrors every write
into Neo4j, loads the graph back on startup, and can answer neighbourhood queries in Cypher.
"""
import logging
import time

from neo4j import GraphDatabase

from rag.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

log = logging.getLogger(__name__)
BATCH = 1000


def _batches(items: list, size: int = BATCH):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Neo4jStore:
    def __init__(self, retries: int = 30, delay: float = 2.0):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        for attempt in range(1, retries + 1):  # Neo4j can take a while to boot in Docker
            try:
                self.driver.verify_connectivity()
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries:
                    self.driver.close()
                    raise
                log.info("Waiting for Neo4j at %s (%s/%s): %s", NEO4J_URI, attempt, retries, e)
                time.sleep(delay)
        self._run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")
        self._run("CREATE INDEX rel_source IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.source)")

    def _run(self, cypher: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(cypher, params, database_=NEO4J_DATABASE)
        return [r.data() for r in records]

    def close(self):
        self.driver.close()

    # ---------- reads ----------
    def load(self) -> dict:
        nodes = self._run(
            "MATCH (e:Entity) RETURN e.key AS id, e.name AS name, e.type AS type, e.chunk_ids AS chunk_ids"
        )
        edges = self._run(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            "RETURN a.key AS s, b.key AS t, r.relation AS relation, r.chunk_id AS chunk_id, "
            "r.source AS source, r.page AS page"
        )
        return {"nodes": nodes, "edges": edges}

    def is_empty(self) -> bool:
        return not self._run("MATCH (e:Entity) RETURN e LIMIT 1")

    def neighborhood(self, seeds: list[str], hops: int, max_facts: int) -> list[dict]:
        """Same contract as GraphStore.neighborhood, done as a variable-length Cypher match."""
        if not seeds:
            return []
        hops = max(1, int(hops))  # path length can't be a query parameter
        return self._run(
            f"MATCH p = (s:Entity)-[:RELATES_TO*1..{hops}]-() WHERE s.key IN $seeds "
            "UNWIND relationships(p) AS r "
            "WITH r, min(length(p)) AS dist ORDER BY dist LIMIT $max "
            "RETURN startNode(r).name AS subject, r.relation AS relation, endNode(r).name AS object, "
            "r.chunk_id AS chunk_id, r.source AS source, r.page AS page",
            seeds=seeds, max=max_facts,
        )

    def export(self, limit: int, focus: list[str] | None = None) -> dict:
        """Same contract as GraphStore.export: top-`limit` nodes by degree, plus edges between them."""
        keys = None
        if focus:
            rows = self._run(
                "MATCH (f:Entity) WHERE f.key IN $focus "
                "OPTIONAL MATCH (f)-[:RELATES_TO]-(n:Entity) "
                "RETURN collect(DISTINCT f.key) + collect(DISTINCT n.key) AS keys",
                focus=focus,
            )
            keys = list(set(rows[0]["keys"])) if rows else []
        nodes = self._run(
            "MATCH (e:Entity) WHERE $keys IS NULL OR e.key IN $keys "
            "OPTIONAL MATCH (e)-[r:RELATES_TO]-(o:Entity) WHERE $keys IS NULL OR o.key IN $keys "
            "WITH e, count(r) AS degree ORDER BY degree DESC, e.key LIMIT $limit "
            "RETURN e.key AS id, e.name AS name, e.type AS type, degree",
            keys=keys, limit=limit,
        )
        ids = [n["id"] for n in nodes]
        edges = self._run(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.key IN $ids AND b.key IN $ids "
            "RETURN a.key AS source, b.key AS target, r.relation AS relation, r.page AS page, r.source AS file",
            ids=ids,
        )
        return {"nodes": nodes, "edges": edges}

    def counts(self) -> dict:
        rows = self._run(
            "CALL { MATCH (e:Entity) RETURN count(e) AS entities } "
            "CALL { MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS relations } "
            "RETURN entities, relations"
        )
        return rows[0]

    # ---------- writes ----------
    def replace_sources(self, sources: list[str], nodes: list[dict], edges: list[dict]):
        """Drop the relations of `sources`, then upsert `nodes` and recreate `edges`."""
        self._run("MATCH ()-[r:RELATES_TO]->() WHERE r.source IN $sources DELETE r", sources=sources)
        for b in _batches(nodes):
            self._run(
                "UNWIND $rows AS n MERGE (e:Entity {key: n.id}) "
                "SET e.name = n.name, e.type = n.type, e.chunk_ids = n.chunk_ids",
                rows=b,
            )
        for b in _batches(edges):
            self._run(
                "UNWIND $rows AS x "
                "MATCH (a:Entity {key: x.s}), (b:Entity {key: x.t}) "
                "CREATE (a)-[:RELATES_TO {relation: x.relation, chunk_id: x.chunk_id, "
                "source: x.source, page: x.page}]->(b)",
                rows=b,
            )

    def remove_source(self, source: str):
        self._run("MATCH ()-[r:RELATES_TO]->() WHERE r.source = $s DELETE r", s=source)
        self._run(
            "MATCH (e:Entity) WHERE any(c IN e.chunk_ids WHERE c STARTS WITH $p) "
            "SET e.chunk_ids = [c IN e.chunk_ids WHERE NOT c STARTS WITH $p]",
            p=f"{source}::",
        )
        self._run("MATCH (e:Entity) WHERE size(e.chunk_ids) = 0 DETACH DELETE e")

    def reset(self):
        self._run("MATCH (e:Entity) DETACH DELETE e")
