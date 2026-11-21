# docker-demo: RAG, GraphRAG and Hybrid RAG over PDFs

FastAPI backend + Streamlit frontend, each in its own container. Sample document:
`data/pdfs/spotify_web_app_architecture.pdf` (the Spotify-like streaming app architecture).

| Mode | How it retrieves context |
|---|---|
| **RAG** (`vector`) | Question embedding → FAISS top-k chunks |
| **GraphRAG** (`graph`) | Question → matched entities in the knowledge graph → walk N hops → triples + the chunks they came from |
| **Hybrid RAG** (`hybrid`) | Both lists merged with Reciprocal Rank Fusion, plus the graph triples as extra context |

OpenAI (`gpt-5.5`, same as askmypdf) does two jobs: extracts entities/relations from every chunk to build
the graph (strict JSON-schema output), and writes the final answer with `[file, p. N]` citations.
Embeddings stay local and free (fastembed).

## Knowledge graph: NetworkX + Neo4j

| Layer | Role |
|---|---|
| **NetworkX** (`MultiDiGraph`) | In-memory copy: entity linking, plus traversal/export when `GRAPH_QUERY_ENGINE=networkx` or Neo4j is down |
| **Neo4j** | Source of truth: every write goes to it, the graph is loaded from it on start, and GraphRAG traversal + `/graph` run as Cypher |
| `graph.json` | Snapshot written on every save; used when `NEO4J_URI` is unset or Neo4j is unreachable |

Neo4j schema: `(:Entity {key, name, type, chunk_ids})-[:RELATES_TO {relation, chunk_id, source, page}]->(:Entity)`.
On the first start with Neo4j, an existing `graph.json` is migrated into it automatically.
By default (`GRAPH_QUERY_ENGINE=neo4j`) the GraphRAG traversal and the graph explorer run as Cypher queries against Neo4j;
set `GRAPH_QUERY_ENGINE=networkx` to answer them from the in-memory copy instead (automatic when Neo4j is unreachable).

Explore the graph in the Neo4j Browser at http://localhost:7474 (user `neo4j`, password `NEO4J_PASSWORD`, default `hybridrag123`):

```cypher
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a, r, b LIMIT 100;
MATCH (s:Entity {key: 'postgresql'})<-[r]-(svc) RETURN svc.name, r.relation;
```

## Run

```bash
cp .env.example .env          # put OPENAI_API_KEY and the NEO4J_* values in .env
docker compose up --build                          # Neo4j from .env (e.g. Aura)
docker compose --profile local-neo4j up --build    # or: run Neo4j locally in a container
```

All settings live in `.env` (gitignored). Compose passes them to the backend, and `backend/rag/config.py`
also loads `.env` itself, so a local `uvicorn api:app` picks up the same values. Real environment variables override `.env`.

| `NEO4J_URI` in `.env` | Graph store used |
|---|---|
| `neo4j+s://<id>.databases.neo4j.io` | Neo4j Aura (set `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` too) |
| not set | Local container `bolt://neo4j:7687` (needs `--profile local-neo4j`) |
| empty (`NEO4J_URI=`) | No Neo4j: `graph.json` only |

### Run locally (no Docker)

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements-local.txt   # once

cd backend && ../.venv/bin/uvicorn api:app --port 8000 --reload   # reads ../.env
```

Open **http://localhost:8000**. The web UI (`backend/static/`: plain HTML/CSS/JS, no build step) is served by
the backend itself: Ask, Compare modes, an interactive Knowledge graph explorer, and Documents.
The older Streamlit UI in `frontend/` still works (`cd frontend && ../.venv/bin/streamlit run app.py`).

With no `NEO4J_URI` (or Neo4j unreachable) the backend uses `graph.json`; it can't reach the compose-only
`bolt://neo4j:7687` host outside Docker.

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474 (local profile) or the Aura console for Aura

On start the backend indexes vectors for every PDF in `data/pdfs` (free, local embeddings).
The knowledge graph is built only when you click **Index + build knowledge graph** on the Documents page
(or `POST /sync`), because it makes one OpenAI call per chunk.

## Pages

1. **Documents**: upload PDFs, index vectors, build/rebuild the graph, preview PDFs
2. **Ask**: pick a mode, get an answer plus the entities, triples and chunks used
3. **Compare modes**: same question through all three modes side by side
4. **Knowledge graph**: interactive graph (pyvis), optionally focused on one entity

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Vector count, graph size, graph store (`neo4j`/`json`) and query engine |
| `GET /pdfs` · `POST /upload` · `GET /pdfs/{name}` | Documents |
| `POST /sync?build_graph=true` | Index new/changed PDFs; extract graph for files that have none |
| `POST /graph/rebuild` | Clear and re-extract the whole graph |
| `GET /graph?entity=Kafka&limit=300` | Nodes/edges for visualisation |
| `POST /query` | Retrieval only: `{"question", "mode": "vector|graph|hybrid", "top_k"}` |
| `POST /ask` | Retrieval + OpenAI answer (same body) |
| `DELETE /reset` | Clear vectors and graph |

## Layout

```
docker-demo/
├── docker-compose.yml   .env.example
├── backend/   Dockerfile  requirements.txt  api.py
│   └── rag/   config  loader  splitter  embedder  vector_store
│              graph_store (NetworkX)  neo4j_store  llm  retriever  ingest
├── frontend/  Dockerfile  requirements.txt  app.py
└── data/      pdfs/ (input)   index/ (FAISS + metadata.json + graph.json)
```

## Tuning (env vars)

`TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `GRAPH_HOPS` (default 1), `GRAPH_MAX_FACTS` (40),
`ENTITY_MATCH_THRESHOLD` (0.6), `EXTRACT_WORKERS` (4), `OPENAI_REASONING_EFFORT` (unset by default), `LLM_MODEL` (`gpt-5.5`),
`NEO4J_URI`, `NEO4J_USERNAME` (or `NEO4J_USER`), `NEO4J_PASSWORD`, `NEO4J_DATABASE`,
`GRAPH_QUERY_ENGINE` (`neo4j` default | `networkx`).

## Sample questions for the Spotify PDF

- Which data stores does the platform use, and what does each hold? *(vector does well)*
- What happens between a creator uploading a track and it being playable? *(graph/hybrid connect services and events)*
- Which services depend on PostgreSQL? *(graph)*
