"""FastAPI backend: RAG, GraphRAG and Hybrid RAG over PDFs.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000   (from the backend/ folder)
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import openai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import embedder, llm
from rag.config import LLM_MODEL, PDF_DIR, TOP_K
from rag.graph_store import GraphStore
from rag.ingest import list_pdfs, sync_folder
from rag.retriever import retrieve
from rag.vector_store import VectorStore

vs: VectorStore | None = None
gs: GraphStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vs, gs
    vs = VectorStore(embedder.dimension())
    gs = GraphStore()
    sync_folder(vs, gs, build_graph=False)  # vectors only on start; the graph costs LLM calls
    yield
    if gs.neo4j:
        gs.neo4j.close()


app = FastAPI(title="Hybrid RAG API (FAISS + Knowledge Graph: NetworkX + Neo4j)", lifespan=lifespan)

Mode = Literal["vector", "graph", "hybrid"]


class AskRequest(BaseModel):
    question: str
    mode: Mode = "hybrid"
    top_k: int = TOP_K


def _public(r: dict) -> dict:
    return {
        "entities": r["entities"],
        "facts": r["facts"],
        "chunks": [
            {k: c[k] for k in ("chunk_id", "source", "page", "text", "via")} | {"score": round(c["score"], 4)}
            for c in r["chunks"]
        ],
    }


def _retrieve(req: AskRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    return retrieve(vs, gs, req.question, req.mode, req.top_k)


@app.get("/health")
def health():
    return {"status": "ok", "vectors": vs.total, "graph": gs.stats, "model": LLM_MODEL}


# ---------- documents ----------
@app.get("/pdfs")
def pdfs():
    counts, in_graph = vs.chunk_counts(), gs.sources()
    return {
        "folder": str(PDF_DIR),
        "pdfs": [
            {
                "name": p.name,
                "size_kb": round(p.stat().st_size / 1024, 1),
                "chunks": counts.get(p.name, 0),
                "vector_indexed": p.name in counts,
                "graph_built": p.name in in_graph,
            }
            for p in list_pdfs()
        ],
        "total_vectors": vs.total,
        "graph": gs.stats,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = (file.filename or "").replace("/", "_").replace("\\", "_")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")
    (PDF_DIR / name).write_bytes(await file.read())
    return {"saved": name}


@app.get("/pdfs/{name}")
def view_pdf(name: str):
    path = (PDF_DIR / name).resolve()
    if path.parent != PDF_DIR.resolve() or not path.is_file():
        raise HTTPException(404, "PDF not found")
    return FileResponse(path, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{path.name}"'})


@app.post("/sync")
def sync(build_graph: bool = True):
    """Index new/changed PDFs into FAISS and (optionally) extract the knowledge graph with OpenAI."""
    return sync_folder(vs, gs, build_graph=build_graph)


@app.post("/graph/rebuild")
def rebuild_graph():
    gs.reset()
    return sync_folder(vs, gs, build_graph=True)


# ---------- knowledge graph ----------
@app.get("/graph")
def graph(limit: int = 300, entity: str | None = None):
    focus = [m["id"] for m in gs.match_entities(entity)] if entity else None
    return gs.export(limit=limit, focus=focus)


# ---------- question answering ----------
@app.post("/query")
def query(req: AskRequest):
    """Retrieval only (no LLM call)."""
    return {"question": req.question, "mode": req.mode, **_public(_retrieve(req))}


@app.post("/ask")
def ask(req: AskRequest):
    r = _retrieve(req)
    try:
        text = llm.answer(req.question, r["chunks"], r["facts"])
    except openai.AuthenticationError:
        raise HTTPException(401, "LLM auth failed: set OPENAI_API_KEY before starting the backend")
    except openai.RateLimitError:
        raise HTTPException(429, "LLM rate limited - try again shortly")
    except openai.APIStatusError as e:
        raise HTTPException(502, f"LLM error ({e.status_code}): {e.message}")
    except openai.APIConnectionError:
        raise HTTPException(503, "Could not reach the LLM API - check the network")
    return {"question": req.question, "mode": req.mode, "answer": text, **_public(r)}


@app.delete("/reset")
def reset():
    vs.reset()
    gs.reset()
    return {"status": "vector index and graph cleared"}


# ---------- web UI (registered last so the API routes above take precedence) ----------
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
