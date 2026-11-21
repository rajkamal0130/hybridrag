"""Ingestion: mirror the pdfs/ folder into FAISS (vector RAG) and the knowledge graph."""
import hashlib
from concurrent.futures import ThreadPoolExecutor

from rag import embedder, llm
from rag.config import EXTRACT_WORKERS, PDF_DIR
from rag.graph_store import GraphStore
from rag.loader import load_pdf
from rag.splitter import split_segments
from rag.vector_store import VectorStore


def list_pdfs():
    return sorted(
        (p for p in PDF_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )


def build_graph_for(gs: GraphStore, chunks: list[dict]) -> dict:
    """Extract entities/relations from every chunk with OpenAI, in parallel."""
    errors = []

    def work(c):
        try:
            return c, llm.extract_kg(c["text"])
        except Exception as e:  # noqa: BLE001 - one bad chunk shouldn't stop the file
            errors.append(f"{c['chunk_id']}: {type(e).__name__}: {e}")
            return c, None

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        for chunk, kg in pool.map(work, chunks):
            if kg:
                gs.add_extraction(chunk, kg)
    gs.save()
    return {"chunks": len(chunks), "failed_chunks": len(errors), "errors": errors[:5]}


def sync_folder(vs: VectorStore, gs: GraphStore, build_graph: bool = True) -> dict:
    on_disk = {p.name: p for p in list_pdfs()}
    indexed = vs.file_hashes()
    report = {"added": [], "updated": [], "removed": [], "unchanged": [], "failed": [], "graph": {}}

    for name in sorted(set(indexed) - set(on_disk)):
        vs.remove_source(name)
        gs.remove_source(name)
        report["removed"].append(name)

    for name, path in on_disk.items():
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if indexed.get(name) == file_hash:
            report["unchanged"].append(name)
            continue
        is_update = name in indexed
        try:
            chunks = split_segments(load_pdf(path))
        except Exception as e:  # noqa: BLE001 - corrupt / encrypted PDF
            report["failed"].append({"file": name, "reason": str(e)})
            continue
        if not chunks:
            report["failed"].append({"file": name, "reason": "No extractable text (scanned PDF?)"})
            continue
        if is_update:
            vs.remove_source(name)
            gs.remove_source(name)
        for c in chunks:
            c["file_hash"] = file_hash
        vs.add(embedder.embed([c["text"] for c in chunks]), chunks)
        report["updated" if is_update else "added"].append(name)

    if build_graph:
        # (re)build the graph for every indexed file that has no graph yet
        in_graph = gs.sources()
        for name in sorted(vs.chunk_counts()):
            if name not in in_graph:
                report["graph"][name] = build_graph_for(gs, vs.chunks_for(name))

    report["total_vectors"] = vs.total
    report["graph_stats"] = gs.stats
    return report
