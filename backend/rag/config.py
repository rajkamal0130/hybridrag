"""All settings in one place. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Local runs: read the repo-root .env (real environment variables always win).
# In Docker, compose passes the same values in as environment variables.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# TLS trust store for neo4j+s:// (Aura). python.org's macOS Python ships without a CA bundle,
# which makes Aura fail with "Unable to retrieve routing information"; certifi's bundle fixes it.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

# In Docker DATA_DIR=/data (a mounted volume). Locally it defaults to docker-demo/data.
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
FAISS_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "metadata.json"
GRAPH_FILE = INDEX_DIR / "graph.json"

# Embeddings: fastembed (ONNX, no torch), 384 dimensions
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_CACHE = os.getenv("MODEL_CACHE") or None

# Chunking + retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Knowledge graph retrieval
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "1"))            # how far to walk from matched entities
GRAPH_MAX_FACTS = int(os.getenv("GRAPH_MAX_FACTS", "40"))  # cap on triples sent to the LLM
ENTITY_MATCH_THRESHOLD = float(os.getenv("ENTITY_MATCH_THRESHOLD", "0.6"))
ENTITY_TOP_K = int(os.getenv("ENTITY_TOP_K", "6"))
RRF_K = 60  # reciprocal-rank-fusion constant

# Neo4j (optional). Leave NEO4J_URI empty to keep the graph in graph.json only.
# When enabled, Neo4j is the source of truth and answers graph queries; NetworkX keeps an in-memory copy.
NEO4J_URI = os.getenv("NEO4J_URI", "")   # bolt://localhost:7687 or neo4j+s://<id>.databases.neo4j.io
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"  # Aura uses NEO4J_USERNAME
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
# Which engine answers graph queries (GraphRAG traversal + /graph explorer):
# "neo4j" (Cypher against the database, default) or "networkx" (in-memory copy).
# Falls back to networkx automatically when Neo4j isn't configured or reachable.
GRAPH_QUERY_ENGINE = os.getenv("GRAPH_QUERY_ENGINE", "neo4j").lower()

# LLM (OpenAI) - needs OPENAI_API_KEY in the environment
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.5")
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "")  # optional: low | medium | high
EXTRACT_WORKERS = int(os.getenv("EXTRACT_WORKERS", "4"))

PDF_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
