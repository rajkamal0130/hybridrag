"""Streamlit UI for RAG / GraphRAG / Hybrid RAG."""
import os

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

API_URL = os.getenv("API_URL", "http://localhost:8000")          # used by this server
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000")  # used by your browser

MODES = {
    "vector": "RAG (vector only)",
    "graph": "GraphRAG (knowledge graph)",
    "hybrid": "Hybrid RAG (vector + graph)",
}

st.set_page_config(page_title="Hybrid RAG", page_icon="🕸️", layout="wide")
st.title("🕸️ Hybrid RAG: vectors + knowledge graph")
page = st.sidebar.radio("Page", ["1 · Documents", "2 · Ask", "3 · Compare modes", "4 · Knowledge graph"])


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=kw.pop("timeout", 60), **kw)
    except requests.ConnectionError:
        st.error(f"Backend not reachable at {API_URL}. Is the backend container running?")
        st.stop()
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(detail)
        st.stop()
    return r.json()


def show_retrieval(res: dict, key: str):
    if res.get("entities"):
        st.markdown("**Matched entities:** " + ", ".join(f"`{e['name']}` ({e['type']})" for e in res["entities"]))
    if res.get("facts"):
        with st.expander(f"Graph facts used ({len(res['facts'])})"):
            st.dataframe(
                pd.DataFrame(res["facts"])[["subject", "relation", "object", "source", "page"]],
                hide_index=True, use_container_width=True,
            )
    st.markdown(f"**Chunks used ({len(res['chunks'])})**")
    if not res["chunks"]:
        st.info("No chunks retrieved.")
    for i, c in enumerate(res["chunks"], 1):
        with st.expander(f"#{i} · {c['source']} p.{c['page']} · via {c['via']} · score {c['score']}"):
            st.write(c["text"])


# ---------------------------------------------------------------- 1 · Documents
if page.startswith("1"):
    st.header("Documents")
    up = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if up and st.button("Save uploads"):
        for f in up:
            api("POST", "/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
        st.success(f"Saved {len(up)} file(s). Now index them below.")

    data = api("GET", "/pdfs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Vectors (chunks)", data["total_vectors"])
    c2.metric("Graph entities", data["graph"]["entities"])
    c3.metric("Graph relations", data["graph"]["relations"])

    b1, b2, b3 = st.columns(3)
    if b1.button("① Index vectors only (free)", use_container_width=True):
        with st.spinner("Loading, splitting and embedding…"):
            st.json(api("POST", "/sync", params={"build_graph": False}, timeout=1800))
    if b2.button("② Index + build knowledge graph (uses OpenAI)", type="primary", use_container_width=True):
        with st.spinner("Embedding chunks and extracting entities/relations with OpenAI…"):
            st.json(api("POST", "/sync", params={"build_graph": True}, timeout=3600))
    if b3.button("Rebuild graph from scratch", use_container_width=True):
        with st.spinner("Re-extracting the whole graph…"):
            st.json(api("POST", "/graph/rebuild", timeout=3600))

    if data["pdfs"]:
        st.dataframe(
            [
                {"PDF": p["name"], "Size (KB)": p["size_kb"], "Chunks": p["chunks"],
                 "Vectors": "✅" if p["vector_indexed"] else "⏳",
                 "Graph": "✅" if p["graph_built"] else "⏳"}
                for p in data["pdfs"]
            ],
            hide_index=True, use_container_width=True,
        )
        choice = st.selectbox("Preview a PDF", [p["name"] for p in data["pdfs"]])
        url = f"{PUBLIC_API_URL}/pdfs/{requests.utils.quote(choice)}"
        st.markdown(f"[Open in new tab]({url})")
        components.iframe(url, height=700)
    else:
        st.info(f"No PDFs yet. Upload some, or copy them into {data['folder']}.")

# ---------------------------------------------------------------- 2 · Ask
elif page.startswith("2"):
    st.header("Ask your PDFs")
    q = st.text_area("Question", placeholder="e.g. Which services publish events to Kafka, and who consumes them?")
    c1, c2, c3 = st.columns([2, 1, 1])
    mode = c1.radio("Retrieval mode", list(MODES), format_func=MODES.get, index=2, horizontal=True)
    k = c2.slider("top-k chunks", 1, 20, 5)
    retrieval_only = c3.toggle("Retrieval only (no LLM)")

    if st.button("Ask", type="primary", disabled=not q.strip()):
        body = {"question": q, "mode": mode, "top_k": k}
        if retrieval_only:
            res = api("POST", "/query", json=body)
        else:
            with st.spinner("Retrieving and asking OpenAI…"):
                res = api("POST", "/ask", json=body, timeout=300)
            st.subheader("Answer")
            st.markdown(res["answer"])
        st.subheader("Retrieved context")
        show_retrieval(res, "ask")

# ---------------------------------------------------------------- 3 · Compare
elif page.startswith("3"):
    st.header("Compare RAG vs GraphRAG vs Hybrid")
    st.caption("Same question, three retrieval strategies, side by side.")
    q = st.text_area("Question", placeholder="e.g. How does an uploaded track become playable?")
    k = st.slider("top-k chunks", 1, 20, 5)
    retrieval_only = st.toggle("Retrieval only (no LLM)")
    if st.button("Run all three", type="primary", disabled=not q.strip()):
        cols = st.columns(3)
        for col, mode in zip(cols, MODES):
            with col:
                st.subheader(MODES[mode])
                body = {"question": q, "mode": mode, "top_k": k}
                if retrieval_only:
                    res = api("POST", "/query", json=body)
                else:
                    with st.spinner("Asking OpenAI…"):
                        res = api("POST", "/ask", json=body, timeout=300)
                    st.markdown(res["answer"])
                    st.divider()
                show_retrieval(res, mode)

# ---------------------------------------------------------------- 4 · Graph
else:
    st.header("Knowledge graph")
    c1, c2 = st.columns([3, 1])
    entity = c1.text_input("Focus on an entity (optional)", placeholder="e.g. Kafka, Playback, PostgreSQL")
    limit = c2.number_input("Max nodes", 20, 1000, 150, step=10)
    g = api("GET", "/graph", params={"limit": int(limit), **({"entity": entity} if entity.strip() else {})})

    if not g["nodes"]:
        st.info("The graph is empty. On the Documents page, click “Index + build knowledge graph”.")
    else:
        palette = ["#2f43c4", "#0d6e66", "#b3541e", "#8a3ffc", "#c0267a", "#4a7a0a", "#7a5c00", "#1b7fa6"]
        types = sorted({n["type"] for n in g["nodes"]})
        color = {t: palette[i % len(palette)] for i, t in enumerate(types)}
        net = Network(height="680px", width="100%", directed=True, cdn_resources="in_line")
        net.barnes_hut(gravity=-6000, spring_length=140)
        for n in g["nodes"]:
            net.add_node(n["id"], label=n["name"], title=f"{n['name']} ({n['type']})",
                         color=color[n["type"]], size=10 + min(n["degree"], 20))
        for e in g["edges"]:
            net.add_edge(e["source"], e["target"], label=e["relation"],
                         title=f"{e['relation']} · {e['file']} p.{e['page']}", arrows="to")
        components.html(net.generate_html(), height=700, scrolling=False)
        st.caption("Legend: " + " · ".join(f"<span style='color:{color[t]}'>■</span> {t}" for t in types),
                   unsafe_allow_html=True)
        with st.expander(f"Edges table ({len(g['edges'])})"):
            st.dataframe(pd.DataFrame(g["edges"]), hide_index=True, use_container_width=True)
