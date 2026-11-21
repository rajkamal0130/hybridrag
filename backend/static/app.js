/* Hybrid RAG Studio - vanilla JS front end for the FastAPI backend (same origin). */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  const MODES = {
    vector: { label: "Vector RAG", desc: "FAISS similarity only" },
    graph: { label: "GraphRAG", desc: "Entities → graph walk" },
    hybrid: { label: "Hybrid RAG", desc: "Vector + graph, RRF fused" },
  };
  const SAMPLES = [
    "Which data stores does the platform use, and what does each hold?",
    "What happens between a creator uploading a track and it being playable?",
    "Which services depend on PostgreSQL?",
    "Which services publish events to Kafka, and who consumes them?",
  ];
  const PALETTE = ["#4f6df5", "#0c9a87", "#e0702b", "#8b45e8", "#d63c7f", "#5c9a12", "#b8860b", "#1f8fbf", "#c0453a", "#6b7a99"];

  // ------------------------------------------------------------------ utils
  async function api(path, opts = {}) {
    const init = { ...opts };
    if (opts.json) {
      init.method = init.method || "POST";
      init.headers = { "content-type": "application/json" };
      init.body = JSON.stringify(opts.json);
    }
    let res;
    try {
      res = await fetch(path, init);
    } catch {
      throw new Error("Backend not reachable. Is uvicorn running on port 8000?");
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
    return data;
  }

  function toast(msg, err = false) {
    const t = document.createElement("div");
    t.className = "toast" + (err ? " err" : "");
    t.textContent = msg;
    $("#toasts").append(t);
    setTimeout(() => t.remove(), err ? 7000 : 3500);
  }

  async function busy(btn, fn) {
    btn.disabled = true; btn.classList.add("busy");
    try { return await fn(); }
    finally { btn.disabled = false; btn.classList.remove("busy"); }
  }

  const pdfUrl = (file, page) => `/pdfs/${encodeURIComponent(file)}${page ? `#page=${page}` : ""}`;
  const srcLink = (file, page) =>
    `<a href="${pdfUrl(file, page)}" target="_blank" title="${esc(file)}">${esc(shortName(file))} · p.${esc(page)}</a>`;
  const shortName = (f) => (f.length > 28 ? f.slice(0, 25) + "…" : f);

  const typeColors = {};
  function typeColor(t) {
    if (!typeColors[t]) {
      let h = 0;
      for (const ch of t) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
      typeColors[t] = PALETTE[h % PALETTE.length];
    }
    return typeColors[t];
  }

  /** Markdown -> sanitized HTML, with [file.pdf, p. N] citations turned into chips. */
  function renderMd(text) {
    let html = DOMPurify.sanitize(marked.parse(text || ""));
    html = html.replace(/\[([^\[\]]*?\.pdf[^\[\]]*?)\]/gi, (whole, inner) => {
      const pairs = [...inner.matchAll(/([^,;\[\]]+?\.pdf)\s*,\s*pp?\.\s*(\d+)/gi)];
      if (!pairs.length) return whole;
      return pairs
        .map(([, f, p]) => {
          f = f.trim();
          return `<a class="cite" href="${pdfUrl(f, p)}" target="_blank" title="${esc(f)}, page ${p}">p.${p}</a>`;
        })
        .join("");
    });
    return html;
  }

  // ------------------------------------------------------------------ theme + nav
  const root = document.documentElement;
  try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch {}
  $("#theme-toggle").onclick = () => {
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch {}
    if (graphState.loaded) renderGraph(graphState.data);
  };

  function show(view) {
    $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
    if (view === "graph" && !graphState.loaded) loadGraph();
    if (view === "docs") loadDocs();
    history.replaceState(null, "", `#${view}`);
  }
  $$(".nav-item").forEach((b) => (b.onclick = () => show(b.dataset.view)));

  // ------------------------------------------------------------------ status
  async function refreshStatus() {
    try {
      const h = await api("/health");
      $("#st-vectors").textContent = h.vectors.toLocaleString();
      $("#st-entities").textContent = h.graph.entities.toLocaleString();
      $("#st-relations").textContent = h.graph.relations.toLocaleString();
      const store = $("#st-store");
      store.textContent = h.graph.store === "neo4j" ? "Neo4j" : "JSON file";
      store.className = "pill " + (h.graph.store === "neo4j" ? "ok" : h.graph.neo4j_error ? "warn" : "");
      $("#st-engine").textContent = h.graph.query_engine === "neo4j" ? "Cypher" : "NetworkX";
      $("#st-model").textContent = h.model;
      $("#st-model").title = h.model;
      const warn = $("#st-warn");
      warn.hidden = !h.graph.neo4j_error;
      warn.textContent = h.graph.neo4j_error ? "Neo4j not connected: " + h.graph.neo4j_error.slice(0, 140) : "";
    } catch (e) {
      $("#st-store").textContent = "offline";
      $("#st-store").className = "pill warn";
    }
  }

  // ------------------------------------------------------------------ evidence rendering
  function entitiesHtml(entities) {
    if (!entities?.length) return "";
    return `<div class="card ev-block">
      <div class="ev-title">Matched entities <span>${entities.length}</span></div>
      <div class="chips">${entities
        .map((e) => `<span class="chip"><i class="dot" style="background:${typeColor(e.type)}"></i>${esc(e.name)}<span class="t">${esc(e.type)}</span><span class="score">${e.score}</span></span>`)
        .join("")}</div></div>`;
  }

  function factsTable(facts) {
    return `<table class="facts">${facts
      .map((f) => `<tr><td>${esc(f.subject)}</td><td class="rel">${esc(f.relation)} →</td><td>${esc(f.object)}</td><td class="src small">${srcLink(f.source, f.page)}</td></tr>`)
      .join("")}</table>`;
  }

  function chunksHtml(chunks) {
    if (!chunks?.length) return `<div class="card ev-block"><div class="ev-title">Retrieved chunks</div><div class="empty small">No chunks retrieved.</div></div>`;
    const max = Math.max(...chunks.map((c) => c.score)) || 1;
    return `<div class="card ev-block">
      <div class="ev-title">Retrieved chunks <span>${chunks.length}</span></div>
      ${chunks
        .map((c, i) => {
          const via = c.via.includes("+") ? "both" : c.via;
          const viaLabel = via === "both" ? "vector + graph" : c.via;
          return `<details class="chunk">
            <summary>
              <span class="rank">${i + 1}</span>
              <span class="chunk-src">${srcLink(c.source, c.page)}</span>
              <span class="chunk-right"><span class="via ${via}">${viaLabel}</span>
                <span class="bar" title="score ${c.score}"><i style="width:${Math.max(6, (c.score / max) * 100)}%"></i></span></span>
            </summary>
            <div class="chunk-text">${esc(c.text)}</div>
          </details>`;
        })
        .join("")}
    </div>`;
  }

  /** Small force graph of the facts used, matched entities highlighted. */
  function miniGraph(el, facts, entities) {
    const matched = new Set((entities || []).map((e) => e.name.toLowerCase()));
    const nodes = new Map();
    const edges = [];
    for (const f of facts) {
      for (const n of [f.subject, f.object]) {
        if (!nodes.has(n)) {
          const hit = matched.has(n.toLowerCase());
          nodes.set(n, {
            id: n, label: n,
            color: { background: hit ? css("--accent") : css("--panel"), border: hit ? css("--accent") : css("--c-graph") },
            font: { color: hit ? "#fff" : css("--text"), size: 12, face: "Inter" },
            shape: "box", margin: 7, borderWidth: 1.5,
          });
        }
      }
      edges.push({ from: f.subject, to: f.object, label: f.relation, title: `${f.relation} · ${f.source} p.${f.page}` });
    }
    new vis.Network(el, { nodes: [...nodes.values()], edges }, {
      physics: { barnesHut: { gravitationalConstant: -3500, springLength: 120 }, stabilization: { iterations: 150 } },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.5 } }, color: { color: css("--border"), highlight: css("--c-graph") },
        font: { size: 10, color: css("--muted"), strokeWidth: 3, strokeColor: css("--panel-2"), face: "Inter" },
        smooth: { type: "continuous" },
      },
      interaction: { hover: true, zoomView: true, tooltipDelay: 120 },
    }).once("stabilizationIterationsDone", function () { this.setOptions({ physics: false }); });
  }

  function evidence(res, { compact = false } = {}) {
    const wrap = document.createElement("div");
    wrap.className = "evidence";
    let html = entitiesHtml(res.entities);
    if (res.facts?.length) {
      html += `<div class="card ev-block">
        <div class="ev-title">Graph facts used <span>${res.facts.length}</span></div>
        ${compact ? "" : `<div class="mini-graph"></div><details style="margin-top:10px"><summary class="muted small" style="cursor:pointer">Show as table</summary>${factsTable(res.facts)}</details>`}
        ${compact ? factsTable(res.facts.slice(0, 12)) + (res.facts.length > 12 ? `<div class="muted small" style="margin-top:6px">+${res.facts.length - 12} more</div>` : "") : ""}
      </div>`;
    }
    html += chunksHtml(res.chunks);
    wrap.innerHTML = html;
    if (!compact && res.facts?.length) requestAnimationFrame(() => miniGraph($(".mini-graph", wrap), res.facts, res.entities));
    return wrap;
  }

  const skeleton = (n = 5) => Array.from({ length: n }, (_, i) => `<div class="skeleton" style="width:${92 - i * 9}%"></div>`).join("");

  // ------------------------------------------------------------------ Ask
  let askMode = "hybrid";
  $$("#ask-mode button").forEach((b) => (b.onclick = () => {
    askMode = b.dataset.mode;
    $$("#ask-mode button").forEach((x) => x.classList.toggle("on", x === b));
  }));
  $("#ask-samples").innerHTML = SAMPLES.map((s) => `<button class="sample">${esc(s)}</button>`).join("");
  $$("#ask-samples .sample").forEach((b) => (b.onclick = () => { $("#ask-q").value = b.textContent; runAsk(); }));

  async function runAsk() {
    const q = $("#ask-q").value.trim();
    if (!q) return $("#ask-q").focus();
    const ronly = $("#ask-ronly").checked;
    const out = $("#ask-out");
    out.innerHTML = `<div class="card answer"><div class="answer-head"><h3>${ronly ? "Retrieving" : "Thinking"}…</h3><span class="mode-tag ${askMode}">${MODES[askMode].label}</span></div>${skeleton()}</div>
      <div class="card ev-block">${skeleton(4)}</div>`;
    const t0 = performance.now();
    await busy($("#ask-go"), async () => {
      try {
        const res = await api(ronly ? "/query" : "/ask", { json: { question: q, mode: askMode, top_k: +$("#ask-k").value || 5 } });
        const ms = Math.round(performance.now() - t0);
        out.innerHTML = "";
        const ans = document.createElement("div");
        ans.className = "card answer";
        ans.innerHTML = `<div class="answer-head"><h3>${ronly ? "Retrieval only" : "Answer"}</h3><span class="mode-tag ${askMode}"><i class="dot ${askMode}"></i>${MODES[askMode].label}</span><span class="meta">${(ms / 1000).toFixed(1)}s · ${res.chunks.length} chunks · ${res.facts.length} facts</span></div>
          ${ronly ? `<div class="muted">No LLM call. The context that would be sent to the model is on the right.</div>` : `<div class="md">${renderMd(res.answer)}</div>`}`;
        out.append(ans, evidence(res));
      } catch (e) {
        out.innerHTML = `<div class="card error-card full">${esc(e.message)}</div>`;
      }
    });
    refreshStatus();
  }
  $("#ask-go").onclick = runAsk;
  $("#ask-q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runAsk(); });

  // ------------------------------------------------------------------ Compare
  async function runCompare() {
    const q = $("#cmp-q").value.trim();
    if (!q) return $("#cmp-q").focus();
    const ronly = $("#cmp-ronly").checked, k = +$("#cmp-k").value || 5;
    const out = $("#cmp-out");
    out.innerHTML = Object.entries(MODES).map(([m, info]) => `
      <div class="card cmp-col" id="cmp-${m}">
        <div class="cmp-head ${m}"><i class="dot ${m}"></i><b>${info.label}</b><span class="meta">${info.desc}</span></div>
        <div class="cmp-body">${skeleton(6)}</div>
      </div>`).join("");
    await busy($("#cmp-go"), () => Promise.all(Object.keys(MODES).map(async (m) => {
      const col = $(`#cmp-${m}`);
      const t0 = performance.now();
      try {
        const res = await api(ronly ? "/query" : "/ask", { json: { question: q, mode: m, top_k: k } });
        const s = ((performance.now() - t0) / 1000).toFixed(1);
        col.innerHTML = `
          <div class="cmp-head ${m}"><i class="dot ${m}"></i><b>${MODES[m].label}</b><span class="meta">${s}s</span></div>
          <div class="cmp-stats"><div><b>${res.entities.length}</b>entities</div><div><b>${res.facts.length}</b>facts</div><div><b>${res.chunks.length}</b>chunks</div></div>
          <div class="cmp-body">${ronly ? `<div class="muted small">Retrieval only.</div>` : `<div class="md">${renderMd(res.answer)}</div>`}</div>
          <div class="cmp-ev"><details ${ronly ? "open" : ""}><summary>Evidence</summary></details></div>`;
        $("details", col).append(evidence(res, { compact: true }));
      } catch (e) {
        $(".cmp-body", col).innerHTML = `<div class="error-card">${esc(e.message)}</div>`;
      }
    })));
  }
  $("#cmp-go").onclick = runCompare;
  $("#cmp-q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runCompare(); });

  // ------------------------------------------------------------------ Graph explorer
  const graphState = { loaded: false, data: null, network: null, nodes: null, edges: null, hidden: new Set() };

  async function loadGraph(entity = "") {
    const canvas = $("#g-canvas");
    canvas.innerHTML = `<div class="graph-loading">Loading graph…</div>`;
    try {
      const params = new URLSearchParams({ limit: $("#g-limit").value });
      if (entity) params.set("entity", entity);
      const data = await api(`/graph?${params}`);
      graphState.loaded = true;
      graphState.data = data;
      renderGraph(data);
    } catch (e) {
      canvas.innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    }
  }

  function renderGraph(data) {
    const canvas = $("#g-canvas");
    canvas.innerHTML = "";
    if (!data.nodes.length) {
      canvas.innerHTML = `<div class="empty">The graph is empty. Go to <b>Documents</b> and run “Index + build knowledge graph”.</div>`;
      $("#g-legend").innerHTML = "";
      return;
    }
    const counts = {};
    data.nodes.forEach((n) => (counts[n.type] = (counts[n.type] || 0) + 1));
    $("#g-legend").innerHTML = Object.entries(counts).sort((a, b) => b[1] - a[1])
      .map(([t, n]) => `<button data-type="${esc(t)}" class="${graphState.hidden.has(t) ? "off" : ""}"><i class="sw" style="background:${typeColor(t)}"></i>${esc(t)}<span class="n">${n}</span></button>`).join("");
    $$("#g-legend button").forEach((b) => (b.onclick = () => {
      const t = b.dataset.type;
      graphState.hidden.has(t) ? graphState.hidden.delete(t) : graphState.hidden.add(t);
      b.classList.toggle("off");
      graphState.nodes.update(data.nodes.filter((n) => n.type === t).map((n) => ({ id: n.id, hidden: graphState.hidden.has(t) })));
    }));

    const labels = $("#g-labels").checked;
    const maxDeg = Math.max(...data.nodes.map((n) => n.degree)) || 1;
    graphState.nodes = new vis.DataSet(data.nodes.map((n) => ({
      id: n.id, label: n.name, title: `${n.name} (${n.type}) · ${n.degree} links`,
      value: n.degree, hidden: graphState.hidden.has(n.type),
      color: { background: typeColor(n.type), border: typeColor(n.type), highlight: { background: css("--accent"), border: css("--accent") } },
      font: { size: 11 + Math.round((n.degree / maxDeg) * 9) },
    })));
    graphState.edges = new vis.DataSet(data.edges.map((e, i) => ({
      id: i, from: e.source, to: e.target, label: labels ? e.relation : undefined,
      title: `${e.relation} · ${e.file} p.${e.page}`, raw: e,
    })));
    const net = new vis.Network(canvas, { nodes: graphState.nodes, edges: graphState.edges }, {
      nodes: { shape: "dot", scaling: { min: 6, max: 30 }, font: { color: css("--text"), face: "Inter", strokeWidth: 3, strokeColor: css("--panel") }, borderWidth: 0 },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.4 } }, width: 0.8,
        color: { color: css("--border"), highlight: css("--accent"), hover: css("--muted") },
        font: { size: 9, color: css("--muted"), strokeWidth: 3, strokeColor: css("--panel"), face: "Inter" },
        smooth: { type: "continuous" },
      },
      physics: { solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -60, springLength: 90, avoidOverlap: 0.4 }, stabilization: { iterations: 250 } },
      interaction: { hover: true, tooltipDelay: 120, navigationButtons: false },
    });
    canvas.insertAdjacentHTML("beforeend", `<div class="graph-loading" id="g-stab">Laying out ${data.nodes.length} nodes…</div>`);
    net.once("stabilizationIterationsDone", () => { net.setOptions({ physics: false }); $("#g-stab")?.remove(); });
    net.on("click", (p) => {
      if (p.nodes.length) sideNode(p.nodes[0]);
      else if (p.edges.length) sideEdge(graphState.edges.get(p.edges[0]).raw);
    });
    graphState.network = net;
  }

  function sideNode(id) {
    const { data } = graphState;
    const n = data.nodes.find((x) => x.id === id);
    if (!n) return;
    const out = data.edges.filter((e) => e.source === id), inc = data.edges.filter((e) => e.target === id);
    const name = (k) => data.nodes.find((x) => x.id === k)?.name || k;
    const rel = (e, other) => `<li><span class="r">${esc(e.relation)}</span> <span class="node-link" data-id="${esc(other)}">${esc(name(other))}</span><span class="s">${srcLink(e.file, e.page)}</span></li>`;
    $("#g-side").innerHTML = `
      <span class="side-type" style="background:${typeColor(n.type)}">${esc(n.type)}</span>
      <h2>${esc(n.name)}</h2>
      <div class="muted small">${n.degree} relations in view</div>
      <div class="side-actions">
        <button class="btn" id="side-focus">Focus neighbourhood</button>
        <button class="btn primary" id="side-ask">Ask about this</button>
      </div>
      ${out.length ? `<div class="side-sec">Outgoing · ${out.length}</div><ul class="rel-list">${out.map((e) => rel(e, e.target)).join("")}</ul>` : ""}
      ${inc.length ? `<div class="side-sec">Incoming · ${inc.length}</div><ul class="rel-list">${inc.map((e) => `<li><span class="node-link" data-id="${esc(e.source)}">${esc(name(e.source))}</span> <span class="r">${esc(e.relation)}</span> → this<span class="s">${srcLink(e.file, e.page)}</span></li>`).join("")}</ul>` : ""}`;
    $$("#g-side .node-link").forEach((a) => (a.onclick = () => { graphState.network.selectNodes([a.dataset.id]); graphState.network.focus(a.dataset.id, { scale: 1.2, animation: true }); sideNode(a.dataset.id); }));
    $("#side-focus").onclick = () => { $("#g-entity").value = n.name; loadGraph(n.name); };
    $("#side-ask").onclick = () => { $("#ask-q").value = `What is ${n.name}, and how does it relate to the other components?`; show("ask"); runAsk(); };
  }

  function sideEdge(e) {
    const name = (k) => graphState.data.nodes.find((x) => x.id === k)?.name || k;
    $("#g-side").innerHTML = `<div class="side-sec" style="margin-top:0">Relation</div>
      <h2>${esc(name(e.source))} <span style="color:var(--c-graph)">${esc(e.relation)}</span> ${esc(name(e.target))}</h2>
      <div class="side-sec">Source</div><div>${srcLink(e.file, e.page)}</div>`;
  }

  $("#g-focus").onclick = () => loadGraph($("#g-entity").value.trim());
  $("#g-entity").addEventListener("keydown", (e) => { if (e.key === "Enter") loadGraph(e.target.value.trim()); });
  $("#g-reset").onclick = () => { $("#g-entity").value = ""; loadGraph(); };
  $("#g-limit").onchange = () => loadGraph($("#g-entity").value.trim());
  $("#g-labels").onchange = () => graphState.data && renderGraph(graphState.data);

  // ------------------------------------------------------------------ Documents
  let selectedPdf = null;
  async function loadDocs() {
    try {
      const d = await api("/pdfs");
      const tb = $("#docs-table tbody");
      if (!d.pdfs.length) {
        tb.innerHTML = `<tr><td colspan="6"><div class="empty small">No PDFs yet. Drop some above.</div></td></tr>`;
        return;
      }
      tb.innerHTML = d.pdfs.map((p) => `
        <tr data-name="${esc(p.name)}" class="${p.name === selectedPdf ? "sel" : ""}">
          <td><b>${esc(p.name)}</b></td><td>${p.size_kb} KB</td><td>${p.chunks}</td>
          <td><span class="badge ${p.vector_indexed ? "yes" : "no"}">${p.vector_indexed ? "Indexed" : "Pending"}</span></td>
          <td><span class="badge ${p.graph_built ? "yes" : "no"}">${p.graph_built ? "Built" : "Pending"}</span></td>
          <td><a href="${pdfUrl(p.name)}" target="_blank" onclick="event.stopPropagation()">Open ↗</a></td>
        </tr>`).join("");
      $$("#docs-table tbody tr[data-name]").forEach((tr) => (tr.onclick = () => preview(tr.dataset.name)));
      if (!selectedPdf) preview(d.pdfs[0].name);
    } catch (e) { toast(e.message, true); }
  }

  function preview(name) {
    selectedPdf = name;
    $$("#docs-table tbody tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.name === name));
    $("#preview-name").textContent = name;
    $("#preview-frame").src = pdfUrl(name);
    const open = $("#preview-open"); open.href = pdfUrl(name); open.hidden = false;
  }

  function logReport(title, r) {
    const g = Object.values(r.graph || {});
    const failedChunks = g.reduce((a, x) => a + (x.failed_chunks || 0), 0);
    $("#docs-log").innerHTML = `<div class="card log">
      <div class="log-head">✓ ${esc(title)}</div>
      <div class="log-stats">
        <span>Added <b>${r.added?.length ?? 0}</b></span><span>Updated <b>${r.updated?.length ?? 0}</b></span>
        <span>Removed <b>${r.removed?.length ?? 0}</b></span><span>Unchanged <b>${r.unchanged?.length ?? 0}</b></span>
        <span>Failed <b>${r.failed?.length ?? 0}</b></span><span>Graph built for <b>${g.length}</b> file(s)${failedChunks ? ` · ${failedChunks} chunk errors` : ""}</span>
        <span>Vectors <b>${r.total_vectors ?? "–"}</b></span>
        ${r.graph_stats ? `<span>Entities <b>${r.graph_stats.entities}</b></span><span>Relations <b>${r.graph_stats.relations}</b></span>` : ""}
      </div>
      <details><summary class="muted small" style="cursor:pointer;margin-top:8px">Raw report</summary><pre>${esc(JSON.stringify(r, null, 2))}</pre></details>
    </div>`;
  }

  async function action(btn, title, path) {
    $("#docs-log").innerHTML = `<div class="card log"><div class="log-head">${esc(title)}… this can take a while</div>${skeleton(2)}</div>`;
    await busy(btn, async () => {
      try {
        const r = await api(path, { method: "POST" });
        logReport(title, r);
        graphState.loaded = false;
        toast(`${title}: done`);
      } catch (e) {
        $("#docs-log").innerHTML = `<div class="card error-card">${esc(e.message)}</div>`;
      }
    });
    loadDocs(); refreshStatus();
  }
  $("#act-vectors").onclick = (e) => action(e.currentTarget, "Index vectors", "/sync?build_graph=false");
  $("#act-graph").onclick = (e) => action(e.currentTarget, "Index + build knowledge graph", "/sync?build_graph=true");
  $("#act-rebuild").onclick = (e) => {
    if (confirm("Clear the whole knowledge graph and re-extract it with OpenAI?")) action(e.currentTarget, "Rebuild graph", "/graph/rebuild");
  };

  async function upload(files) {
    const pdfs = [...files].filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (!pdfs.length) return toast("Only PDF files are accepted", true);
    for (const f of pdfs) {
      const fd = new FormData();
      fd.append("file", f);
      try { await api("/upload", { method: "POST", body: fd }); }
      catch (e) { toast(`${f.name}: ${e.message}`, true); }
    }
    toast(`Saved ${pdfs.length} file(s). Now index them.`);
    loadDocs();
  }
  const dz = $("#dropzone");
  $("#file-input").onchange = (e) => upload(e.target.files);
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => upload(e.dataTransfer.files));

  // ------------------------------------------------------------------ boot
  refreshStatus();
  setInterval(refreshStatus, 30000);
  const start = location.hash.slice(1);
  show(["ask", "compare", "graph", "docs"].includes(start) ? start : "ask");
})();
