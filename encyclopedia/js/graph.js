/* Shared graph store used by both the public viewer and the local editor. */
(function (global) {
  const state = {
    graph: null,
    nodeById: new Map(),
    nodeTypeById: new Map(),
    edgeTypeById: new Map(),
    edgesByNode: new Map(),
  };

  function reindex() {
    const g = state.graph;
    state.nodeById = new Map(g.nodes.map((n) => [n.id, n]));
    state.nodeTypeById = new Map(g.nodeTypes.map((t) => [t.id, t]));
    state.edgeTypeById = new Map(g.edgeTypes.map((t) => [t.id, t]));
    state.edgesByNode = new Map();
    for (const edge of g.edges) {
      if (!state.nodeById.has(edge.from) || !state.nodeById.has(edge.to)) continue;
      push(state.edgesByNode, edge.from, edge);
      push(state.edgesByNode, edge.to, edge);
    }
  }

  function push(map, key, value) {
    const list = map.get(key);
    if (list) list.push(value);
    else map.set(key, [value]);
  }

  function today() {
    return new Date().toISOString().slice(0, 10);
  }

  /* Older graphs predate per-node dates — fill gaps so the UI never shows blanks. */
  function ensureNodeDates() {
    const fallback = state.graph.updated || today();
    for (const node of state.graph.nodes) {
      if (!node.created) node.created = fallback;
      if (!node.updated) node.updated = node.created;
    }
  }

  function setGraph(data) {
    state.graph = data;
    ensureNodeDates();
    reindex();
    return data;
  }

  async function load(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`);
    return setGraph(await res.json());
  }

  const getNode = (id) => state.nodeById.get(id) || null;
  const nodeType = (id) => state.nodeTypeById.get(id) || { id, label: id, color: "#8a94a6" };
  const edgeType = (id) => state.edgeTypeById.get(id) || { id, label: id, inverse: id };

  /* Connections of a node, with direction resolved so labels read naturally. */
  function connections(nodeId) {
    return (state.edgesByNode.get(nodeId) || []).map((edge) => {
      const outgoing = edge.from === nodeId;
      const type = edgeType(edge.type);
      return {
        edge,
        outgoing,
        label: outgoing ? type.label : type.inverse,
        other: getNode(outgoing ? edge.to : edge.from),
      };
    });
  }

  /* Connections grouped by their directional label, for sectioned rendering. */
  function groupedConnections(nodeId) {
    const groups = new Map();
    for (const c of connections(nodeId)) {
      if (!c.other) continue;
      push(groups, c.label, c);
    }
    return [...groups.entries()];
  }

  function degree(nodeId) {
    return (state.edgesByNode.get(nodeId) || []).length;
  }

  function search(query, types) {
    const q = query.trim().toLowerCase();
    const active = types && types.size ? types : null;
    const results = [];
    for (const node of state.graph.nodes) {
      if (active && !active.has(node.type)) continue;
      if (!q) {
        results.push({ node, score: 0 });
        continue;
      }
      const score = matchScore(node, q);
      if (score > 0) results.push({ node, score });
    }
    results.sort((a, b) => b.score - a.score || a.node.title.localeCompare(b.node.title));
    return results.map((r) => r.node);
  }

  function matchScore(node, q) {
    const title = node.title.toLowerCase();
    if (title === q) return 100;
    if (title.startsWith(q)) return 80;
    if (title.includes(q)) return 60;
    if ((node.tags || []).some((t) => t.toLowerCase().includes(q))) return 40;
    if ((node.summary || "").toLowerCase().includes(q)) return 25;
    if ((node.notes || "").toLowerCase().includes(q)) return 10;
    return 0;
  }

  const STOPWORDS = new Set([
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "by", "at", "as", "it", "this", "that", "from", "into", "than", "then",
    "so", "not", "no", "can", "should", "would", "could", "also", "if", "when", "how", "what",
    "why", "which", "its", "their", "our", "your", "you", "we", "they", "about",
  ]);

  function tokenize(text) {
    return (text || "")
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((w) => w.length > 2 && !STOPWORDS.has(w));
  }

  /* Rough keyword overlap between free text and existing nodes, used to
     suggest links while capturing a new thought. Cheap on purpose. */
  function suggestLinks(text, excludeId, limit = 6) {
    const tokens = new Set(tokenize(text));
    if (!tokens.size) return [];
    const scored = [];
    for (const node of state.graph.nodes) {
      if (node.id === excludeId) continue;
      const nodeTokens = tokenize(`${node.title} ${(node.tags || []).join(" ")} ${node.summary || ""}`);
      let score = 0;
      for (const t of nodeTokens) if (tokens.has(t)) score += 1;
      if (score > 0) scored.push({ node, score });
    }
    scored.sort((a, b) => b.score - a.score || a.node.title.localeCompare(b.node.title));
    return scored.slice(0, limit).map((s) => s.node);
  }

  function slugify(text) {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }

  function uniqueId(base, excludeId) {
    let id = base || "node";
    let n = 2;
    while (state.nodeById.has(id) && id !== excludeId) id = `${base}-${n++}`;
    return id;
  }

  /* Id follows the title. excludeId lets a rename keep its own slug. */
  function idFromTitle(title, excludeId) {
    return uniqueId(slugify(title) || "node", excludeId);
  }

  function serialize() {
    const g = state.graph;
    return JSON.stringify(
      { ...g, updated: today() },
      null,
      2
    ) + "\n";
  }

  global.Atlas = {
    state,
    setGraph,
    load,
    reindex,
    getNode,
    nodeType,
    edgeType,
    connections,
    groupedConnections,
    degree,
    search,
    suggestLinks,
    slugify,
    uniqueId,
    idFromTitle,
    today,
    serialize,
  };
})(window);
