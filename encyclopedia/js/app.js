/* One app for reading and writing the atlas. Editing unlocks only when the
   local server is reachable, so the published site stays read-only. */
(function () {
  const el = UI.el;
  const dom = {
    content: document.getElementById("content"),
    graphWrap: document.getElementById("graphWrap"),
    nodeList: document.getElementById("nodeList"),
    typeFilters: document.getElementById("typeFilters"),
    filterInput: document.getElementById("filterInput"),
    sidebar: document.getElementById("sidebar"),
    sidebarMeta: document.getElementById("sidebarMeta"),
    legend: document.getElementById("graphLegend"),
    main: document.getElementById("main"),
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    brandSub: document.getElementById("brandSub"),
    inboxCount: document.getElementById("inboxCount"),
  };

  const activeTypes = new Set();
  const PALETTE = ["#7c9cff", "#59c2a4", "#e0a33e", "#d8709b", "#9b8cf5", "#6fb7e0", "#c98b62"];

  let canEdit = false;
  let graphView = null;
  let currentId = null;
  let view = "home"; // home | node | edit | types | graph | inbox
  let dirty = false;
  let previewMode = false;
  let graphStale = false;
  let focusTitle = false;
  let baseline = null; // last saved graph JSON — Cancel restores this
  let saveTimer = null;
  let saving = false;
  let saveAgain = false;
  const AUTOSAVE_MS = 700;

  /* ---------- routing ---------- */

  function navigate(id, mode) {
    location.hash = id ? `#/n/${id}${mode === "edit" ? "/edit" : ""}` : "#/";
    dom.sidebar.classList.remove("open");
  }

  function route() {
    const hash = location.hash.replace(/^#\/?/, "");

    if (hash === "graph") {
      view = "graph";
      setGraphVisible(true);
      refreshSidebar();
      return;
    }
    setGraphVisible(false);

    if (hash === "types" && canEdit) {
      view = "types";
      currentId = null;
      renderTypes();
      refreshSidebar();
      return;
    }

    if (hash === "inbox" && canEdit) {
      view = "inbox";
      currentId = null;
      renderInbox();
      refreshSidebar();
      return;
    }

    const match = hash.match(/^n\/([^/]+)(\/edit)?$/);
    if (!match) {
      view = "home";
      currentId = null;
      renderHome();
      refreshSidebar();
      dom.main.scrollTop = 0;
      return;
    }

    currentId = match[1];
    const node = Atlas.getNode(currentId);
    if (!node) {
      view = "node";
      renderMissing(currentId);
    } else if (match[2] && canEdit) {
      view = "edit";
      previewMode = false;
      renderForm(node);
    } else {
      view = "node";
      renderNode(node);
    }
    graphView?.select(currentId);
    refreshSidebar();
    dom.main.scrollTop = 0;
  }

  function setGraphVisible(isGraph) {
    dom.graphWrap.classList.toggle("hidden", !isGraph);
    dom.content.classList.toggle("hidden", isGraph);
    document.getElementById("viewBrowse").setAttribute("aria-pressed", String(!isGraph));
    document.getElementById("viewGraph").setAttribute("aria-pressed", String(isGraph));
    if (!isGraph) {
      graphView?.stop();
      return;
    }
    if (!graphView) {
      graphView = createGraphView(document.getElementById("graphCanvas"), {
        onSelect: (id) => navigate(id),
      });
      graphStale = true;
    }
    if (graphStale) {
      graphStale = false;
      graphView.setData(Atlas.state.graph.nodes, Atlas.state.graph.edges);
      renderLegend();
    }
    graphView.select(currentId);
    graphView.start();
  }

  /* ---------- sidebar ---------- */

  function refreshSidebar() {
    const nodes = Atlas.search(dom.filterInput.value, activeTypes);
    UI.renderNodeList(dom.nodeList, nodes, currentId, (id) => navigate(id));
    UI.renderListMeta(dom.sidebarMeta, {
      shown: nodes.length,
      total: Atlas.state.graph.nodes.length,
      filtered: Boolean(dom.filterInput.value) || activeTypes.size > 0,
      onClear: clearFilters,
    });
    if (dom.inboxCount) {
      const count = draftNodes().length;
      dom.inboxCount.textContent = count ? `(${count})` : "";
    }
  }

  const draftNodes = () => Atlas.state.graph.nodes.filter((n) => n.draft);

  /* Every tag in the atlas, most used first — feeds the tag suggestions. */
  function allTags() {
    const counts = new Map();
    for (const node of Atlas.state.graph.nodes) {
      for (const tag of node.tags || []) counts.set(tag, (counts.get(tag) || 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([tag]) => tag);
  }

  /* Tiny reward so saving/capturing doesn't feel like it vanished into
     the void — the counter briefly pops. */
  function pulse(node) {
    if (!node) return;
    node.classList.remove("pulse");
    requestAnimationFrame(() => {
      node.classList.add("pulse");
      setTimeout(() => node.classList.remove("pulse"), 500);
    });
  }

  function clearFilters() {
    dom.filterInput.value = "";
    activeTypes.clear();
    UI.renderTypeFilters(dom.typeFilters, activeTypes, refreshSidebar);
    refreshSidebar();
  }

  function filterByTag(tag) {
    dom.filterInput.value = tag;
    refreshSidebar();
    dom.sidebar.classList.add("open");
  }

  /* ---------- persistence ---------- */

  async function detectServer() {
    try {
      const res = await fetch("api/health", { cache: "no-store" });
      canEdit = res.ok && (await res.json()).canSave === true;
    } catch {
      canEdit = false;
    }
  }

  function setDirty(value) {
    dirty = value;
    dom.statusDot.classList.toggle("dirty", value);
    if (value) {
      dom.statusText.textContent = "unsaved…";
      scheduleSave();
    } else if (!saving) {
      dom.statusText.textContent = "saved";
    }
  }

  function scheduleSave() {
    if (!canEdit) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      saveTimer = null;
      save({ quiet: true });
    }, AUTOSAVE_MS);
  }

  function captureBaseline() {
    baseline = Atlas.serialize();
  }

  function touch() {
    Atlas.reindex();
    setDirty(true);
    graphStale = true;
  }

  /* Bump a node's updated date whenever its content changes. */
  function touchNode(node) {
    if (node) node.updated = Atlas.today();
    setDirty(true);
  }

  function stampNew(node) {
    const day = Atlas.today();
    node.created = day;
    node.updated = day;
  }

  /* Discard in-memory edits and leave the form. A brand-new unsaved node
     disappears because it was never in the baseline. */
  function discardChanges() {
    if (!dirty && !saving) return true;
    if (!confirm("Discard unsaved changes?")) return false;
    clearTimeout(saveTimer);
    saveTimer = null;
    saveAgain = false;
    if (baseline) Atlas.setGraph(JSON.parse(baseline));
    dirty = false;
    dom.statusDot.classList.remove("dirty");
    dom.statusText.textContent = "saved";
    graphStale = true;
    UI.renderTypeFilters(dom.typeFilters, activeTypes, refreshSidebar);
    refreshSidebar();
    return true;
  }

  function cancelEdit() {
    const leavingId = currentId;
    if (!discardChanges()) return;
    const stillThere = leavingId && Atlas.getNode(leavingId);
    navigate(stillThere ? leavingId : null);
  }

  async function save({ quiet = false } = {}) {
    if (!canEdit) return;
    clearTimeout(saveTimer);
    saveTimer = null;
    if (saving) {
      saveAgain = true;
      return;
    }
    if (!dirty && quiet) return;

    saving = true;
    dom.statusText.textContent = "saving…";
    try {
      const res = await fetch("api/graph", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: Atlas.serialize(),
      });
      const result = await res.json();
      if (!res.ok || !result.ok) throw new Error(result.error || `HTTP ${res.status}`);
      captureBaseline();
      dirty = false;
      dom.statusDot.classList.remove("dirty");
      pulse(dom.sidebarMeta);
      if (!quiet) UI.toast(`Saved ${result.nodes} nodes · ${result.edges} links`, "success");
    } catch (error) {
      dom.statusDot.classList.add("dirty");
      dirty = true;
      UI.toast(`Save failed: ${error.message}`, "error");
    } finally {
      saving = false;
      if (saveAgain || dirty) {
        saveAgain = false;
        if (dirty) {
          dom.statusText.textContent = "unsaved…";
          scheduleSave();
        } else {
          dom.statusText.textContent = "saved";
        }
      } else {
        dom.statusText.textContent = "saved";
      }
    }
  }

  function download() {
    const blob = new Blob([Atlas.serialize()], { type: "application/json" });
    const link = el("a", { href: URL.createObjectURL(blob), download: "graph.json" });
    link.click();
    URL.revokeObjectURL(link.href);
    UI.toast("Downloaded a copy of graph.json.");
  }

  window.addEventListener("beforeunload", (event) => {
    if (!dirty && !saving) return;
    event.preventDefault();
    event.returnValue = "";
  });

  /* ---------- editing ---------- */

  function defaultEdgeType() {
    const types = Atlas.state.graph.edgeTypes;
    return types.find((t) => t.id === "related_to")?.id || types[0].id;
  }

  function edgeId(from, type, to) {
    return `e-${from}-${type}-${to}`.slice(0, 80);
  }

  function addEdge(from, type, to, note) {
    if (from === to) {
      UI.toast("A node cannot link to itself.", "error");
      return false;
    }
    const exists = Atlas.state.graph.edges.some(
      (e) => e.type === type && e.from === from && e.to === to
    );
    if (exists) {
      UI.toast("That connection already exists.", "error");
      return false;
    }
    const edge = { id: edgeId(from, type, to), from, to, type };
    if (note) edge.note = note;
    Atlas.state.graph.edges.push(edge);
    touch();
    return true;
  }

  function removeEdge(id) {
    Atlas.state.graph.edges = Atlas.state.graph.edges.filter((e) => e.id !== id);
    touch();
    renderMain();
  }

  /* New nodes link back to whatever node you were on, so the graph never
     grows orphans. The relation is editable straight away. */
  function createNode() {
    if (!canEdit) return;
    const origin = currentId && Atlas.getNode(currentId) ? Atlas.getNode(currentId) : null;
    const title = "New node";
    const node = {
      id: Atlas.idFromTitle(title),
      type: Atlas.state.graph.nodeTypes[0].id,
      title,
      summary: "",
      tags: [],
      notes: "",
    };
    stampNew(node);
    Atlas.state.graph.nodes.push(node);
    Atlas.reindex();

    if (origin) {
      const type = defaultEdgeType();
      Atlas.state.graph.edges.push({
        id: edgeId(origin.id, type, node.id),
        from: origin.id,
        to: node.id,
        type,
      });
    }
    touch();

    focusTitle = true;
    navigate(node.id, "edit");
    if (origin) UI.toast(`Linked to "${origin.title}" — change the relation below.`);
  }

  /* Builds a node from free text: first line becomes the title, type and
     tags are primed from whatever node you're currently on (or the most
     recent node, as a fallback) so nothing starts out orphaned or bare. */
  function buildCapturedNode(text, linkedIds, draft) {
    const raw = text.trim();
    if (!raw) {
      UI.toast("Write something first.", "error");
      return null;
    }
    const origin = currentId && Atlas.getNode(currentId) ? Atlas.getNode(currentId) : null;
    const title = raw.split("\n")[0].trim().slice(0, 80) || "Untitled spark";
    const seedType =
      origin?.type ||
      Atlas.state.graph.nodes[Atlas.state.graph.nodes.length - 1]?.type ||
      Atlas.state.graph.nodeTypes[0].id;
    const node = {
      id: Atlas.idFromTitle(title),
      type: seedType,
      title,
      summary: "",
      tags: origin ? [...(origin.tags || [])] : [],
      notes: raw,
    };
    stampNew(node);
    if (draft) node.draft = true;
    Atlas.state.graph.nodes.push(node);
    Atlas.reindex();
    const type = defaultEdgeType();
    for (const id of linkedIds) {
      Atlas.state.graph.edges.push({ id: edgeId(id, type, node.id), from: id, to: node.id, type });
    }
    touch();
    return node;
  }

  /* Zero-friction capture: one box, no required fields, drops into the
     Inbox for triage later. Suggests existing nodes to link to as you
     type, and the current node is pre-linked so nothing ends up orphaned. */
  function openQuickCapture() {
    if (!canEdit) return;
    const origin = currentId && Atlas.getNode(currentId) ? Atlas.getNode(currentId) : null;
    const linked = new Map(origin ? [[origin.id, origin]] : []);
    let backdrop = null;

    const close = () => {
      document.removeEventListener("keydown", onKey);
      backdrop?.remove();
      backdrop = null;
    };

    const textarea = el("textarea", {
      rows: 4,
      style: { minHeight: "110px" },
      placeholder: "What's on your mind? The first line becomes the title.",
    });
    const linkedRow = el("div", { class: "chips" });
    const suggestRow = el("div", { class: "chips" });

    function renderLinked() {
      linkedRow.replaceChildren(
        ...[...linked.values()].map((n) =>
          el(
            "button",
            {
              class: "chip",
              type: "button",
              "aria-pressed": "true",
              title: "Click to unlink",
              onclick: () => {
                linked.delete(n.id);
                renderLinked();
                renderSuggestions();
              },
            },
            [UI.typeDot(n.type), el("span", { text: `${n.title} ✕` })]
          )
        )
      );
    }

    function renderSuggestions() {
      const matches = Atlas.suggestLinks(textarea.value, null, 6).filter((n) => !linked.has(n.id));
      suggestRow.replaceChildren(
        ...matches.map((n) =>
          el(
            "button",
            {
              class: "chip",
              type: "button",
              title: "Click to link",
              onclick: () => {
                linked.set(n.id, n);
                renderLinked();
                renderSuggestions();
              },
            },
            [UI.typeDot(n.type), el("span", { text: `+ ${n.title}` })]
          )
        )
      );
    }

    textarea.addEventListener("input", renderSuggestions);

    const captureBtn = el("button", {
      class: "btn btn-primary",
      text: "Capture",
      title: "⌘Enter",
      onclick: () => {
        const node = buildCapturedNode(textarea.value, linked.keys(), true);
        if (!node) return;
        close();
        pulse(dom.sidebarMeta);
        refreshSidebar();
        UI.toast(
          `Captured — atlas now has ${Atlas.state.graph.nodes.length} nodes. File it from Inbox anytime.`,
          "success"
        );
      },
    });

    const fullFormBtn = el("button", {
      class: "btn btn-ghost",
      text: "More fields…",
      onclick: () => {
        const node = buildCapturedNode(textarea.value || "New node", linked.keys(), false);
        if (!node) return;
        close();
        focusTitle = true;
        navigate(node.id, "edit");
      },
    });

    function onKey(event) {
      if (event.key === "Escape") close();
      else if ((event.metaKey || event.ctrlKey) && event.key === "Enter") captureBtn.click();
    }

    renderLinked();
    backdrop = el(
      "div",
      {
        class: "palette-backdrop",
        onclick: (event) => {
          if (event.target === backdrop) close();
        },
      },
      [
        el("div", { class: "modal", style: { width: "min(560px, 92vw)" } }, [
          el("h3", { class: "modal-title", text: "Quick capture" }),
          el("p", {
            class: "modal-desc",
            text: origin
              ? `Linked to "${origin.title}" — click a chip to unlink or link a suggestion.`
              : "No type, tags, or summary needed. Just get it down.",
          }),
          linkedRow,
          textarea,
          suggestRow,
          el("div", { class: "modal-actions" }, [
            fullFormBtn,
            el("button", { class: "btn btn-ghost", text: "Cancel", onclick: close }),
            captureBtn,
          ]),
        ]),
      ]
    );

    document.body.append(backdrop);
    document.addEventListener("keydown", onKey);
    textarea.focus();
  }

  /* ---------- inbox / triage ---------- */

  function renderInbox() {
    const drafts = draftNodes();
    dom.content.replaceChildren(
      el("div", {}, [
        toolbar([
          el("button", { class: "btn btn-ghost", text: "← Atlas", onclick: () => navigate(null) }),
          el("span", { class: "crumb", text: "Inbox" }),
          el("span", { style: { flex: "1" } }),
          el("span", { class: "status-text", text: `${drafts.length} unfiled` }),
        ]),
        el("h2", { class: "section-title", text: "Quick captures waiting to be filed" }),
        drafts.length
          ? el("div", { class: "inbox-list" }, drafts.map(inboxCard))
          : el("p", {
              class: "empty-note",
              text: 'Nothing to triage. Capture something with the "+ New node" button or the "n" key.',
            }),
      ])
    );
  }

  function inboxCard(node) {
    const titleInput = el("input", {
      type: "text",
      value: node.title,
      oninput: (e) => {
        node.title = e.target.value;
        syncIdFromTitle(node);
        touchNode(node);
        refreshSidebar();
        const slugEl = e.target.closest(".inbox-card")?.querySelector("[data-slug]");
        if (slugEl) slugEl.textContent = node.id;
      },
    });

    const typeSelect = el(
      "select",
      {
        onchange: (e) => {
          node.type = e.target.value;
          touchNode(node);
          touch();
          refreshSidebar();
        },
      },
      Atlas.state.graph.nodeTypes.map((t) =>
        el("option", { value: t.id, selected: t.id === node.type, text: t.label })
      )
    );

    const tagsInput = el("input", {
      type: "text",
      value: (node.tags || []).join(", "),
      placeholder: "tags, comma separated",
    });
    const tagSuggestions = el("div", { class: "chips" });

    function commitTags() {
      node.tags = [
        ...new Set(
          tagsInput.value
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean)
        ),
      ];
      tagsInput.value = node.tags.join(", ");
      touchNode(node);
      renderTagSuggestions();
    }

    /* Suggest tags already used elsewhere in the atlas, narrowed by
       whatever fragment is being typed after the last comma. */
    function renderTagSuggestions() {
      const used = new Set(node.tags || []);
      const fragment = tagsInput.value.split(",").pop().trim().toLowerCase();
      const matches = allTags()
        .filter((t) => !used.has(t) && (!fragment || t.includes(fragment)))
        .slice(0, 8);
      tagSuggestions.replaceChildren(
        ...matches.map((tag) =>
          el("button", {
            class: "chip",
            type: "button",
            text: `#${tag}`,
            onclick: () => {
              const parts = tagsInput.value.split(",").map((t) => t.trim());
              const last = (parts[parts.length - 1] || "").toLowerCase();
              if (last && tag.startsWith(last)) parts.pop();
              tagsInput.value = [...parts, tag].filter(Boolean).join(", ");
              commitTags();
            },
          })
        )
      );
    }

    tagsInput.addEventListener("input", renderTagSuggestions);
    tagsInput.addEventListener("change", commitTags);
    renderTagSuggestions();

    const notesArea = el("textarea", {
      rows: 4,
      oninput: (e) => {
        node.notes = e.target.value;
        touchNode(node);
      },
    });
    notesArea.value = node.notes || "";

    const links = Atlas.connections(node.id).filter((c) => c.other);
    const suggestions = Atlas.suggestLinks(`${node.title} ${node.notes}`, node.id, 6).filter(
      (n) => !links.some((l) => l.other.id === n.id)
    );

    return el("div", { class: "card inbox-card" }, [
      el("div", { class: "row" }, [field("Title", titleInput), field("Type", typeSelect)]),
      field("Tags", tagsInput),
      tagSuggestions,
      el("p", { class: "node-meta" }, [
        el("span", { text: "Slug " }),
        el("code", { "data-slug": "1", text: node.id }),
        el("span", {
          text: ` · Added ${node.created || "—"} · Updated ${node.updated || "—"}`,
        }),
      ]),
      field("Notes", notesArea),
      links.length
        ? el(
            "div",
            { class: "chips" },
            links.map((l) =>
              el("span", { class: "chip", "aria-pressed": "true" }, [
                UI.typeDot(l.other.type),
                el("span", { text: `${l.label} ${l.other.title}` }),
              ])
            )
          )
        : el("p", { class: "hint", text: "Not linked to anything yet." }),
      suggestions.length
        ? el(
            "div",
            { class: "chips" },
            suggestions.map((n) =>
              el(
                "button",
                {
                  class: "chip",
                  type: "button",
                  onclick: () => {
                    addEdge(node.id, defaultEdgeType(), n.id);
                    touchNode(node);
                    renderInbox();
                  },
                },
                [UI.typeDot(n.type), el("span", { text: `+ ${n.title}` })]
              )
            )
          )
        : null,
      el("div", { class: "row", style: { justifyContent: "space-between", marginTop: "6px" } }, [
        el("button", {
          class: "btn btn-ghost",
          text: "Open full editor",
          onclick: () => navigate(node.id, "edit"),
        }),
        el("div", { class: "row", style: { gap: "8px" } }, [
          el("button", { class: "btn btn-danger", text: "Discard", onclick: () => discardDraft(node) }),
          el("button", { class: "btn btn-primary", text: "File it ✓", onclick: () => fileDraft(node) }),
        ]),
      ]),
    ]);
  }

  function fileDraft(node) {
    delete node.draft;
    touchNode(node);
    touch();
    pulse(dom.sidebarMeta);
    UI.toast(`Filed "${node.title}".`, "success");
    renderInbox();
    refreshSidebar();
  }

  function discardDraft(node) {
    if (!confirm(`Discard "${node.title}"? This can't be undone.`)) return;
    const graph = Atlas.state.graph;
    graph.nodes = graph.nodes.filter((n) => n.id !== node.id);
    graph.edges = graph.edges.filter((e) => e.from !== node.id && e.to !== node.id);
    touch();
    renderInbox();
    refreshSidebar();
  }

  function deleteNode(node) {
    const links = Atlas.degree(node.id);
    if (!confirm(`Delete "${node.title}"${links ? ` and its ${links} connection(s)` : ""}?`)) return;
    const graph = Atlas.state.graph;
    graph.nodes = graph.nodes.filter((n) => n.id !== node.id);
    graph.edges = graph.edges.filter((e) => e.from !== node.id && e.to !== node.id);
    touch();
    navigate(null);
    UI.toast(`Deleted "${node.title}".`);
  }

  /* Id is always derived from the title — never typed by hand. Edges and
     the URL hash follow along when the slug changes. */
  function syncIdFromTitle(node) {
    const next = Atlas.idFromTitle(node.title, node.id);
    if (next === node.id) return;
    const prev = node.id;
    for (const edge of Atlas.state.graph.edges) {
      if (edge.from === prev) edge.from = next;
      if (edge.to === prev) edge.to = next;
    }
    node.id = next;
    Atlas.reindex();
    graphStale = true;
    if (currentId === prev) {
      currentId = next;
      if (view === "edit") history.replaceState(null, "", `#/n/${next}/edit`);
      else if (view === "node") history.replaceState(null, "", `#/n/${next}`);
    }
  }

  /* ---------- types ---------- */

  const typeUsage = (id) => Atlas.state.graph.nodes.filter((n) => n.type === id).length;
  const edgeTypeUsage = (id) => Atlas.state.graph.edges.filter((e) => e.type === id).length;

  function newNodeType(onDone) {
    UI.modal({
      title: "New node type",
      description: "Types drive the sidebar filters and the graph colours.",
      fields: [
        { name: "label", label: "Label", placeholder: "Dataset", required: true },
        {
          name: "color",
          label: "Colour",
          type: "color",
          value: PALETTE[Atlas.state.graph.nodeTypes.length % PALETTE.length],
        },
      ],
      submitLabel: "Add type",
      onSubmit: ({ label, color }) => {
        const id = Atlas.slugify(label);
        if (Atlas.state.graph.nodeTypes.some((t) => t.id === id)) {
          UI.toast("That type already exists.", "error");
          return false;
        }
        Atlas.state.graph.nodeTypes.push({ id, label, color });
        touch();
        UI.renderTypeFilters(dom.typeFilters, activeTypes, refreshSidebar);
        onDone?.(id);
      },
    });
  }

  function newEdgeType(onDone) {
    UI.modal({
      title: "New relation type",
      description: "Relations are directional. The inverse label reads from the other node.",
      fields: [
        { name: "label", label: "Label (forward)", placeholder: "benchmarked on", required: true },
        { name: "inverse", label: "Inverse label", placeholder: "benchmark for" },
      ],
      submitLabel: "Add relation",
      onSubmit: ({ label, inverse }) => {
        const id = Atlas.slugify(label).replace(/-/g, "_");
        if (Atlas.state.graph.edgeTypes.some((t) => t.id === id)) {
          UI.toast("That relation already exists.", "error");
          return false;
        }
        Atlas.state.graph.edgeTypes.push({ id, label, inverse: inverse || label });
        touch();
        onDone?.(id);
      },
    });
  }

  /* ---------- rendering ---------- */

  function renderMain() {
    if (view === "types") return renderTypes();
    if (view === "inbox") return renderInbox();
    const node = Atlas.getNode(currentId);
    if (view === "edit" && node) return renderForm(node);
    if (view === "node" && node) return renderNode(node);
    return renderHome();
  }

  const toolbar = (children) => el("div", { class: "editor-toolbar" }, children);

  const stat = (value, label) =>
    el("div", { class: "stat" }, [
      el("div", { class: "stat-value", text: String(value) }),
      el("div", { class: "stat-label", text: label }),
    ]);

  function pipelineOrder() {
    const precedes = Atlas.state.graph.edges.filter((e) => e.type === "precedes");
    const targets = new Set(precedes.map((e) => e.to));
    let cursor = precedes.find((e) => !targets.has(e.from))?.from;
    const order = [];
    const seen = new Set();
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      order.push(cursor);
      cursor = precedes.find((e) => e.from === cursor)?.to;
    }
    return order.map(Atlas.getNode).filter(Boolean);
  }

  function cardFor(node) {
    return el("button", { class: "card", onclick: () => navigate(node.id) }, [
      el("div", { class: "badges" }, [
        el("span", { class: "badge" }, [
          UI.typeDot(node.type),
          el("span", { text: Atlas.nodeType(node.type).label }),
        ]),
      ]),
      el("h4", { text: node.title }),
      el("p", { text: node.summary || "" }),
    ]);
  }

  function renderHome() {
    const graph = Atlas.state.graph;
    const domains = graph.nodes.filter((n) => n.type === "domain" && n.id !== "voice-agent");
    const hubs = [...graph.nodes]
      .filter((n) => n.type !== "domain")
      .sort((a, b) => Atlas.degree(b.id) - Atlas.degree(a.id))
      .slice(0, 6);

    dom.content.replaceChildren(
      el("section", { class: "hero" }, [
        el("h1", { text: "Voice Agent Atlas" }),
        el("p", {
          text: "A connected encyclopedia of real-time voice agents: the stages of the pipeline, the models and providers that fill them, and the tradeoffs between them.",
        }),
      ]),
      el("div", { class: "stats" }, [
        stat(graph.nodes.length, "nodes"),
        stat(graph.edges.length, "connections"),
        stat(domains.length, "domains"),
        stat(graph.updated || "—", "updated"),
      ]),
      canEdit
        ? el("div", { class: "row" }, [
            el("button", { class: "btn btn-primary", text: "+ New node", onclick: openQuickCapture }),
            el("button", {
              class: "btn",
              text: "Manage types",
              onclick: () => {
                location.hash = "#/types";
              },
            }),
            el("button", { class: "btn btn-ghost", text: "Download copy", onclick: download }),
          ])
        : null,
      el("h2", { class: "section-title", text: "The pipeline, end to end" }),
      el(
        "div",
        { class: "pipeline" },
        pipelineOrder().map((node, i) =>
          el("button", { class: "pipeline-step", onclick: () => navigate(node.id) }, [
            el("div", { class: "step-index", text: String(i + 1).padStart(2, "0") }),
            el("h3", { text: node.title }),
            el("p", { text: node.summary || "" }),
          ])
        )
      ),
      el("h2", { class: "section-title", text: "Domains" }),
      el("div", { class: "card-grid" }, domains.map(cardFor)),
      el("h2", { class: "section-title", text: "Most connected" }),
      el("div", { class: "card-grid" }, hubs.map(cardFor))
    );
  }

  function renderNode(node) {
    const groups = Atlas.groupedConnections(node.id);

    dom.content.replaceChildren(
      el("article", {}, [
        toolbar([
          el("button", { class: "btn btn-ghost", text: "← Atlas", onclick: () => navigate(null) }),
          el("span", { style: { flex: "1" } }),
          el("button", {
            class: "btn btn-ghost",
            text: "Graph view",
            onclick: () => {
              location.hash = "#/graph";
            },
          }),
          canEdit
            ? el("button", {
                class: "btn",
                text: "Edit",
                title: "e",
                onclick: () => navigate(node.id, "edit"),
              })
            : null,
          canEdit
            ? el("button", {
                class: "btn btn-danger",
                text: "Delete",
                onclick: () => deleteNode(node),
              })
            : null,
        ]),
        el("header", { class: "article-head" }, [
          el("div", { class: "badges" }, [
            el("span", { class: "badge" }, [
              UI.typeDot(node.type),
              el("span", { text: Atlas.nodeType(node.type).label }),
            ]),
            ...(node.tags || []).map((tag) =>
              el("button", {
                class: "tag",
                text: `#${tag}`,
                title: `Filter by ${tag}`,
                onclick: () => filterByTag(tag),
              })
            ),
          ]),
          el("h1", { text: node.title }),
          node.summary ? el("p", { class: "lead", text: node.summary }) : null,
          node.created || node.updated
            ? el("p", {
                class: "node-dates",
                text: [
                  node.created ? `Added ${node.created}` : null,
                  node.updated ? `Updated ${node.updated}` : null,
                ]
                  .filter(Boolean)
                  .join(" · "),
              })
            : null,
        ]),
        node.notes && node.notes.trim()
          ? el("div", { class: "prose", html: Markdown.render(node.notes) })
          : el("p", { class: "empty-note", text: "No notes yet." }),
        groups.length
          ? el("section", { class: "connections" }, [
              el("h2", {
                class: "section-title",
                text: `Connections (${Atlas.degree(node.id)})`,
              }),
              ...groups.map(([label, items]) =>
                el("div", { class: "conn-group" }, [
                  el("div", { class: "conn-label", text: label }),
                  el(
                    "div",
                    { class: "conn-items" },
                    items.map((item) =>
                      el("button", { class: "conn-pill", onclick: () => navigate(item.other.id) }, [
                        UI.typeDot(item.other.type),
                        el("span", { text: item.other.title }),
                      ])
                    )
                  ),
                  ...items
                    .filter((item) => item.edge.note)
                    .map((item) =>
                      el("div", {
                        class: "conn-note",
                        text: `${item.other.title}: ${item.edge.note}`,
                      })
                    ),
                ])
              ),
            ])
          : null,
      ])
    );
  }

  function renderMissing(id) {
    dom.content.replaceChildren(
      el("section", { class: "hero" }, [
        el("h1", { text: "Not found" }),
        el("p", { text: `No node with id "${id}".` }),
      ])
    );
  }

  function renderLegend() {
    dom.legend.replaceChildren(
      ...Atlas.state.graph.nodeTypes.map((type) =>
        el("span", { class: "legend-item" }, [
          el("span", { class: "dot", style: { background: type.color } }),
          el("span", { text: type.label }),
        ])
      )
    );
  }

  /* ---------- editor views ---------- */

  function field(label, control, hint) {
    return el("div", { class: "field" }, [
      el("label", { text: label }),
      control,
      hint ? el("span", { class: "hint", text: hint }) : null,
    ]);
  }

  function renderTypes() {
    const graph = Atlas.state.graph;

    dom.content.replaceChildren(
      toolbar([
        el("button", { class: "btn btn-ghost", text: "← Atlas", onclick: () => navigate(null) }),
        el("span", { class: "crumb", text: "Types" }),
        el("span", { style: { flex: "1" } }),
        el("button", { class: "btn", text: "+ Node type", onclick: () => newNodeType(renderTypes) }),
        el("button", { class: "btn", text: "+ Relation", onclick: () => newEdgeType(renderTypes) }),
      ]),
      el("h2", { class: "section-title", text: `Node types (${graph.nodeTypes.length})` }),
      ...graph.nodeTypes.map((type) => {
        const used = typeUsage(type.id);
        return el("div", { class: "edge-row" }, [
          el("input", {
            type: "color",
            value: type.color,
            style: { width: "44px", height: "32px", padding: "2px", flex: "none" },
            oninput: (e) => {
              type.color = e.target.value;
              setDirty(true);
              UI.renderTypeFilters(dom.typeFilters, activeTypes, refreshSidebar);
              refreshSidebar();
            },
          }),
          el("input", {
            type: "text",
            value: type.label,
            style: { flex: "1 1 160px", width: "auto" },
            oninput: (e) => {
              type.label = e.target.value;
              setDirty(true);
              UI.renderTypeFilters(dom.typeFilters, activeTypes, refreshSidebar);
              refreshSidebar();
            },
          }),
          el("span", { class: "status-text", text: `${type.id} · ${used} nodes` }),
          el("button", {
            class: "btn btn-danger",
            text: "Delete",
            disabled: used > 0,
            title: used > 0 ? "In use — reassign those nodes first" : "",
            onclick: () => {
              graph.nodeTypes = graph.nodeTypes.filter((t) => t.id !== type.id);
              touch();
              UI.renderTypeFilters(dom.typeFilters, activeTypes, refreshSidebar);
              renderTypes();
            },
          }),
        ]);
      }),
      el("h2", { class: "section-title", text: `Relation types (${graph.edgeTypes.length})` }),
      ...graph.edgeTypes.map((type) => {
        const used = edgeTypeUsage(type.id);
        return el("div", { class: "edge-row" }, [
          el("input", {
            type: "text",
            value: type.label,
            placeholder: "forward label",
            style: { flex: "1 1 140px", width: "auto" },
            oninput: (e) => {
              type.label = e.target.value;
              setDirty(true);
            },
          }),
          el("span", { class: "status-text", text: "↔" }),
          el("input", {
            type: "text",
            value: type.inverse || "",
            placeholder: "inverse label",
            style: { flex: "1 1 140px", width: "auto" },
            oninput: (e) => {
              type.inverse = e.target.value;
              setDirty(true);
            },
          }),
          el("span", { class: "status-text", text: `${used} links` }),
          el("button", {
            class: "btn btn-danger",
            text: "Delete",
            disabled: used > 0,
            title: used > 0 ? "In use — remove those links first" : "",
            onclick: () => {
              graph.edgeTypes = graph.edgeTypes.filter((t) => t.id !== type.id);
              touch();
              renderTypes();
            },
          }),
        ]);
      })
    );
  }

  function renderForm(node) {
    const titleInput = el("input", {
      type: "text",
      id: "titleInput",
      value: node.title,
      oninput: (e) => {
        node.title = e.target.value;
        syncIdFromTitle(node);
        touchNode(node);
        refreshSidebar();
        const crumb = dom.content.querySelector(".crumb");
        if (crumb) crumb.textContent = node.title || "Untitled";
        const slugEl = dom.content.querySelector("[data-slug]");
        if (slugEl) slugEl.textContent = node.id;
      },
    });

    const typeSelect = el(
      "select",
      {
        onchange: (e) => {
          if (e.target.value === "__new__") {
            e.target.value = node.type;
            newNodeType((id) => {
              node.type = id;
              touchNode(node);
              touch();
              refreshSidebar();
              renderForm(node);
            });
            return;
          }
          node.type = e.target.value;
          touchNode(node);
          touch();
          refreshSidebar();
          renderForm(node);
        },
      },
      [
        ...Atlas.state.graph.nodeTypes.map((type) =>
          el("option", { value: type.id, selected: type.id === node.type, text: type.label })
        ),
        el("option", { value: "__new__", text: "+ New type…" }),
      ]
    );

    const summaryInput = el("input", {
      type: "text",
      value: node.summary || "",
      placeholder: "One line a stranger would understand",
      oninput: (e) => {
        node.summary = e.target.value;
        touchNode(node);
      },
    });

    const tagsInput = el("input", {
      type: "text",
      value: (node.tags || []).join(", "),
      placeholder: "latency, tts",
      onchange: (e) => {
        node.tags = e.target.value
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean);
        touchNode(node);
      },
    });

    const notesArea = el("textarea", {
      rows: 16,
      placeholder: "Markdown notes: what it is, why it matters, what you measured…",
      oninput: (e) => {
        node.notes = e.target.value;
        touchNode(node);
      },
    });
    notesArea.value = node.notes || "";

    const notesBody = previewMode
      ? el("div", {
          class: "notes-preview prose",
          html: Markdown.render(node.notes) || "<p class='empty-note'>Nothing to preview.</p>",
        })
      : notesArea;

    dom.content.replaceChildren(
      el("div", { class: "editor-form" }, [
        toolbar([
          el("button", {
            class: "btn btn-ghost",
            text: "← Atlas",
            onclick: () => {
              if (!discardChanges()) return;
              navigate(null);
            },
          }),
          UI.typeDot(node.type),
          el("span", { class: "crumb", text: node.title || "Untitled" }),
          el("span", { style: { flex: "1" } }),
          el("button", {
            class: "btn btn-ghost",
            text: "Cancel",
            title: "Discard changes (Esc)",
            onclick: cancelEdit,
          }),
        ]),
        el("div", { class: "row" }, [field("Title", titleInput), field("Type", typeSelect)]),
        field("Tags", tagsInput, "Comma separated."),
        el("p", { class: "node-meta" }, [
          el("span", { text: "Slug " }),
          el("code", { "data-slug": "1", text: node.id }),
          el("span", {
            text: ` · Added ${node.created || "—"} · Updated ${node.updated || "—"}`,
          }),
        ]),
        field("Summary", summaryInput),
        el("div", { class: "field" }, [
          el("div", { class: "notes-toolbar" }, [
            el("span", { class: "field-label", text: "Notes (markdown)" }),
            el("button", {
              class: "btn btn-ghost",
              text: previewMode ? "Edit" : "Preview",
              onclick: () => {
                previewMode = !previewMode;
                renderForm(node);
              },
            }),
          ]),
          notesBody,
        ]),
        renderConnections(node),
        el("div", { class: "row", style: { justifyContent: "space-between", marginTop: "10px" } }, [
          el("button", {
            class: "btn btn-ghost",
            text: "+ New node linked to this one",
            onclick: createNode,
          }),
          el("button", {
            class: "btn btn-danger",
            text: "Delete node",
            onclick: () => deleteNode(node),
          }),
        ]),
      ])
    );

    if (focusTitle) {
      focusTitle = false;
      titleInput.focus();
      titleInput.select();
    }
  }

  /* Relation picker listing both directions, so the reading stays natural
     and a wrong-way link can be flipped in one go. */
  function relationSelect(node, item) {
    const options = [];
    for (const type of Atlas.state.graph.edgeTypes) {
      options.push(
        el("option", {
          value: `${type.id}|out`,
          selected: type.id === item.edge.type && item.outgoing,
          text: type.label,
        })
      );
      if (type.inverse && type.inverse !== type.label) {
        options.push(
          el("option", {
            value: `${type.id}|in`,
            selected: type.id === item.edge.type && !item.outgoing,
            text: type.inverse,
          })
        );
      }
    }
    options.push(el("option", { value: "__new__", text: "+ New relation…" }));

    return el(
      "select",
      {
        style: { flex: "0 1 160px", width: "auto" },
        onchange: (e) => {
          if (e.target.value === "__new__") {
            newEdgeType(() => renderForm(node));
            return;
          }
          const [type, direction] = e.target.value.split("|");
          const other = item.other.id;
          item.edge.type = type;
          item.edge.from = direction === "out" ? node.id : other;
          item.edge.to = direction === "out" ? other : node.id;
          item.edge.id = edgeId(item.edge.from, type, item.edge.to);
          touch();
          renderForm(node);
        },
      },
      options
    );
  }

  function renderConnections(node) {
    const items = Atlas.connections(node.id).filter((c) => c.other);
    return el("section", {}, [
      el("h2", { class: "section-title", text: `Connections (${items.length})` }),
      ...(items.length
        ? items.map((item) =>
            el("div", { class: "edge-row" }, [
              relationSelect(node, item),
              el("span", { class: "edge-text" }, [
                UI.typeDot(item.other.type),
                el("a", {
                  href: `#/n/${item.other.id}/edit`,
                  text: item.other.title,
                }),
              ]),
              el("input", {
                type: "text",
                value: item.edge.note || "",
                placeholder: "note about this link",
                style: { flex: "1 1 160px", width: "auto" },
                oninput: (e) => {
                  const value = e.target.value.trim();
                  if (value) item.edge.note = value;
                  else delete item.edge.note;
                  touchNode(node);
                },
              }),
              el("button", {
                class: "btn btn-danger",
                text: "Remove",
                onclick: () => removeEdge(item.edge.id),
              }),
            ])
          )
        : [el("p", { class: "empty-note", text: "Not connected to anything yet." })]),
      renderEdgeBuilder(node),
    ]);
  }

  function renderEdgeBuilder(node) {
    let targetId = null;

    const typeSelect = el(
      "select",
      {
        onchange: (e) => {
          if (e.target.value !== "__new__") return;
          e.target.value = defaultEdgeType();
          newEdgeType(() => renderForm(node));
        },
      },
      [
        ...Atlas.state.graph.edgeTypes.map((type) =>
          el("option", { value: type.id, text: type.label })
        ),
        el("option", { value: "__new__", text: "+ New relation…" }),
      ]
    );

    const noteInput = el("input", { type: "text", placeholder: "optional note" });
    const menu = el("div", { class: "combo-menu", hidden: true });
    const targetInput = el("input", {
      type: "text",
      placeholder: "search a node…",
      autocomplete: "off",
    });

    const closeMenu = () => {
      menu.hidden = true;
    };

    function openMenu() {
      const matches = Atlas.search(targetInput.value)
        .filter((n) => n.id !== node.id)
        .slice(0, 12);
      menu.replaceChildren(
        ...(matches.length
          ? matches.map((match) =>
              el(
                "button",
                {
                  class: "combo-option",
                  type: "button",
                  onclick: () => {
                    targetId = match.id;
                    targetInput.value = match.title;
                    closeMenu();
                  },
                },
                [
                  UI.typeDot(match.type),
                  el("span", { text: match.title }),
                  el("small", { text: Atlas.nodeType(match.type).label }),
                ]
              )
            )
          : [el("div", { class: "sidebar-empty", text: "No match." })])
      );
      menu.hidden = false;
    }

    const connect = () => {
      if (!targetId) {
        UI.toast("Pick a target node from the list.", "error");
        targetInput.focus();
        return;
      }
      if (addEdge(node.id, typeSelect.value, targetId, noteInput.value.trim())) renderForm(node);
    };

    targetInput.addEventListener("focus", openMenu);
    targetInput.addEventListener("input", () => {
      targetId = null;
      openMenu();
    });
    targetInput.addEventListener("blur", () => setTimeout(closeMenu, 150));
    targetInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && targetId) connect();
    });

    return el("div", { class: "edge-builder" }, [
      el("div", { class: "field", style: { flex: "0 0 auto" } }, [
        el("span", { class: "field-label", text: "From" }),
        el("span", { text: node.title, style: { padding: "9px 0", fontWeight: "550" } }),
      ]),
      el("div", { class: "field" }, [el("span", { class: "field-label", text: "Relation" }), typeSelect]),
      el("div", { class: "field combo" }, [
        el("span", { class: "field-label", text: "To" }),
        targetInput,
        menu,
      ]),
      el("div", { class: "field" }, [el("span", { class: "field-label", text: "Note" }), noteInput]),
      el("button", { class: "btn btn-primary", text: "Connect", onclick: connect }),
    ]);
  }

  /* ---------- boot ---------- */

  async function boot() {
    UI.initTheme(document.getElementById("themeBtn"));
    await detectServer();

    try {
      await Atlas.load(canEdit ? "api/graph" : "data/graph.json");
    } catch (error) {
      dom.content.replaceChildren(
        el("div", { class: "callout" }, [
          el("strong", { text: "Could not load the graph. " }),
          el("span", {
            text: `${error.message}. Run "npm start" in encyclopedia/ and open the address it prints.`,
          }),
        ])
      );
      return;
    }

    if (canEdit) {
      for (const node of document.querySelectorAll(".edit-only")) node.classList.remove("hidden");
      dom.brandSub.textContent = "local · editing";
      captureBaseline();
      setDirty(false);
    }

    UI.renderTypeFilters(dom.typeFilters, activeTypes, refreshSidebar);
    dom.filterInput.addEventListener("input", refreshSidebar);

    const palette = UI.createPalette((id) => navigate(id));
    document.getElementById("searchTrigger").addEventListener("click", palette.open);
    document.getElementById("viewBrowse").addEventListener("click", () => {
      location.hash = currentId ? `#/n/${currentId}` : "#/";
      route();
    });
    document.getElementById("viewGraph").addEventListener("click", () => {
      location.hash = "#/graph";
    });
    document.getElementById("menuBtn").addEventListener("click", () => {
      dom.sidebar.classList.toggle("open");
    });
    document.getElementById("newNodeBtn").addEventListener("click", openQuickCapture);
    document.getElementById("inboxBtn").addEventListener("click", () => {
      location.hash = "#/inbox";
    });
    document.getElementById("typesBtn").addEventListener("click", () => {
      location.hash = "#/types";
    });

    document.addEventListener("keydown", (event) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
      if (canEdit && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        save({ quiet: false });
        return;
      }
      if (typing || event.metaKey || event.ctrlKey) return;
      if (event.key === "Escape") {
        if (view === "edit") cancelEdit();
        else if (location.hash.replace(/^#\/?/, "")) navigate(null);
      } else if (event.key === "e" && canEdit && view === "node" && currentId) {
        event.preventDefault();
        navigate(currentId, "edit");
      } else if (event.key === "n" && canEdit) {
        event.preventDefault();
        openQuickCapture();
      }
    });

    window.addEventListener("hashchange", route);
    route();
  }

  boot();
})();
