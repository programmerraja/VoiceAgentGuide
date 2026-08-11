/* Shared UI helpers: DOM building, theme, toasts, sidebar list, command palette. */
(function (global) {
  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "html") node.innerHTML = value;
      else if (key === "style") Object.assign(node.style, value);
      else if (key.startsWith("on")) node.addEventListener(key.slice(2).toLowerCase(), value);
      else if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value);
    }
    for (const child of [].concat(children)) {
      if (child) node.append(child);
    }
    return node;
  }

  function toast(message, kind = "") {
    const stack = document.getElementById("toasts");
    if (!stack) return;
    const node = el("div", { class: `toast ${kind}`.trim(), text: message });
    stack.append(node);
    setTimeout(() => {
      node.style.opacity = "0";
      node.style.transition = "opacity .2s";
      setTimeout(() => node.remove(), 220);
    }, 2600);
  }

  function initTheme(button) {
    const stored = localStorage.getItem("atlas-theme");
    if (stored) document.documentElement.dataset.theme = stored;
    button?.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("atlas-theme", next);
    });
  }

  function typeDot(typeId) {
    return el("span", { class: "dot", style: { background: Atlas.nodeType(typeId).color } });
  }

  function renderTypeFilters(container, active, onChange) {
    container.replaceChildren(
      ...Atlas.state.graph.nodeTypes.map((type) => {
        const count = Atlas.state.graph.nodes.filter((n) => n.type === type.id).length;
        return el(
          "button",
          {
            class: "chip",
            "aria-pressed": String(active.has(type.id)),
            style: active.has(type.id) ? { color: type.color } : {},
            onclick: () => {
              active.has(type.id) ? active.delete(type.id) : active.add(type.id);
              renderTypeFilters(container, active, onChange);
              onChange();
            },
          },
          [typeDot(type.id), el("span", { text: `${type.label} ${count}` })]
        );
      })
    );
  }

  function renderListMeta(container, { shown, total, filtered, onClear }) {
    if (!container) return;
    container.replaceChildren(
      el("span", { text: filtered ? `${shown} of ${total} nodes` : `${total} nodes` }),
      filtered ? el("button", { text: "Clear", onclick: onClear }) : null
    );
  }

  function renderNodeList(container, nodes, currentId, onSelect) {
    if (!nodes.length) {
      container.replaceChildren(el("div", { class: "sidebar-empty", text: "No nodes match." }));
      return;
    }
    container.replaceChildren(
      ...nodes.map((node) =>
        el(
          "button",
          {
            class: "node-item",
            "aria-current": String(node.id === currentId),
            onclick: () => onSelect(node.id),
          },
          [
            typeDot(node.type),
            el("span", { class: "node-item-body" }, [
              el("div", { class: "node-item-title", text: node.title }),
              el("div", {
                class: "node-item-meta",
                text: `${Atlas.nodeType(node.type).label} · ${Atlas.degree(node.id)} links${
                  node.draft ? " · unfiled" : ""
                }`,
              }),
            ]),
          ]
        )
      )
    );
  }

  /* Command palette: fuzzy-ish search over nodes, keyboard driven. */
  function createPalette(onSelect) {
    let backdrop = null;
    let results = [];
    let cursor = 0;

    function close() {
      backdrop?.remove();
      backdrop = null;
    }

    function paint(listEl) {
      listEl.replaceChildren(
        ...results.slice(0, 40).map((node, i) =>
          el(
            "button",
            {
              class: "palette-item",
              "aria-selected": String(i === cursor),
              onclick: () => {
                close();
                onSelect(node.id);
              },
            },
            [
              typeDot(node.type),
              el("span", { text: node.title }),
              el("small", { text: Atlas.nodeType(node.type).label }),
            ]
          )
        )
      );
      listEl.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
    }

    function open() {
      if (backdrop) return;
      const listEl = el("div", { class: "palette-results" });
      const input = el("input", {
        type: "text",
        placeholder: "Search nodes, tags, notes…",
        "aria-label": "Search",
      });

      const update = () => {
        results = Atlas.search(input.value);
        cursor = 0;
        paint(listEl);
      };

      input.addEventListener("input", update);
      input.addEventListener("keydown", (event) => {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          const max = Math.min(results.length, 40) - 1;
          cursor = event.key === "ArrowDown" ? Math.min(cursor + 1, max) : Math.max(cursor - 1, 0);
          paint(listEl);
        } else if (event.key === "Enter" && results[cursor]) {
          close();
          onSelect(results[cursor].id);
        } else if (event.key === "Escape") {
          close();
        }
      });

      backdrop = el(
        "div",
        {
          class: "palette-backdrop",
          onclick: (event) => {
            if (event.target === backdrop) close();
          },
        },
        [el("div", { class: "palette" }, [input, listEl])]
      );
      document.body.append(backdrop);
      update();
      input.focus();
    }

    document.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        backdrop ? close() : open();
      }
    });

    return { open, close };
  }

  /* Small form modal. fields: [{name, label, type, value, placeholder, required}] */
  function modal({ title, description, fields = [], submitLabel = "Save", onSubmit }) {
    const inputs = new Map();
    let backdrop = null;

    const close = () => {
      document.removeEventListener("keydown", onKey);
      backdrop?.remove();
      backdrop = null;
    };

    function onKey(event) {
      if (event.key === "Escape") close();
    }

    const submit = () => {
      const values = {};
      for (const [name, input] of inputs) values[name] = input.value.trim();
      const missing = fields.find((f) => f.required && !values[f.name]);
      if (missing) {
        toast(`${missing.label} is required.`, "error");
        inputs.get(missing.name).focus();
        return;
      }
      if (onSubmit(values) !== false) close();
    };

    const body = fields.map((field) => {
      const input = el("input", {
        type: field.type || "text",
        value: field.value || "",
        placeholder: field.placeholder || "",
      });
      if (field.type === "color") input.style.height = "38px";
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") submit();
      });
      inputs.set(field.name, input);
      return el("div", { class: "field" }, [el("label", { text: field.label }), input]);
    });

    backdrop = el(
      "div",
      {
        class: "palette-backdrop",
        onclick: (event) => {
          if (event.target === backdrop) close();
        },
      },
      [
        el("div", { class: "modal" }, [
          el("h3", { class: "modal-title", text: title }),
          description ? el("p", { class: "modal-desc", text: description }) : null,
          ...body,
          el("div", { class: "modal-actions" }, [
            el("button", { class: "btn btn-ghost", text: "Cancel", onclick: close }),
            el("button", { class: "btn btn-primary", text: submitLabel, onclick: submit }),
          ]),
        ]),
      ]
    );

    document.body.append(backdrop);
    document.addEventListener("keydown", onKey);
    inputs.values().next().value?.focus();
  }

  global.UI = {
    el,
    toast,
    initTheme,
    typeDot,
    renderTypeFilters,
    renderListMeta,
    renderNodeList,
    createPalette,
    modal,
  };
})(window);
