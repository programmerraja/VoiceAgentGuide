/* Force-directed graph on a canvas. Small enough graphs that O(n^2) repulsion is fine. */
(function (global) {
  function createGraphView(canvas, { onSelect } = {}) {
    const ctx = canvas.getContext("2d");
    const view = { x: 0, y: 0, scale: 1 };
    let nodes = [];
    let links = [];
    let byId = new Map();
    let running = false;
    let frame = null;
    let alpha = 1;
    let hovered = null;
    let dragging = null;
    let panning = null;
    let pointerStart = null;
    let selectedId = null;
    let width = 0;
    let height = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function setData(graphNodes, graphEdges) {
      const previous = byId;
      nodes = graphNodes.map((node, i) => {
        const old = previous.get(node.id);
        const angle = (i / graphNodes.length) * Math.PI * 2;
        return {
          data: node,
          x: old ? old.x : Math.cos(angle) * 220 + (Math.random() - 0.5) * 40,
          y: old ? old.y : Math.sin(angle) * 220 + (Math.random() - 0.5) * 40,
          vx: 0,
          vy: 0,
          degree: 0,
        };
      });
      byId = new Map(nodes.map((n) => [n.data.id, n]));
      links = graphEdges
        .filter((e) => byId.has(e.from) && byId.has(e.to))
        .map((e) => {
          const source = byId.get(e.from);
          const target = byId.get(e.to);
          source.degree++;
          target.degree++;
          return { source, target, data: e };
        });
      alpha = 1;
      start();
    }

    function radiusOf(node) {
      return 6 + Math.min(node.degree, 12) * 1.1;
    }

    function tick() {
      const repulsion = 2600;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let distSq = dx * dx + dy * dy || 0.01;
          if (distSq > 90000) continue;
          const force = repulsion / distSq;
          const dist = Math.sqrt(distSq);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          a.vx -= fx;
          a.vy -= fy;
          b.vx += fx;
          b.vy += fy;
        }
      }

      const targetLength = 90;
      for (const link of links) {
        const dx = link.target.x - link.source.x;
        const dy = link.target.y - link.source.y;
        const dist = Math.hypot(dx, dy) || 0.01;
        const force = (dist - targetLength) * 0.02;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        link.source.vx += fx;
        link.source.vy += fy;
        link.target.vx -= fx;
        link.target.vy -= fy;
      }

      for (const node of nodes) {
        node.vx -= node.x * 0.006;
        node.vy -= node.y * 0.006;
        if (node === dragging) {
          node.vx = 0;
          node.vy = 0;
          continue;
        }
        node.vx *= 0.82;
        node.vy *= 0.82;
        node.x += node.vx * alpha;
        node.y += node.vy * alpha;
      }

      alpha = Math.max(alpha * 0.994, 0.06);
    }

    function css(name) {
      return getComputedStyle(document.body).getPropertyValue(name).trim();
    }

    function draw() {
      const border = css("--color-border-strong") || "#2f3648";
      const textColor = css("--color-ink") || "#e7eaf3";
      const mutedColor = css("--color-faint") || "#6b7488";
      const accent = css("--color-accent") || "#7c9cff";
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.translate(width / 2 + view.x, height / 2 + view.y);
      ctx.scale(view.scale, view.scale);

      const focus = hovered || byId.get(selectedId);
      const connected = new Set();
      if (focus) {
        connected.add(focus.data.id);
        for (const link of links) {
          if (link.source === focus) connected.add(link.target.data.id);
          if (link.target === focus) connected.add(link.source.data.id);
        }
      }

      for (const link of links) {
        const active =
          focus && (connected.has(link.source.data.id) && connected.has(link.target.data.id));
        ctx.strokeStyle = active ? accent : border;
        ctx.globalAlpha = focus ? (active ? 0.9 : 0.15) : 0.55;
        ctx.lineWidth = active ? 1.6 : 1;
        ctx.beginPath();
        ctx.moveTo(link.source.x, link.source.y);
        ctx.lineTo(link.target.x, link.target.y);
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
      for (const node of nodes) {
        const dim = focus && !connected.has(node.data.id);
        const radius = radiusOf(node);
        ctx.globalAlpha = dim ? 0.25 : 1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = Atlas.nodeType(node.data.type).color;
        ctx.fill();
        if (node.data.id === selectedId) {
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = textColor;
          ctx.stroke();
        }

        if (view.scale > 0.55 || node.degree > 4 || node === focus) {
          ctx.font = "500 11px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillStyle = dim ? mutedColor : textColor;
          ctx.fillText(node.data.title, node.x, node.y + radius + 13);
        }
      }

      ctx.restore();
    }

    function loop() {
      tick();
      draw();
      frame = requestAnimationFrame(loop);
    }

    function start() {
      if (running) return;
      running = true;
      loop();
    }

    function stop() {
      running = false;
      if (frame) cancelAnimationFrame(frame);
      frame = null;
    }

    function toWorld(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left - width / 2 - view.x) / view.scale,
        y: (event.clientY - rect.top - height / 2 - view.y) / view.scale,
      };
    }

    function nodeAt(point) {
      for (let i = nodes.length - 1; i >= 0; i--) {
        const node = nodes[i];
        const radius = radiusOf(node) + 5;
        if ((node.x - point.x) ** 2 + (node.y - point.y) ** 2 <= radius * radius) return node;
      }
      return null;
    }

    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      pointerStart = { x: event.clientX, y: event.clientY, moved: false };
      const point = toWorld(event);
      const node = nodeAt(point);
      if (node) {
        dragging = node;
        alpha = Math.max(alpha, 0.5);
      } else {
        panning = { x: event.clientX - view.x, y: event.clientY - view.y };
      }
    });

    canvas.addEventListener("pointermove", (event) => {
      const point = toWorld(event);
      if (pointerStart && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4) {
        pointerStart.moved = true;
      }
      if (dragging) {
        dragging.x = point.x;
        dragging.y = point.y;
        return;
      }
      if (panning) {
        view.x = event.clientX - panning.x;
        view.y = event.clientY - panning.y;
        draw();
        return;
      }
      const next = nodeAt(point);
      if (next !== hovered) {
        hovered = next;
        canvas.style.cursor = next ? "pointer" : "grab";
        draw();
      }
    });

    canvas.addEventListener("pointerup", (event) => {
      const node = nodeAt(toWorld(event));
      if (node && onSelect && !pointerStart?.moved) onSelect(node.data.id);
      pointerStart = null;
      dragging = null;
      panning = null;
    });

    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.1 : 0.9;
        view.scale = Math.min(3, Math.max(0.25, view.scale * factor));
        draw();
      },
      { passive: false }
    );

    new ResizeObserver(resize).observe(canvas);
    resize();

    return {
      setData,
      start,
      stop,
      select(id) {
        selectedId = id;
        draw();
      },
    };
  }

  global.createGraphView = createGraphView;
})(window);
