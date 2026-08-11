/* Local-only dev server: serves the atlas and lets the editor save data/graph.json.
   Binds to localhost because it can write to disk. */
const http = require("node:http");
const fs = require("node:fs/promises");
const path = require("node:path");

const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, "data");
const GRAPH_PATH = path.join(DATA_DIR, "graph.json");
const BACKUP_PATH = path.join(DATA_DIR, "graph.backup.json");
const PORT = Number(process.env.PORT) || 4000;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
};

function json(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > 20 * 1024 * 1024) {
        reject(new Error("Payload too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function saveGraph(raw) {
  const graph = JSON.parse(raw);
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    throw new Error("Graph must have nodes[] and edges[]");
  }

  const ids = new Set(graph.nodes.map((n) => n.id));
  if (ids.size !== graph.nodes.length) throw new Error("Duplicate node ids");
  const dangling = graph.edges.filter((e) => !ids.has(e.from) || !ids.has(e.to));
  if (dangling.length) {
    throw new Error(`Edges reference missing nodes: ${dangling.map((e) => e.id).join(", ")}`);
  }

  graph.updated = new Date().toISOString().slice(0, 10);

  await fs.copyFile(GRAPH_PATH, BACKUP_PATH).catch(() => {});
  const tmp = `${GRAPH_PATH}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(graph, null, 2) + "\n", "utf8");
  await fs.rename(tmp, GRAPH_PATH);

  return { nodes: graph.nodes.length, edges: graph.edges.length, updated: graph.updated };
}

async function serveStatic(req, res, pathname) {
  const relative =
    pathname === "/" || pathname === "/edit.html"
      ? "index.html"
      : decodeURIComponent(pathname).replace(/^\/+/, "");
  const target = path.join(ROOT, relative);
  if (!target.startsWith(ROOT)) {
    res.writeHead(403).end("Forbidden");
    return;
  }
  try {
    const body = await fs.readFile(target);
    res.writeHead(200, {
      "content-type": MIME[path.extname(target)] || "application/octet-stream",
      "cache-control": "no-store",
    });
    res.end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
}

const server = http.createServer(async (req, res) => {
  const { pathname } = new URL(req.url, `http://${req.headers.host}`);

  if (pathname === "/api/health") {
    json(res, 200, { ok: true, canSave: true });
    return;
  }

  if (pathname === "/api/graph") {
    if (req.method === "GET") {
      try {
        res.writeHead(200, {
          "content-type": "application/json; charset=utf-8",
          "cache-control": "no-store",
        });
        res.end(await fs.readFile(GRAPH_PATH));
      } catch (error) {
        json(res, 500, { ok: false, error: error.message });
      }
      return;
    }
    if (req.method === "PUT" || req.method === "POST") {
      try {
        const result = await saveGraph(await readBody(req));
        console.log(`saved graph.json — ${result.nodes} nodes, ${result.edges} edges`);
        json(res, 200, { ok: true, ...result });
      } catch (error) {
        json(res, 400, { ok: false, error: error.message });
      }
      return;
    }
    res.writeHead(405).end("Method not allowed");
    return;
  }

  await serveStatic(req, res, pathname);
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Voice Agent Atlas
  open    http://localhost:${PORT}
  editing enabled — the Edit button appears because this server can write
          data/graph.json (backup at data/graph.backup.json)`);
});
