# Voice Agent Atlas

A connected encyclopedia of voice agents: nodes (domains, concepts, models, providers, tools, metrics, patterns), typed edges between them, and notes on each node.

No database, no framework. One page, one JSON file, and a local server that lets you edit it.
The published site is read-only; the editing UI only appears when that server is running.

## Layout

```
encyclopedia/
  index.html            the whole app (reading and editing)
  server.js             local server: serves the app, writes data/graph.json
  data/graph.json       the whole graph: nodes, edges, notes
  js/graph.js           store, indexing, search
  js/markdown.js        tiny markdown renderer for notes
  js/ui.js              shared UI (theme, toasts, sidebar, ⌘K palette)
  js/graphview.js       canvas force-directed graph
  js/app.js             routing, pages, editing
  css/tailwind.src.css  styles you edit (Tailwind v4)
  css/styles.css        built output, committed so the site needs no build
```

## Run locally

```bash
cd encyclopedia
npm start
```

Open http://localhost:4000. Because the server answers `/api/health`, the app unlocks
**+ New node**, **Edit**, **Types** and **Save**. Opening the same files without the
server (GitHub Pages, `file://`, `python3 -m http.server`) gives the read-only atlas.

Restyling: `npm run watch:css` rebuilds `css/styles.css` from `css/tailwind.src.css`.
Commit the built file.

## Authoring workflow

1. Open a node and press **e** (or click **Edit**). Press **n** anywhere for a new node —
   it is created already linked to the node you were on, and you can change or drop
   that relation in the Connections list.
2. Write notes in markdown, connect it to other nodes.
3. Press **⌘S**. `data/graph.json` is rewritten, with a backup at `data/graph.backup.json`.
4. Commit and push:

```bash
git add encyclopedia/data/graph.json && git commit -m "atlas: add barge-in notes" && git push
```

## Data model

```json
{
  "nodes": [
    { "id": "kokoro-82m", "type": "model", "title": "Kokoro-82M",
      "summary": "one line", "tags": ["tts"], "notes": "markdown" }
  ],
  "edges": [
    { "id": "e-kokoro-tts", "from": "kokoro-82m", "to": "tts",
      "type": "part_of", "note": "optional" }
  ]
}
```

Node and edge types live in the same file under `nodeTypes` and `edgeTypes`, so adding a
new type (say `dataset`) is a two-line change plus a colour.

Edge types are directional and render both ways: `part_of` shows as "part of" on the source
node and "contains" on the target.

## Publishing

Point GitHub Pages at the repo and the atlas is live at `/encyclopedia/`. Nothing needs to be
excluded: without `server.js` answering `/api/health`, the app never shows the editing controls
and has no way to write anything.
