"""Tiny stdlib web UI for exploring the matcher (no extra dependencies).

`python -m tcr.cli serve` opens a query box + method dropdown; results render the
actual icon glyphs (served from data/icons/) next to names, scores, latency, and
the abstention decision. This is the seed the next-round FastAPI labelling tool
grows from.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import config, search

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>task-concept-retrieval</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #222; }
  h1 { font-size: 1.2rem; }
  form { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin-bottom: 1rem; }
  input[type=text] { flex: 1; min-width: 260px; padding: .6rem; font-size: 1rem; }
  select, button { padding: .6rem; font-size: 1rem; }
  #meta { color: #666; margin: .5rem 0 1rem; }
  .grid { display: flex; flex-wrap: wrap; gap: .6rem; }
  .card { width: 130px; border: 1px solid #ddd; border-radius: 8px; padding: .6rem;
          text-align: center; }
  .card.shown { border-color: #1a7f37; box-shadow: 0 0 0 2px #1a7f3733; }
  .card.abstained { opacity: .55; }
  .card img { width: 64px; height: 64px; }
  .name { font-size: .8rem; word-break: break-all; margin-top: .3rem; }
  .score { font-variant-numeric: tabular-nums; color: #444; font-size: .85rem; }
  .badge { font-size: .7rem; color: #1a7f37; font-weight: 600; }
</style></head><body>
<h1>task → icon search</h1>
<form id="f">
  <input type="text" id="q" placeholder="type a task, e.g. save money for vacation" autofocus>
  <select id="method">__METHODS__</select>
  <input type="number" id="k" value="12" min="1" max="50" style="width:5rem">
  <button type="submit">Search</button>
</form>
<div id="meta"></div>
<div class="grid" id="grid"></div>
<script>
const f = document.getElementById('f');
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value;
  const method = document.getElementById('method').value;
  const k = document.getElementById('k').value;
  const r = await fetch(`/search?q=${encodeURIComponent(q)}&method=${method}&k=${k}`);
  const data = await r.json();
  document.getElementById('meta').textContent =
    `method ${data.method} · ${data.latency_ms} ms · ` +
    (data.shown ? `would show: ${data.decision_icon}` : `would abstain (${data.reason})`);
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const h of data.hits) {
    const div = document.createElement('div');
    div.className = 'card' + (h.shown ? ' shown' : (data.shown ? '' : ''));
    if (!data.shown && h === data.hits[0]) div.className = 'card abstained';
    div.innerHTML =
      `<img src="/icon/${h.name}.png" alt="${h.name}">` +
      `<div class="name">${h.name}</div>` +
      `<div class="score">${h.score.toFixed(3)}</div>` +
      (h.shown ? `<div class="badge">SHOWN</div>` : ``);
    grid.appendChild(div);
  }
});
</script>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code, content_type, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            options = "".join(
                f'<option value="{m}"{" selected" if m == "M3" else ""}>{m}</option>'
                for m in search.available_methods()
            )
            html = _PAGE.replace("__METHODS__", options)
            self._send(200, "text/html; charset=utf-8", html.encode())
            return
        if parsed.path == "/search":
            qs = parse_qs(parsed.query)
            q = (qs.get("q", [""])[0]).strip()
            method = qs.get("method", ["M3"])[0]
            try:
                k = int(qs.get("k", ["12"])[0])
            except ValueError:
                k = 12
            if method not in search.available_methods():
                method = "M3"
            result = search.search(q, method=method, k=k) if q else None
            payload = result.to_dict() if result else {
                "query": "", "method": method, "latency_ms": 0.0,
                "decision_icon": None, "shown": False, "confidence": 0.0,
                "reason": "empty query", "hits": [],
            }
            self._send(200, "application/json", json.dumps(payload).encode())
            return
        if parsed.path.startswith("/icon/"):
            name = parsed.path[len("/icon/"):]
            if name.endswith(".png"):
                name = name[:-4]
            if not _SAFE_NAME.match(name):
                self._send(400, "text/plain", b"bad name")
                return
            png = config.ICON_PNG_DIR / f"{name}.png"
            if not png.exists():
                self._send(404, "text/plain", b"not found")
                return
            self._send(200, "image/png", png.read_bytes())
            return
        self._send(404, "text/plain", b"not found")


def serve(host: str = config.SERVER_HOST, port: int = config.SERVER_PORT,
          method: str = "M3") -> None:
    # Warm the default matcher AND run one query so the encoder model is loaded
    # and its first forward pass is paid now (index cache hits skip model load).
    print(f"Warming method {method} ...", flush=True)
    search.search("warmup query", method=method, k=1)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"Serving on http://{host}:{port}  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        httpd.server_close()
