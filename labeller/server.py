#!/usr/bin/env python3
"""Icon labelling server — Python 3.8+, standard library only.

Serves the single-page UI (index.html), the bundle built by
scripts/build_label_bundle.py, and stores labels as an append-only event log,
one file per dataset (labels/<dataset>.jsonl; the latest record per task wins).

Run:  python3 server.py [--host 127.0.0.1] [--port 8766] [--bundle ./bundle]
                        [--labels ./labels] [--token SECRET]
See README.md for NAS / Tailscale deployment.
"""

import argparse
import hashlib
import json
import math
import re
import threading
import time
from collections import Counter
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TOKEN_RE = re.compile(r"[^\W_]+")
STATUSES = {"labelled", "none", "skip", "flagged", "cleared"}
DONE = {"labelled", "none", "skip", "flagged"}  # "cleared" = selection undone
ORDER_SEED = "labeller-order-v1"
SPLIT_SEED = "split-v1"
PWA_FILES = {
    "manifest.webmanifest": ("application/manifest+json", "no-cache"),
    "sw.js": ("text/javascript; charset=utf-8", "no-cache"),
    "app-icon-192.png": ("image/png", "public, max-age=86400"),
    "app-icon-512.png": ("image/png", "public, max-age=86400"),
    "apple-touch-icon.png": ("image/png", "public, max-age=86400"),
}


def read_jsonl(path: Path) -> List[dict]:
    with open(str(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Light English suffix stripping, so "invest" matches investment/investing/invests.
SUFFIXES = ("ments", "ment", "ings", "ing", "ies", "ed", "es", "s")


def stem(tok: str) -> str:
    for suf in SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3 and not tok.endswith("ss"):
            base = tok[: -len(suf)]
            return base + "y" if suf == "ies" else base
    return tok


def split_of(task_id: str) -> str:
    """Same dev/test assignment as tcr.datasets.split_of (kept stdlib-only here)."""
    h = int(hashlib.sha1((SPLIT_SEED + task_id).encode("utf-8")).hexdigest()[:8], 16)
    return "test" if h % 2 == 0 else "dev"


def labelling_order(rows: List[dict]) -> List[dict]:
    """Fixed pseudo-random order that alternates test and dev tasks, so any labelled
    prefix is a uniform sample split evenly between the two; adding tasks later does
    not reshuffle the existing ones' relative order."""
    key = lambda r: hashlib.sha1((ORDER_SEED + r["id"]).encode()).hexdigest()  # noqa: E731
    test = sorted((r for r in rows if split_of(r["id"]) == "test"), key=key)
    dev = sorted((r for r in rows if split_of(r["id"]) == "dev"), key=key)
    out = []
    for i in range(max(len(test), len(dev))):
        out += [x[i] for x in (test, dev) if i < len(x)]
    return out


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in TOKEN_RE.findall(text.casefold())]


class Search:
    """BM25 over icon descriptions, with prefix expansion for search-as-you-type."""

    K1, B = 1.2, 0.75

    def __init__(self, index: Dict[str, str]):
        self.names = list(index)
        self.tf = [Counter(tokenize(index[n])) for n in self.names]
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(t for tf in self.tf for t in tf)
        n = len(self.names)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.vocab = sorted(self.idf)

    PREFIX_WEIGHT = 0.6  # a completion counts less than the word as typed

    def _expand(self, tok: str, prefix: bool) -> List[tuple]:
        """Terms for one query word as (term, weight). Only the word being typed
        (the last one) is prefix-expanded, for search-as-you-type."""
        terms = [(tok, 1.0)] if tok in self.idf else []
        if prefix and len(tok) >= 3:
            # Shorter completions are likelier what the user means ("invest" →
            # "investment" over "investigation").
            terms += [(t, self.PREFIX_WEIGHT * len(tok) / len(t)) for t in self.vocab
                      if t.startswith(tok) and t != tok][:50]
        return terms

    def query(self, q: str, k: int = 60) -> List[str]:
        toks = tokenize(q)
        typing = not q[-1:].isspace()
        groups = [self._expand(t, typing and i == len(toks) - 1) for i, t in enumerate(toks)]
        groups = [g for g in groups if g]
        if not groups:
            return []
        scores = []
        for i, tf in enumerate(self.tf):
            norm = self.K1 * (1 - self.B + self.B * self.len[i] / self.avg)
            s, hit_all = 0.0, True
            for g in groups:
                best = 0.0
                for t, w in g:
                    f = tf.get(t)
                    if f:
                        best = max(best, w * self.idf[t] * f * (self.K1 + 1) / (f + norm))
                if best == 0.0:
                    hit_all = False
                s += best
            if s > 0:
                # Icons matching every query word rank above partial matches.
                scores.append((hit_all, s, self.names[i]))
        scores.sort(reverse=True)
        return [name for _, _, name in scores[:k]]


class Store:
    def __init__(self, bundle: Path, labels_dir: Path):
        self.bundle = bundle
        self.labels_dir = labels_dir
        labels_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.datasets = json.loads((bundle / "datasets.json").read_text(encoding="utf-8"))
        self.tasks = {}   # key -> [task] in labelling order
        self.labels = {}  # key -> {task_id: latest record}
        for d in self.datasets:
            key = d["key"]
            rows = read_jsonl(bundle / "tasks" / (key + ".jsonl"))
            self.tasks[key] = labelling_order(rows)
            latest = {}
            path = labels_dir / (key + ".jsonl")
            if path.exists():
                for rec in read_jsonl(path):
                    latest[rec["task_id"]] = rec
            self.labels[key] = latest
        index = json.loads((bundle / "search_index.json").read_text(encoding="utf-8"))
        self.search = Search(index)

    def summary(self) -> List[dict]:
        out = []
        for d in self.datasets:
            key = d["key"]
            ids = {t["id"] for t in self.tasks[key]}
            done = sum(1 for tid, r in self.labels[key].items() if tid in ids and r["status"] in DONE)
            out.append({"key": key, "display": d["display"], "total": len(ids), "done": done})
        return out

    def first_open(self, key: str) -> int:
        lab = self.labels[key]
        for i, t in enumerate(self.tasks[key]):
            rec = lab.get(t["id"])
            if rec is None or rec["status"] not in DONE:
                return i
        return max(0, len(self.tasks[key]) - 1)

    def task(self, key: str, pos: Optional[int]) -> dict:
        tasks = self.tasks[key]
        if pos is None:
            pos = self.first_open(key)
        pos = max(0, min(pos, len(tasks) - 1))
        t = tasks[pos]
        rec = self.labels[key].get(t["id"])
        s = next(x for x in self.summary() if x["key"] == key)
        return {"dataset": key, "pos": pos, "total": s["total"], "done": s["done"],
                "task": {"id": t["id"], "title": t["title"], "body": t["body"], "lang": t["lang"],
                         "alts": t.get("alts", []), "parents": t.get("parents", [])},
                "icons": t["icons"], "more": t.get("more", []),
                "label": None if rec is None else {"status": rec["status"], "ranking": rec["ranking"],
                                                   "expanded": rec.get("expanded", False)}}

    def save(self, payload: dict) -> dict:
        key = payload["dataset"]
        tid = payload["task_id"]
        task = next((t for t in self.tasks[key] if t["id"] == tid), None)
        if task is None:
            raise KeyError("unknown task")
        status = payload.get("status")
        if status not in STATUSES:
            raise ValueError("bad status")
        ranking = [n for n in payload.get("ranking", []) if isinstance(n, str) and NAME_RE.match(n)]
        expanded = bool(payload.get("expanded"))
        shown = task["icons"] + (task.get("more", []) if expanded else [])
        rec = {
            "task_id": tid,
            "status": status,
            "ranking": ranking,
            "expanded": expanded,                     # "More icons" was opened
            "shown": shown,                           # grid as displayed (shuffled order)
            "prov": {n: task["prov"].get(n, ["search"]) for n in set(ranking) | set(shown)},
            "search_added": [n for n in ranking if n not in shown],
            "queries": payload.get("queries", [])[:50],
            "ms_spent": int(payload.get("ms_spent", 0)),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "client": payload.get("client", "")[:40],
        }
        with self.lock:
            with open(str(self.labels_dir / (key + ".jsonl")), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.labels[key][tid] = rec
        return next(x for x in self.summary() if x["key"] == key)


class Handler(BaseHTTPRequestHandler):
    store = None   # type: Store
    token = None   # type: Optional[str]

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", {"Cache-Control": "no-store"})

    def _authorized(self, qs) -> bool:
        if not self.token:
            return True
        jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        return (jar.get("lt") is not None and jar["lt"].value == self.token) or \
            qs.get("token", [""])[0] == self.token

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        # App shell for installing to the home screen: public (the manifest is fetched
        # without cookies) and free of any task data.
        if url.path.lstrip("/") in PWA_FILES:
            name = url.path.lstrip("/")
            ctype, cache = PWA_FILES[name]
            return self._send(200, (HERE / "pwa" / name).read_bytes(), ctype,
                              {"Cache-Control": cache})
        if not self._authorized(qs):
            return self._send(403, b"forbidden: open /?token=...", "text/plain")
        path = url.path
        try:
            if path in ("/", "/index.html"):
                extra = {"Cache-Control": "no-cache"}
                if self.token and qs.get("token", [""])[0] == self.token:
                    extra["Set-Cookie"] = "lt=%s; Max-Age=31536000; Path=/; HttpOnly; SameSite=Strict" % self.token
                body = (HERE / "index.html").read_bytes()
                return self._send(200, body, "text/html; charset=utf-8", extra)
            if path == "/api/datasets":
                return self._json(self.store.summary())
            if path == "/api/task":
                key = qs.get("dataset", [""])[0]
                if key not in self.store.tasks:
                    return self._json({"error": "unknown dataset"}, 404)
                pos = qs.get("pos", [None])[0]
                return self._json(self.store.task(key, None if pos in (None, "") else int(pos)))
            if path == "/api/search":
                q = qs.get("q", [""])[0][:200]
                return self._json({"icons": self.store.search.query(q)})
            if path.startswith("/icons/") and path.endswith(".png"):
                name = path[len("/icons/"):-len(".png")]
                p = self.store.bundle / "icons" / (name + ".png")
                if not NAME_RE.match(name) or not p.is_file():
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, p.read_bytes(), "image/png",
                                  {"Cache-Control": "public, max-age=31536000, immutable"})
            return self._send(404, b"not found", "text/plain")
        except (ValueError, KeyError) as e:
            return self._json({"error": str(e)}, 400)

    def do_POST(self):
        url = urlparse(self.path)
        if not self._authorized(parse_qs(url.query)):
            return self._send(403, b"forbidden", "text/plain")
        if url.path != "/api/label":
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(min(n, 1 << 20)).decode("utf-8"))
            return self._json(self.store.save(payload))
        except (ValueError, KeyError, TypeError) as e:
            return self._json({"error": str(e)}, 400)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--bundle", type=Path, default=HERE / "bundle")
    ap.add_argument("--labels", type=Path, default=HERE / "labels")
    ap.add_argument("--token", default=None, help="require ?token=… once per device")
    args = ap.parse_args()

    Handler.store = Store(args.bundle, args.labels)
    Handler.token = args.token
    for s in Handler.store.summary():
        print("  %-16s %5d / %d" % (s["display"], s["done"], s["total"]))
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Serving on http://%s:%d/" % (args.host, args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
