#!/usr/bin/env python3
"""Build the self-contained labelling bundle that labeller/server.py serves.

Runs here (needs tcr + its ML deps); the NAS only needs the output + stdlib.

labeller/bundle/  (gitignored — includes private tasks)
  datasets.json          [{key, display}] in registry order
  tasks/<key>.jsonl      {id, title, body, lang, alts, icons[30], prov{icon: ["M3@1", …]}}
                         alts = the title in the other languages of a translated item
  icons/<name>.png       non-discarded icons only
  search_index.json      {name: description text} for the lexical search box

Candidates per task: a round-robin union of several matchers' rankings (diverse
pool) plus uniformly random icons (a control for pool bias), shuffled with a
per-task seed so every device shows the same grid. Existing candidates are
reused on rebuild so a grid never changes under the labeller (--refresh to redo).

Run: python scripts/build_label_bundle.py [--datasets personal,realistic] [--refresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr.data import load_icons  # noqa: E402
from tcr.datasets import DATASETS, available, read_jsonl, write_jsonl  # noqa: E402
from tcr.methods import build_method  # noqa: E402
from tcr.org_tasks import query_text  # noqa: E402

BUNDLE = ROOT / "labeller" / "bundle"

# (method, query uses body?) — the pool sources, in round-robin order.
SOURCES = [("M3", False), ("M1", True), ("M2", True), ("B1", True)]
N_METHOD = 24   # icons drawn from the matchers
N_RANDOM = 6    # uniformly random icons
DEPTH = 30      # how deep to read each matcher's ranking


def blank_icons(names):
    """Icons whose rendered PNG has no ink (a render failure): unlabelable, so never shown."""
    import numpy as np
    from PIL import Image
    return {n for n in names
            if not (np.asarray(Image.open(config.ICON_PNG_DIR / f"{n}.png").convert("L")) < 128).any()}


def _seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12], 16)


def alt_titles(rec):
    """Translations of the title (meta.parallel), shown together so one label covers all."""
    par = rec.get("meta", {}).get("parallel") or {}
    out = []
    for lang in sorted(par):
        if lang == rec.get("lang"):
            continue
        v = par[lang]
        out.append({"lang": lang, "title": v["title"] if isinstance(v, dict) else v})
    return out


def candidates(rec, matchers, all_names):
    title, body = rec["title"], rec.get("body", "")
    rankings = []
    for key, with_body in SOURCES:
        q = query_text(title, body, with_body=with_body)
        rankings.append((key, [n for n, _ in matchers[key].rank(q, top_k=DEPTH)]))

    prov = {}
    picked = []
    for depth in range(DEPTH):
        for key, ranked in rankings:
            if depth < len(ranked):
                name = ranked[depth]
                prov.setdefault(name, []).append(f"{key}@{depth + 1}")
                if name not in picked and len(picked) < N_METHOD:
                    picked.append(name)
    # prov lists every method that ranked a picked icon within DEPTH (pool analysis).
    prov = {n: prov[n] for n in picked}

    rng = random.Random(_seed("cands", rec["id"]))
    pool = [n for n in all_names if n not in prov]
    for name in rng.sample(pool, N_RANDOM):
        picked.append(name)
        prov[name] = ["random"]
    rng.shuffle(picked)
    return picked, prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", help="comma-separated keys (default: all present)")
    ap.add_argument("--refresh", action="store_true", help="recompute existing candidates")
    args = ap.parse_args()

    dsets = available()
    if args.datasets:
        keys = args.datasets.split(",")
        dsets = [DATASETS[k] for k in keys]

    icons = load_icons()
    blank = blank_icons([ic.name for ic in icons])
    icons = tuple(ic for ic in icons if ic.name not in blank)
    all_names = [ic.name for ic in icons]
    print(f"{len(icons)} non-discarded icons ({len(blank)} blank renders skipped: {sorted(blank)}); "
          f"building matchers {[k for k, _ in SOURCES]} …")
    matchers = {key: build_method(key, icons) for key, _ in SOURCES}

    (BUNDLE / "tasks").mkdir(parents=True, exist_ok=True)
    for ds in dsets:
        out = BUNDLE / "tasks" / f"{ds.key}.jsonl"
        old = {}
        if out.exists() and not args.refresh:
            old = {r["id"]: r for r in read_jsonl(out)}
        rows, reused, t0 = [], 0, time.time()
        for i, rec in enumerate(read_jsonl(ds.path)):
            if rec["id"] in old and not blank & set(old[rec["id"]]["icons"]):
                prev = old[rec["id"]]
                icons_, prov = prev["icons"], prev["prov"]
                reused += 1
            else:
                icons_, prov = candidates(rec, matchers, all_names)
            rows.append({"id": rec["id"], "title": rec["title"], "body": rec.get("body", ""),
                         "lang": rec.get("lang", "en"), "alts": alt_titles(rec),
                         "icons": icons_, "prov": prov})
            if (i + 1) % 250 == 0:
                print(f"  {ds.key}: {i + 1} tasks ({time.time() - t0:.0f}s)")
        write_jsonl(out, rows)
        print(f"{ds.key}: {len(rows)} tasks ({reused} reused) -> {out}")

    all_keys = [d.key for d in DATASETS.values() if (BUNDLE / "tasks" / f"{d.key}.jsonl").exists()]
    (BUNDLE / "datasets.json").write_text(json.dumps(
        [{"key": k, "display": DATASETS[k].display} for k in all_keys], ensure_ascii=False, indent=1))

    icon_dir = BUNDLE / "icons"
    icon_dir.mkdir(exist_ok=True)
    keep = set(all_names)
    for p in icon_dir.glob("*.png"):
        if p.stem not in keep:
            p.unlink()
    copied = 0
    for name in all_names:
        dest = icon_dir / f"{name}.png"
        if not dest.exists():
            shutil.copy2(config.ICON_PNG_DIR / f"{name}.png", dest)
            copied += 1

    # Search matches descriptions only — never icon names (CLAUDE.md hard rule 1).
    index = {ic.name: ic.document(include_examples=True) for ic in icons}
    (BUNDLE / "search_index.json").write_text(json.dumps(index, ensure_ascii=False))
    print(f"icons: {len(all_names)} ({copied} copied); search index -> {BUNDLE / 'search_index.json'}")


if __name__ == "__main__":
    main()
