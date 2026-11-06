#!/usr/bin/env python3
"""Build the self-contained labelling bundle that labeller/server.py serves.

Runs here (needs tcr + its ML deps); the NAS only needs the output + stdlib.

labeller/bundle/  (gitignored — includes private tasks)
  datasets.json          [{key, display}] in registry order
  tasks/<key>.jsonl      {id, title, body, lang, alts, icons[30], more[70], prov{icon: ["M3@1", …]}}
                         alts = the title in the other languages of a translated item
  icons/<name>.png       non-discarded icons only
  search_index.json      {name: description text} for the lexical search box

Candidates per task: a round-robin union of several matchers' rankings (diverse
pool) plus uniformly random icons (a control for pool bias), shuffled with a
per-task seed so every device shows the same grid. Every matcher runs on both
description versions (v1 and v2; docs/icon-descriptions-v2.md §6, pool fairness):
provenance says which version proposed an icon ("M3.v2@1"), so each version's
recall of the labels can be measured; the icon set is the union of both versions'
kept icons. "More icons" extends it to
~100 with the matchers' next-best icons and more random ones. Icons are distinct
glyphs (aliases collapsed in tcr.data). Existing grids are reused on rebuild so
what the labeller saw never changes (--refresh to redo).

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
# (label, matcher, query view) — the pool sources, in round-robin order. "parent"
# prefixes the nearest parent heading, whose topic often matters most.
SOURCES = [("M3", "M3", "title"), ("M1", "M1", "body"), ("M2", "M2", "body"),
           ("B1", "B1", "body"), ("M1p", "M1", "parent")]
N_METHOD = 24       # first grid: icons drawn from the matchers …
N_RANDOM = 6        # … plus uniformly random icons (30 in total)
N_MORE_METHOD = 56  # "More icons": the matchers' next-best icons …
N_MORE_RANDOM = 14  # … plus more random ones (70 more, ~100 in total)
DEPTH = 120         # how deep to read each matcher's ranking
VERSIONS = ("v1", "v2")   # icon description versions; every source runs on each


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


def rankings(rec, matchers):
    """[(provenance label, ranked names)] for every source × version. Which version
    comes first at each depth of the round-robin is drawn per task."""
    title, body, parents = rec["title"], rec.get("body", ""), rec.get("parents") or []
    versions = list(VERSIONS)
    random.Random(_seed("versions", rec["id"])).shuffle(versions)
    out = []
    for label, key, view in SOURCES:
        if view == "parent" and not parents:
            continue   # would duplicate the title-only query
        q = query_text(title, body, with_body=(view == "body"),
                       parents=parents if view == "parent" else ())
        for ver in versions:
            out.append((f"{label}.{ver}", [n for n, _ in matchers[key, ver].rank(q, top_k=DEPTH)]))
    return out


def round_robin(ranked_lists, exclude, n):
    """Interleave the matchers' rankings (rank 1 of each, then rank 2, …), skipping
    `exclude`, until n unique icons. Provenance lists every method rank of each pick."""
    prov, picked = {}, []
    for depth in range(DEPTH):
        for key, ranked in ranked_lists:
            if depth < len(ranked) and ranked[depth] not in exclude:
                prov.setdefault(ranked[depth], []).append(f"{key}@{depth + 1}")
                if ranked[depth] not in picked and len(picked) < n:
                    picked.append(ranked[depth])
    return picked, {n_: prov[n_] for n_ in picked}


def add_random(picked, prov, all_names, exclude, n, rng):
    pool = [n_ for n_ in all_names if n_ not in prov and n_ not in exclude]
    for name in rng.sample(pool, n):
        picked.append(name)
        prov[name] = ["random"]
    rng.shuffle(picked)
    return picked, prov


def first_grid(ranked_lists, rec, all_names):
    picked, prov = round_robin(ranked_lists, set(), N_METHOD)
    return add_random(picked, prov, all_names, set(), N_RANDOM,
                      random.Random(_seed("cands", rec["id"])))


def more_grid(ranked_lists, rec, all_names, first):
    """The matchers' next-best icons after the first grid, plus fresh random ones."""
    shown = set(first)
    picked, prov = round_robin(ranked_lists, shown, N_MORE_METHOD)
    return add_random(picked, prov, all_names, shown, N_MORE_RANDOM,
                      random.Random(_seed("more", rec["id"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", help="comma-separated keys (default: all present)")
    ap.add_argument("--refresh", action="store_true", help="recompute existing candidates")
    args = ap.parse_args()

    dsets = available()
    if args.datasets:
        keys = args.datasets.split(",")
        dsets = [DATASETS[k] for k in keys]

    # Non-discarded, one per distinct glyph, no blank renders; a glyph has the same
    # name in both versions.
    icons = {ver: load_icons(version=ver) for ver in VERSIONS}
    all_names = sorted({ic.name for ver in VERSIONS for ic in icons[ver]})
    names = set(all_names)
    keys = sorted({k for _, k, _ in SOURCES})
    print(f"{len(all_names)} distinct icons (" + ", ".join(f"{v}: {len(icons[v])}" for v in VERSIONS)
          + f"); building matchers {keys} on each version …")
    matchers = {(key, ver): build_method(key, icons[ver]) for key in keys for ver in VERSIONS}

    (BUNDLE / "tasks").mkdir(parents=True, exist_ok=True)
    for ds in dsets:
        out = BUNDLE / "tasks" / f"{ds.key}.jsonl"
        old = {}
        if out.exists() and not args.refresh:
            old = {r["id"]: r for r in read_jsonl(out)}
        rows, reused, t0 = [], 0, time.time()
        for i, rec in enumerate(read_jsonl(ds.path)):
            prev = old.get(rec["id"])
            # A grid is kept as long as all its icons still exist, so what the labeller
            # already saw never changes; "more" is added to old grids when missing.
            if prev and set(prev["icons"]) <= names and set(prev.get("more", [])) <= names \
                    and prev.get("more"):
                icons_, prov, more = prev["icons"], prev["prov"], prev["more"]
                reused += 1
            else:
                ranked = rankings(rec, matchers)
                if prev and set(prev["icons"]) <= names:
                    icons_, prov = prev["icons"], prev["prov"]
                else:
                    icons_, prov = first_grid(ranked, rec, all_names)
                more, more_prov = more_grid(ranked, rec, all_names, icons_)
                prov = {**prov, **more_prov}
            rows.append({"id": rec["id"], "title": rec["title"], "body": rec.get("body", ""),
                         "parents": rec.get("parents") or [], "lang": rec.get("lang", "en"), "alts": alt_titles(rec),
                         "icons": icons_, "more": more, "prov": prov})
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
    # An icon's text is both versions' documents (v2 adds ES/RU examples).
    docs = {}
    for ver in VERSIONS:
        for ic in icons[ver]:
            docs.setdefault(ic.name, []).append(ic.document(include_examples=True))
    index = {name: " ".join(d) for name, d in docs.items()}
    (BUNDLE / "search_index.json").write_text(json.dumps(index, ensure_ascii=False))
    print(f"icons: {len(all_names)} ({copied} copied); search index -> {BUNDLE / 'search_index.json'}")


if __name__ == "__main__":
    main()
