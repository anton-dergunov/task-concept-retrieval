"""Candidate pools: the ~100 icons a labeller (human or LLM judge) chooses from.

A round-robin union of several matchers' rankings (a diverse pool) plus uniformly
random icons (a control for pool bias), shuffled with a per-task seed so every device
shows the same grid. Every matcher runs on both description versions (v1 and v2;
docs/icon-descriptions.md §7, pool fairness): provenance says which version proposed an
icon ("M3.v2@1"). The first grid has 30 icons; "more" adds 70.

Used by scripts/build_label_bundle.py (the labelling tool) and tcr/judge.py (the LLM
judge, for tasks that are not in a labelling bundle).
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Sequence, Tuple

from .org_tasks import query_text

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


class PoolBuilder:
    """Builds the matchers once; `pool(rec)` gives (first 30, next 70, provenance)."""

    def __init__(self):
        from .data import load_icons
        from .methods import build_method
        self.icons = {ver: load_icons(version=ver) for ver in VERSIONS}
        self.all_names = sorted({ic.name for ver in VERSIONS for ic in self.icons[ver]})
        keys = sorted({k for _, k, _ in SOURCES})
        self.matchers = {(key, ver): build_method(key, self.icons[ver])
                         for key in keys for ver in VERSIONS}

    def pool(self, rec) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        ranked = rankings(rec, self.matchers)
        first, prov = first_grid(ranked, rec, self.all_names)
        more, more_prov = more_grid(ranked, rec, self.all_names, first)
        return first, more, {**prov, **more_prov}
