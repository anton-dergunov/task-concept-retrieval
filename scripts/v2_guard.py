#!/usr/bin/env python3
"""Contamination guard for the v2 icon descriptions (docs/icon-descriptions.md §4).

Drops every generated task string that equals an eval title (after normalization) or
lies within e5 cosine 0.95 of one, from data/icon_descriptions_v2/*.json (aliases
included): example tasks (EN as `task` and as `parent: task`, ES, RU), context tasks
and near-misses. If eval tasks leaked into training text, later results would be
inflated.

The eval sets include the private Personal list, compared locally only. So the files
record only HOW MANY strings were dropped (`_meta.guard`), never the text or the title
it matched. The guard is idempotent: rerun it after each pass and whenever an eval
dataset changes; a second run drops nothing.

Run: python scripts/v2_guard.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import describe_v2 as dv2  # noqa: E402
from tcr.datasets import available  # noqa: E402

DESC_DIR = config.DATA_DIR / "icon_descriptions_v2"


def candidates(d):
    """[(field, index in that field, text)] for every guarded string of one file."""
    out = []
    for i, e in enumerate(d.get("example_tasks", [])):
        out.append(("example_tasks", i, e["task"]))
        if str(e.get("parent", "")).strip():
            out.append(("example_tasks", i, f"{e['parent']}: {e['task']}"))
    for k in ("example_tasks_es", "example_tasks_ru"):
        out += [(k, i, t) for i, t in enumerate(d.get(k, []))]
    for k in ("context_tasks", "near_misses"):
        out += [(k, i, e["task"]) for i, e in enumerate(d.get(k, []))]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="count, do not rewrite files")
    args = ap.parse_args()

    files = sorted(DESC_DIR.glob("*.json"))
    docs = {p: json.loads(p.read_text(encoding="utf-8")) for p in files}
    texts = sorted({t for d in docs.values() for _, _, t in candidates(d)})
    print(f"{len(files)} files, {len(texts)} distinct strings; checking against eval titles …")
    exact, near = dv2.contamination_hits(texts)
    bad = {texts[i] for i in exact | near}
    print(f"hits: {len(exact)} exact, {len(near)} within cosine {dv2.GUARD_COSINE} "
          f"({len(bad)} distinct strings)")

    eval_sets = [ds.key for ds in available()]
    total, changed = Counter(), 0
    for p, d in docs.items():
        drop = {}
        for field, i, t in candidates(d):
            if t in bad:
                drop.setdefault(field, set()).add(i)
        if not drop:
            continue
        for field, idx in drop.items():
            d[field] = [v for i, v in enumerate(d[field]) if i not in idx]
            total[field] += len(idx)
        g = d["_meta"].get("guard") or {"dropped": {}}
        for field, idx in drop.items():
            g["dropped"][field] = g["dropped"].get(field, 0) + len(idx)
        d["_meta"]["guard"] = g
        changed += 1
    # Every file records the guard that last checked it, dropped or not.
    for p, d in docs.items():
        g = d["_meta"].get("guard") or {"dropped": {}}
        g.update(threshold=dv2.GUARD_COSINE, model=config.MULTILINGUAL_MODEL, eval_sets=eval_sets)
        d["_meta"]["guard"] = g
        if not args.dry_run:
            p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'would drop' if args.dry_run else 'dropped'} {sum(total.values())} strings in "
          f"{changed} files: {dict(total) or 'none'}")


if __name__ == "__main__":
    main()
