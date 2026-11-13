#!/usr/bin/env python3
"""Public training pools for the LLM judge to label (docs/fine-tuning-plan.md §4, item 2).

  data/train_pools/mslatte.jsonl   MS-LaTTE titles no public dataset uses, cleaned the way
                                   Public-short is, with the list name as parent heading.

Disjoint from every eval set: by MS-LaTTE id (the items Public-short and Public-expanded
use are skipped) and by `contamination_hits` (exact or e5 cosine ≥ 0.95 to any eval
title). Shared dataset schema; `meta.source_id` is the original MS-LaTTE `ID`, so every
line traces back to its source. Licence: data/train_pools/NOTICE.md.

The Personal-style pool is private and built by `scripts/datasets/synth_personal.py pool`.

Run: python scripts/datasets/build_train_pools.py [--n 3000]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcr import config  # noqa: E402
from tcr.datasets import DATASETS, read_jsonl, write_jsonl  # noqa: E402
from tcr.describe_v2 import contamination_hits  # noqa: E402

from build_public_short import _JUNK_TITLES, clean_title, list_parent, location_stratum, source_id  # noqa: E402

OUT_DIR = config.DATA_DIR / "train_pools"
SEED = 20261113


def mslatte_pool(n: int) -> list:
    used = set()
    for key in ("public_short", "public_expanded"):
        for r in read_jsonl(DATASETS[key].path):
            m = r.get("meta") or {}
            if m.get("source") == "mslatte":
                used.add(str(m.get("source_id")))
    recs = json.loads((config.RAW_DIR / "MS-LaTTE.json").read_text(encoding="utf-8"))
    seen, pool = set(), []
    for r in recs:
        title = clean_title(r["TaskTitle"])
        key = title.casefold()
        if (str(r["ID"]) in used or len(title) < 3 or not re.search(r"[a-z]{2}", key)
                or key in _JUNK_TITLES or re.match(r"^\d", title) or key in seen
                or re.search(r"https?://|@", title)):
            continue
        seen.add(key)
        pool.append({"id": source_id("mslatte", r["ID"]), "dataset": "train_mslatte",
                     "title": title, "body": "", "parents": list_parent(r["ListTitle"]),
                     "lang": "en", "meta": {"source": "mslatte", "source_id": r["ID"],
                                            "list": r["ListTitle"],
                                            "place": location_stratum(r)}})
    exact, near = contamination_hits([r["title"] for r in pool])
    clean = [r for i, r in enumerate(pool) if i not in exact and i not in near]
    print(f"MS-LaTTE: {len(recs)} items, {len(used)} used by eval sets; {len(pool)} clean "
          f"unused titles; {len(exact | near)} dropped as contaminated; {len(clean)} left")
    rng = random.Random(SEED)
    return rng.sample(clean, min(n, len(clean)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    rows = mslatte_pool(args.n)
    out = OUT_DIR / "mslatte.jsonl"
    write_jsonl(out, rows)
    print(f"{len(rows)} tasks -> {out} ({sum(bool(r['parents']) for r in rows)} with a parent)")


if __name__ == "__main__":
    main()
