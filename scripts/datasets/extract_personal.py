#!/usr/bin/env python3
"""Extract the private personal task dataset (D1) from the org notes.

Writes (both gitignored under data/private/):
  personal.jsonl     one record per task, shared dataset schema, no file names
  style_stats.json   aggregate style statistics only (lengths, markup/link rates),
                     safe to put in generation prompts — no task text in it

Run: python scripts/datasets/extract_personal.py [--src ~/…/notes/org]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr.datasets import DATASETS, write_jsonl  # noqa: E402
from tcr.org_tasks import iter_org_files, parse_org_tasks, to_records  # noqa: E402

DEFAULT_SRC = Path.home() / "Library/CloudStorage/Dropbox/notes/org"
STYLE_STATS = config.PRIVATE_DIR / "style_stats.json"

_FEATURES = {
    "bold": re.compile(r"(?<!\w)\*\S(?:.*?\S)?\*(?!\w)"),
    "italic": re.compile(r"(?<![\w:/])/\S(?:.*?\S)?/(?!\w)"),
    "verbatim": re.compile(r"(?<!\w)=\S(?:.*?\S)?=(?!\w)"),
    "code": re.compile(r"(?<!\w)~\S(?:.*?\S)?~(?!\w)"),
    "obsidian_link": re.compile(r"\[\[obsidian:"),
    "described_link": re.compile(r"\[\[[^\]]+\]\[[^\]]+\]\]"),
    "bare_url": re.compile(r"(?<!\[)https?://"),
    "list": re.compile(r"^\s*(?:[-+]|\d+[.)])\s", re.M),
    "src_block": re.compile(r"^\s*#\+begin_src", re.M | re.I),
}


def _quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    pick = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return {"p10": pick(.1), "p25": pick(.25), "median": pick(.5), "p75": pick(.75),
            "p90": pick(.9), "max": xs[-1], "mean": round(statistics.mean(xs), 1)}


def style_stats(records):
    titles = [r["title"] for r in records]
    bodies = [r["body"] for r in records]
    n = len(records)
    rate = lambda k: round(sum(1 for r in records  # noqa: E731
                               if _FEATURES[k].search(r["title"] + "\n" + r["body"])) / n, 3)
    return {
        "n_tasks": n,
        "title_words": _quantiles([len(t.split()) for t in titles]),
        "body_chars": _quantiles([len(b) for b in bodies]),
        "body_chars_nonempty": _quantiles([len(b) for b in bodies if b]),
        "body_lines_nonempty": _quantiles([b.count("\n") + 1 for b in bodies if b]),
        "share_empty_body": round(sum(1 for b in bodies if not b) / n, 3),
        "feature_rates": {k: rate(k) for k in _FEATURES},
        "lang": dict(Counter(r["lang"] for r in records)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()
    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")

    ds = DATASETS["personal"]
    tasks = []
    files = list(iter_org_files(args.src))
    for f in files:
        tasks.extend(parse_org_tasks(f.read_text(encoding="utf-8")))
    records = to_records(tasks, ds.key, ds.id_prefix)
    n = write_jsonl(ds.path, records)

    stats = style_stats(records)
    STYLE_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    kw = Counter(t.keyword for t in tasks)
    print(f"{len(files)} files, {len(tasks)} tasks {dict(kw)} -> {n} unique -> {ds.path}")
    print(f"style stats -> {STYLE_STATS}")


if __name__ == "__main__":
    main()
