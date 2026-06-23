#!/usr/bin/env python3
"""Vendor the realistic sample org files into this repo for a self-contained,
reproducible eval, and emit a parsed tasks.jsonl.

Source (read-only): productivity-system/samples/realistic/**/*.org
Dest: data/eval/realistic/  (raw .org snapshot, committed)
      data/eval/realistic_tasks.jsonl  (parsed; gitignored, regenerable)

The source is already generated/non-personal data, so it is copied verbatim.
Run: python scripts/vendor_samples.py [--src <path>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr.org_query import iter_tasks  # noqa: E402

DEFAULT_SRC = Path("/Users/anton/projects/products/productivity-system/samples/realistic")
DEST_DIR = ROOT / "data" / "eval" / "realistic"
JSONL = ROOT / "data" / "eval" / "realistic_tasks.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    org_files = sorted(args.src.rglob("*.org"))
    copied = 0
    for f in org_files:
        rel = f.relative_to(args.src)
        dest = DEST_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied += 1

    n_tasks = 0
    with open(JSONL, "w", encoding="utf-8") as out:
        for f in sorted(DEST_DIR.rglob("*.org")):
            text = f.read_text(encoding="utf-8")
            for q in iter_tasks(text):
                rec = {"file": str(f.relative_to(DEST_DIR)),
                       "title": q.title, "tags": q.tags, "text": q.text}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_tasks += 1

    print(f"Copied {copied} org files -> {DEST_DIR}")
    print(f"Parsed {n_tasks} tasks -> {JSONL}")


if __name__ == "__main__":
    main()
