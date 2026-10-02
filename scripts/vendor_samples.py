#!/usr/bin/env python3
"""Vendor the realistic sample org files into this repo and build the
"Realistic" task dataset (D2) from them.

Source (read-only): agentic-org-planner/samples/realistic/**/*.org
Dest: data/eval/realistic/          raw .org snapshot (committed), mirrors the source
      data/datasets/realistic.jsonl  parsed tasks, shared dataset schema (committed)

The source is generated, non-personal data, so it is copied verbatim. Files the
agenda ignores (init/workspace, Journal/, …) are copied but not parsed.
Run: python scripts/vendor_samples.py [--src <path>]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr.datasets import DATASETS, write_jsonl  # noqa: E402
from tcr.org_tasks import iter_org_files, parse_org_tasks, to_records  # noqa: E402

DEFAULT_SRC = Path("/Users/anton/projects/products/agentic-org-planner/samples/realistic")
DEST_DIR = ROOT / "data" / "eval" / "realistic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")

    # Mirror the source exactly: drop files from an older layout first.
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    org_files = sorted(args.src.rglob("*.org"))
    for f in org_files:
        dest = DEST_DIR / f.relative_to(args.src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

    ds = DATASETS["realistic"]
    tasks = []
    for f in iter_org_files(DEST_DIR):
        tasks.extend(parse_org_tasks(f.read_text(encoding="utf-8")))
    n = write_jsonl(ds.path, to_records(tasks, ds.key, ds.id_prefix))

    print(f"Copied {len(org_files)} org files -> {DEST_DIR}")
    print(f"Parsed {len(tasks)} tasks -> {n} unique -> {ds.path}")


if __name__ == "__main__":
    main()
