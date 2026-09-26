#!/usr/bin/env python3
"""Copy label logs pulled from the labelling server into the repo.

Public datasets' labels go to data/labels/<dataset>.jsonl (committed); private
ones to data/private/labels/<dataset>.jsonl (gitignored). Files are append-only
event logs, so the server's copy is the source of truth and simply replaces ours.

Run: python scripts/import_labels.py [labeller/labels]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr.datasets import DATASETS  # noqa: E402


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "labeller" / "labels"
    for path in sorted(src.glob("*.jsonl")):
        ds = DATASETS.get(path.stem)
        if ds is None:
            print(f"skip {path.name}: unknown dataset")
            continue
        dest_dir = (config.PRIVATE_DIR if ds.private else config.DATA_DIR) / "labels"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_dir / path.name)
        n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        print(f"{path.name}: {n} events -> {dest_dir / path.name}")


if __name__ == "__main__":
    main()
