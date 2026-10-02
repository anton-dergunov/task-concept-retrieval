#!/usr/bin/env python3
"""Copy label logs pulled from the labelling server into the repo.

Public datasets' labels go to data/labels/<dataset>.jsonl (committed); private
ones to data/private/labels/<dataset>.jsonl (gitignored). Files are append-only
event logs, so the server's copy is the source of truth and simply replaces ours.

The icon review's log (icon_curation.jsonl) goes to data/labels/ as well (it holds
icon names only), and is compared with the v1 descriptions' `discard`.

Run: python scripts/import_labels.py [labeller/labels]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr.data import glyph_hashes, load_icons, removed_by_hand  # noqa: E402
from tcr.datasets import DATASETS  # noqa: E402


def copy_log(path: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
    print(f"{path.name}: {n} events -> {dest}")


def curation_audit() -> None:
    """Human removals against the model's `discard`, one row per distinct glyph."""
    hashes = glyph_hashes()
    removed = {hashes.get(n) for n in removed_by_hand()}
    kept_by_model = {hashes[ic.name] for ic in load_icons()}
    glyphs = {hashes[ic.name] for ic in load_icons(include_discarded=True)}
    cell = {(h, m): 0 for h in (True, False) for m in (True, False)}
    for g in glyphs:
        cell[(g in removed, g not in kept_by_model)] += 1
    both, human_only, model_only = cell[(True, True)], cell[(True, False)], cell[(False, True)]
    print(f"  removed by hand: {both + human_only} of {len(glyphs)} glyphs")
    print(f"  {'':16}{'model discarded':>16}{'model kept':>12}")
    print(f"  {'human removed':16}{both:16d}{human_only:12d}")
    print(f"  {'human kept':16}{model_only:16d}{cell[(False, False)]:12d}")
    if both + model_only and both + human_only:
        print(f"  model discard vs. human: precision {both / (both + model_only):.2f}, "
              f"recall {both / (both + human_only):.2f}")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "labeller" / "labels"
    for path in sorted(src.glob("*.jsonl")):
        if path.name == config.CURATION_PATH.name:
            copy_log(path, config.CURATION_PATH)
            curation_audit()
            continue
        ds = DATASETS.get(path.stem)
        if ds is None:
            print(f"skip {path.name}: unknown dataset")
            continue
        copy_log(path, (config.PRIVATE_DIR if ds.private else config.DATA_DIR) / "labels" / path.name)


if __name__ == "__main__":
    main()
