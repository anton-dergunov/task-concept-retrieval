#!/usr/bin/env python3
"""Copy label logs pulled from the labelling server into the repo.

Public datasets' labels go to data/labels/<dataset>.jsonl (committed); private
ones to data/private/labels/<dataset>.jsonl (gitignored). Files are append-only
event logs, so the server's copy is the source of truth and simply replaces ours.

The icon review's log (icon_curation.jsonl) goes to data/labels/ as well (it holds
icon names only), and its levels are compared with the v1 descriptions' `discard`
and `icon_usefulness`.

Run: python scripts/import_labels.py [labeller/labels]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr.data import CURATION_LABELS, glyph_hashes, icon_labels, load_icons  # noqa: E402
from tcr.datasets import DATASETS  # noqa: E402


def copy_log(path: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
    print(f"{path.name}: {n} events -> {dest}")


def curation_audit() -> None:
    """Human levels against the v1 descriptions' `discard` and `icon_usefulness`, one
    count per distinct glyph."""
    hashes = glyph_hashes()
    human = {hashes.get(n): lab for n, lab in icon_labels().items()}
    kept_by_model = {hashes[ic.name] for ic in load_icons()}
    cols = ("discard", "≤6", "7", "≥8")
    rows = CURATION_LABELS + ("unlabelled",)
    cell = {(r, c): 0 for r in rows for c in cols}
    for ic in load_icons(include_discarded=True):
        g, u = hashes[ic.name], round(ic.usefulness * 10)
        col = "discard" if g not in kept_by_model else "≤6" if u <= 6 else "7" if u == 7 else "≥8"
        cell[(human.get(g, "unlabelled"), col)] += 1
    total = sum(cell.values())
    done = total - sum(cell[("unlabelled", c)] for c in cols)
    print(f"  labelled by hand: {done} of {total} glyphs")
    print(f"  {'':12}" + "".join(f"{c:>9}" for c in cols) + "   (v1: discard, else icon_usefulness)")
    for r in rows:
        print(f"  {r:12}" + "".join(f"{cell[(r, c)]:9d}" for c in cols))
    # The model's discard as a predictor of a human reject, over the labelled glyphs.
    both = cell[("reject", "discard")]
    rejected = sum(cell[("reject", c)] for c in cols)
    discarded = sum(cell[(r, "discard")] for r in CURATION_LABELS)
    if rejected and discarded:
        print(f"  model discard vs. human reject: precision {both / discarded:.2f}, "
              f"recall {both / rejected:.2f}")


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
