#!/usr/bin/env python3
"""Build the icon-review part of the labelling bundle (served at /curate).

The review shows every distinct glyph once, to remove by hand the ones that cannot
label a task. That includes the glyphs the v1 descriptions discarded, with nothing
telling them apart, so the pass is blind and can audit the model's `discard`.

labeller/bundle/
  curation.json             {build, cell, cols, rows, per_sheet, icons[], groups[[start, size]]}
  sprites/<build>-<n>.png   the glyphs in display order, cols × rows per sheet

Display order: look-alike glyphs are neighbours (image embeddings, hierarchical
clustering; tcr.glyphs), so a family such as 1x / 1.5x / 2x is judged together.
Icon names and font metadata play no part. The order of an existing bundle is
reused when the icon set is unchanged (--refresh to recompute; it takes minutes).

Run: python scripts/build_curation_bundle.py [--groups 80] [--model …] [--refresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from tcr import config  # noqa: E402
from tcr.data import load_icons  # noqa: E402
from tcr.glyphs import GLYPH_MODEL, glyph_embeddings, lookalike_order  # noqa: E402

BUNDLE = ROOT / "labeller" / "bundle"
CELL = 96            # px per glyph: 1.5× the medium tile, sharp on a tablet
COLS, ROWS = 10, 10  # per sheet; small enough for any device to decode


def layout(names, model, n_groups, refresh):
    """(names in display order, [(start, size)] groups), reusing the previous build's."""
    path = BUNDLE / "curation.json"
    if path.exists() and not refresh:
        old = json.loads(path.read_text(encoding="utf-8"))
        if (set(old["icons"]) == set(names) and old.get("model") == model
                and len(old["groups"]) == n_groups):
            print("order: reused from the existing bundle")
            return old["icons"], [tuple(g) for g in old["groups"]]
    print(f"embedding {len(names)} glyphs with {model} and ordering them …")
    order, groups = lookalike_order(glyph_embeddings(names, model), n_groups)
    return [names[i] for i in order], groups


def write_sprites(names, build):
    out = BUNDLE / "sprites"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        p.unlink()
    per_sheet = COLS * ROWS
    for s in range(0, len(names), per_sheet):
        sheet = Image.new("L", (COLS * CELL, ROWS * CELL), 255)
        for k, name in enumerate(names[s:s + per_sheet]):
            glyph = Image.open(config.ICON_PNG_DIR / f"{name}.png").convert("L")
            sheet.paste(glyph.resize((CELL, CELL), Image.LANCZOS),
                        ((k % COLS) * CELL, (k // COLS) * CELL))
        sheet.save(out / f"{build}-{s // per_sheet}.png", optimize=True)
    return sorted(out.glob("*.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=80, help="number of look-alike groups")
    ap.add_argument("--model", default=GLYPH_MODEL, help="image encoder (CLIP / SigLIP)")
    ap.add_argument("--refresh", action="store_true", help="recompute the display order")
    args = ap.parse_args()

    # One name per distinct glyph, discarded ones included, blank renders dropped.
    names = [ic.name for ic in load_icons(include_discarded=True)]
    BUNDLE.mkdir(parents=True, exist_ok=True)
    icons, groups = layout(names, args.model, args.groups, args.refresh)

    build = hashlib.sha1(f"{CELL}|{COLS}|{ROWS}|{'|'.join(icons)}".encode()).hexdigest()[:10]
    sheets = write_sprites(icons, build)
    (BUNDLE / "curation.json").write_text(json.dumps(
        {"build": build, "cell": CELL, "cols": COLS, "rows": ROWS, "per_sheet": COLS * ROWS,
         "model": args.model, "icons": icons, "groups": groups}, ensure_ascii=False))
    size = sum(p.stat().st_size for p in sheets) / 1e6
    print(f"{len(icons)} glyphs in {len(groups)} groups (sizes {min(s for _, s in groups)}–"
          f"{max(s for _, s in groups)}); {len(sheets)} sheets, {size:.1f} MB -> {BUNDLE}")


if __name__ == "__main__":
    main()
