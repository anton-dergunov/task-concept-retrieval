"""Load icon descriptions + catalog into a usable corpus.

Hard rule (see CLAUDE.md): we match ONLY on the generated description fields
(visual_concepts / task_intents / example_tasks / reasoning). The icon `name`,
`tags`, and `categories` are NEVER matching features — `name` is used only as a
stable id and to locate the rendered PNG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from . import config

# Field views used by the various methods.
TEXT_FIELDS = ("visual_concepts", "task_intents", "example_tasks", "reasoning")


@dataclass
class IconDoc:
    name: str
    char: str
    popularity: int
    usefulness: float  # icon_usefulness / 10, in [0, 1]
    visual_concepts: List[str]
    task_intents: List[str]
    example_tasks: List[str]
    poor_matches: List[str]
    reasoning: str

    def field_text(self, fieldname: str) -> str:
        """Flattened text for one field view (list joined with ', ')."""
        val = getattr(self, fieldname)
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val)

    def document(self, include_examples: bool = True) -> str:
        """Concatenated document for lexical / single-vector methods.

        `include_examples=False` excludes example_tasks — used by the bootstrap
        eval, which holds out example_tasks as queries and must not also index
        them (see docs/experimentation-strategy.md).
        """
        parts = [self.field_text("visual_concepts"), self.field_text("task_intents")]
        if include_examples:
            parts.append(self.field_text("example_tasks"))
        parts.append(self.reasoning)
        return ". ".join(p for p in parts if p)


def _coerce_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if value is None:
        return []
    return [str(value)]


@lru_cache(maxsize=1)
def _catalog_index() -> Dict[str, dict]:
    with open(config.CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return {entry["name"]: entry for entry in catalog}


@lru_cache(maxsize=1)
def glyph_hashes() -> Dict[str, Optional[str]]:
    """Pixel hash of every rendered icon; None for a blank render (no ink).

    Material Symbols has many aliases (e.g. `alarm` / `access_alarm`) that draw the
    exact same glyph; to the person looking at the agenda they are one icon.
    """
    import hashlib

    import numpy as np
    from PIL import Image
    out: Dict[str, Optional[str]] = {}
    for path in sorted(config.ICON_PNG_DIR.glob("*.png")):
        a = np.asarray(Image.open(path).convert("L"))
        out[path.stem] = hashlib.sha1(a.tobytes()).hexdigest() if (a < 128).any() else None
    return out


def _dedupe_glyphs(icons: List["IconDoc"]) -> List["IconDoc"]:
    """Keep one icon per distinct glyph (highest usefulness, then popularity, then
    shortest name) and drop blank renders."""
    hashes = glyph_hashes()
    best: Dict[str, IconDoc] = {}
    for ic in icons:
        h = hashes.get(ic.name)
        if h is None:
            continue
        cur = best.get(h)
        key = (ic.usefulness, ic.popularity, -len(ic.name))
        if cur is None or key > (cur.usefulness, cur.popularity, -len(cur.name)):
            best[h] = ic
    keep = {ic.name for ic in best.values()}
    return [ic for ic in icons if ic.name in keep]


@lru_cache(maxsize=4)
def load_icons(include_discarded: bool = False, dedupe: bool = True) -> tuple:
    """Return a tuple of IconDoc (tuple so it is hashable/cacheable).

    Drops icons with discard=true unless include_discarded=True, and (by default)
    collapses glyph aliases and blank renders so every icon looks distinct.
    """
    cat = _catalog_index()
    icons: List[IconDoc] = []
    for path in sorted(config.ICON_DESC_DIR.glob("*.json")):
        name = path.stem
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("discard") and not include_discarded:
            continue
        meta = cat.get(name, {})
        icons.append(
            IconDoc(
                name=name,
                char=meta.get("char", ""),
                popularity=int(meta.get("popularity", 0) or 0),
                usefulness=float(d.get("icon_usefulness", 0) or 0) / 10.0,
                visual_concepts=_coerce_list(d.get("visual_concepts")),
                task_intents=_coerce_list(d.get("task_intents")),
                example_tasks=_coerce_list(d.get("example_tasks")),
                poor_matches=_coerce_list(d.get("poor_matches")),
                reasoning=str(d.get("reasoning", "") or ""),
            )
        )
    return tuple(_dedupe_glyphs(icons) if dedupe else icons)


def icons_by_name(include_discarded: bool = False) -> Dict[str, IconDoc]:
    return {ic.name: ic for ic in load_icons(include_discarded)}


def removed_by_hand(path=None) -> set:
    """Icon names marked for removal in the manual review of the icon set (the
    labeller's /curate page). The file is an event log: the latest event per icon wins.
    Each name stands for its glyph, i.e. for all of its aliases too."""
    path = path or config.CURATION_PATH
    removed: set = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    (removed.add if rec["removed"] else removed.discard)(rec["icon"])
    return removed
