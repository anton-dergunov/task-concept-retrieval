"""Load icon descriptions + catalog into a usable corpus.

Two description versions (docs/icon-descriptions.md): v1 in
data/icon_descriptions/, v2 in data/icon_descriptions_v2/. `load_icons(version=…)`
picks one (default `config.DESC_VERSION`, env TCR_DESC); both map onto the same
IconDoc field views, so every matcher runs on either.

Hard rule (see CLAUDE.md): we match ONLY on the generated description fields
(visual_concepts / readings / task_intents / example_tasks / reasoning). The icon
`name`, `tags`, and `categories` are NEVER matching features — `name` is used only
as a stable id and to locate the rendered PNG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from . import config

# Field views used by the various methods. `readings` is empty in v1.
TEXT_FIELDS = ("visual_concepts", "readings", "task_intents", "example_tasks", "reasoning")
VERSIONS = ("v1", "v2")


@dataclass
class IconDoc:
    name: str
    char: str
    popularity: int
    usefulness: float  # prior in [0, 1]: v1 icon_usefulness / 10; v2 from its scores
    visual_concepts: List[str]   # v1: visual concepts; v2: what is drawn (depiction)
    task_intents: List[str]
    example_tasks: List[str]     # v2: EN ("Parent: task" when nested), then ES and RU
    poor_matches: List[str]      # v1: poor_matches; v2: near-miss tasks
    reasoning: str
    version: str = "v1"
    readings: List[str] = field(default_factory=list)   # v2: meanings, strongest first
    # v2 only; kept for training and the abstention gate, not matched on (yet).
    context_tasks: List[str] = field(default_factory=list)   # rank-2 positives
    scores: Dict[str, int] = field(default_factory=dict)
    ambiguity: str = ""
    distinguishing: str = ""     # from the contrast pass, for glyphs with look-alikes

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
        parts = [self.field_text("visual_concepts"), self.field_text("readings"),
                 self.field_text("task_intents")]
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


def load_icons(include_discarded: bool = False, dedupe: bool = True,
               version: Optional[str] = None) -> tuple:
    """Return a tuple of IconDoc (tuple so it is hashable/cacheable).

    `version` is "v1" or "v2" (default: config.DESC_VERSION). Drops discarded icons
    unless include_discarded=True, and (by default) collapses glyph aliases and blank
    renders so every icon looks distinct.
    """
    return _load_icons(include_discarded, dedupe, version or config.DESC_VERSION)


@lru_cache(maxsize=8)
def _load_icons(include_discarded: bool, dedupe: bool, version: str) -> tuple:
    if version not in VERSIONS:
        raise ValueError(f"unknown description version {version!r}; choices: {VERSIONS}")
    cat = _catalog_index()
    reader = _v1_doc if version == "v1" else _v2_doc
    folder = config.ICON_DESC_DIR if version == "v1" else config.ICON_DESC_V2_DIR
    icons: List[IconDoc] = []
    for path in sorted(folder.glob("*.json")):
        name = path.stem
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # v2 aliases are copies; the original carries pass 1's representative name
        # (v1's choice), so both versions call a glyph by the same name.
        if dedupe and (d.get("_meta") or {}).get("alias_of"):
            continue
        meta = cat.get(name, {})
        ic, discard = reader(name, d, meta)
        if discard and not include_discarded:
            continue
        icons.append(ic)
    return tuple(_dedupe_glyphs(icons) if dedupe else icons)


def _base(name: str, meta: dict) -> dict:
    return {"name": name, "char": meta.get("char", ""),
            "popularity": int(meta.get("popularity", 0) or 0)}


def _v1_doc(name: str, d: dict, meta: dict):
    return IconDoc(
        **_base(name, meta),
        usefulness=float(d.get("icon_usefulness", 0) or 0) / 10.0,
        visual_concepts=_coerce_list(d.get("visual_concepts")),
        task_intents=_coerce_list(d.get("task_intents")),
        example_tasks=_coerce_list(d.get("example_tasks")),
        poor_matches=_coerce_list(d.get("poor_matches")),
        reasoning=str(d.get("reasoning", "") or ""),
    ), bool(d.get("discard"))


def _v2_doc(name: str, d: dict, meta: dict):
    """Map a v2 description onto the IconDoc views (docs/icon-descriptions.md §3.2)."""
    dep = d.get("depiction") or {}
    sc = d.get("scores") or {}
    examples = [f"{e['parent']}: {e['task']}" if str(e.get("parent", "")).strip() else e["task"]
                for e in d.get("example_tasks", [])]
    examples += _coerce_list(d.get("example_tasks_es")) + _coerce_list(d.get("example_tasks_ru"))
    ic = IconDoc(
        **_base(name, meta),
        # Placeholder prior on v1's 0–10 scale until it is fitted on labels (§3):
        # 3 + 3 -> 0.6, 4 + 4 -> 0.8.
        usefulness=(sc.get("recognizability", 0) + sc.get("task_fit", 0)) / 10.0,
        visual_concepts=_coerce_list(dep.get("primary_object")) + _coerce_list(dep.get("modifiers")),
        readings=[r["meaning"] for r in d.get("readings", []) if str(r.get("meaning", "")).strip()],
        task_intents=_coerce_list(d.get("task_intents")),
        example_tasks=examples,
        poor_matches=[n["task"] for n in d.get("near_misses", [])],
        reasoning=str(d.get("reasoning", "") or ""),
        version="v2",
        context_tasks=[c["task"] for c in d.get("context_tasks", [])],
        scores=dict(sc),
        ambiguity=str(d.get("ambiguity", "") or ""),
        distinguishing=str(d.get("distinguishing", "") or ""),
    )
    # The manual review is the reference for discard: a glyph it calls useful or
    # somewhat stays even when the model discarded it. Its rejects have no v2 file.
    return ic, bool(d.get("discard")) and _review_level(name) not in ("useful", "somewhat")


def _review_level(name: str) -> Optional[str]:
    return _review_by_glyph().get(glyph_hashes().get(name))


@lru_cache(maxsize=1)
def _review_by_glyph() -> Dict[Optional[str], str]:
    hashes = glyph_hashes()
    return {hashes.get(n): lab for n, lab in icon_labels().items()}


def icons_by_name(include_discarded: bool = False, version: Optional[str] = None) -> Dict[str, IconDoc]:
    return {ic.name: ic for ic in load_icons(include_discarded, version=version)}


# Levels of the manual review of the icon set, worst to best.
CURATION_LABELS = ("reject", "somewhat", "useful")


def curation_label(rec: dict) -> Optional[str]:
    """The level an event gives its icon, or None when it clears the level. Events
    from before the three levels existed only say `removed`."""
    if "label" in rec:
        return rec["label"]
    return "reject" if rec["removed"] else None


def icon_labels(path=None) -> Dict[str, str]:
    """{icon name: level} from the manual review of the icon set (the labeller's
    /curate page); icons not reviewed yet are absent. The file is an event log: the
    latest event per icon wins. Each name stands for its glyph, i.e. for all of its
    aliases too."""
    path = path or config.CURATION_PATH
    labels: Dict[str, str] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    label = curation_label(rec)
                    if label is None:
                        labels.pop(rec["icon"], None)
                    else:
                        labels[rec["icon"]] = label
    return labels
