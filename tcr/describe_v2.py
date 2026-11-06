"""The v2 icon-description request: prompt, response schema, validation, references.

docs/icon-descriptions-v2.md §3–5. Shared by the model pilot (scripts/v2_pilot.py)
and, later, the full pass 1. Hard rule (CLAUDE.md): the request never contains an
icon's name, tags or categories; the glyph image is the only source of truth. The
name is used here only to locate the glyph.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from . import config

PROMPT_VERSION = "v2.0"

DOMAINS = ("work", "career", "learning", "tech", "health_fitness", "finance", "home",
           "shopping", "food", "social_family", "travel", "hobbies", "admin",
           "communication", "creative", "outdoors")

TRAPS = ("property", "generic_verb", "shared_vocabulary", "lookalike_glyph", "parent_topic")

# Exact counts (within the ranges of v2 §3) keep the models' outputs comparable.
N = {"readings": (1, 4), "modifier_semantics": (0, 6), "domains": (1, 8),
     "task_intents": (6, 12), "example_tasks": (12, 20), "context_tasks": (3, 6),
     "near_misses": (5, 8), "example_tasks_es": (3, 3), "example_tasks_ru": (3, 3)}
LENGTH_MIX = {"2-5": 5, "6-12": 8, "13-16": 3}   # example_tasks by word count: 16 in all
N_WITH_PARENT = 4

FONT_NAME = "MaterialSymbolsOutlined-Regular.ttf"
FONT_DIRS = ("~/Library/Fonts", "/Library/Fonts", "/usr/share/fonts/truetype")
GLYPH_CACHE = config.ROOT / ".cache" / "glyphs256"

SYSTEM = f"""You describe one icon for a task-planning app. You see only the icon's image.

## What the icon is for

In a long to-do list, every task may show one small black-and-white icon. The icon is a
glanceable pictorial reminder of what the task is about: scanning the list, a person
should recall a task's subject from its icon faster than by reading the title. It cues
the ONE thing the task is about. A wrong or vague icon costs attention, so no icon is
better than a misleading one.

What an icon should show for a task, in order of priority:
1. The task's central object or subject, usually the head noun of the title.
   "Buy a small waterproof rechargeable headlamp for trips" -> a headlamp.
2. The context, domain or kind of activity it belongs to (purpose, place, project).
   Headlamp -> light, or trips / hiking.
3. The action, only when the verb is the point of the task (call, pay, fix, read, sign,
   ship). Generic verbs (review, check, update, organize, buy) are rarely the point.
4. Never the properties: adjectives, attributes, quantities, dates, deadlines, names.
   A rain icon for "waterproof" is wrong.
Specific beats generic; literal beats metaphorical; recognizability beats correctness.
A task may sit under a parent heading (a project or list). If the title is specific the
title wins; if the title is generic, the parent's topic becomes the subject.

## How to work

Analyze ONLY the visual appearance. The image is the only source of truth: there is no
name, label or metadata, and you must not guess one.

1. Describe before interpreting. First record literally what is drawn: the base object,
   then every modifier (a badge, plus, slash, check, clock, arrow, dot, exclamation mark,
   magnifier, overlay). Look closely: small marks change the meaning, and some glyphs
   resemble other objects at small sizes.
2. Then say what a viewer would take it to mean, ranked. Tie each modifier to its
   meaning (plus = add/new, slash = disabled/cancelled, clock = scheduled/history,
   check = done/verified, ! = alert, magnifier = search, ...).
3. Then generate tasks for which this icon would be a good choice, written the way real
   people write task titles.

## Fields

- depiction: primary_object (literal, 3-10 words), modifiers (literal; [] if none),
  composition (e.g. "single object, front view").
- readings: 1-4 meanings a viewer would take it to have, strongest first; strength is
  "primary" or "secondary".
- ambiguity: what it could be mistaken for, and when (e.g. at small sizes); "" if none.
- modifier_semantics: each modifier with its meaning; [] if none.
- domains: which of the fixed domains its tasks come from: {", ".join(DOMAINS)}.
- task_intents: 6-12 verb + object phrases of 3-7 words ("track body weight over time").
- example_tasks: exactly 16 task TITLES whose CENTRAL subject this icon shows, so that it
  would be the best icon for them (rank 1). Rules:
  - Lengths: {LENGTH_MIX["2-5"]} of 2-5 words (quick jots), {LENGTH_MIX["6-12"]} of 6-12
    words, {LENGTH_MIX["13-16"]} of 13-16 words.
  - Exactly {N_WITH_PARENT} of them have a parent heading (a project, area or list name),
    and for those the title is generic and the parent supplies the subject; the others
    have parent "".
  - Each has a domain from the list above; together they cover at least 3 domains,
    across ordinary life (home, health, money, family, hobbies, travel, learning, work,
    tech...), not only office work.
  - Concrete and varied: specific objects, places, numbers and details as real people
    write them; no two alike; no template phrasing; not the reference titles.
  - Titles only: what the task is, not how it will be done.
- example_tasks_es / example_tasks_ru: exactly 3 each, written natively in Spanish and in
  Russian (not translations of the English ones), same rules.
- context_tasks: 3-6 tasks where this icon fits as the CONTEXT or activity, not the
  central subject (a good second choice); "central" names what the central subject is.
- near_misses: 5-8 hard negatives: tasks that mention or touch this icon's concept but
  deserve a DIFFERENT icon. "better" names the icon concept that fits better; "trap" is
  one of: property (the concept is only an attribute: "waterproof"), generic_verb (only a
  generic verb: "check", "verify"), shared_vocabulary (same word, other subject),
  lookalike_glyph (a task for a glyph this one resembles), parent_topic (the parent
  matches but a specific title points elsewhere). Use at least 3 different traps. They
  must be hard: never a random unrelated task.
- scores (integers 1-5):
  - recognizability: 1 = most people would guess wrong; 3 = clear in context;
    5 = universally obvious.
  - task_fit: 1 = almost no to-do tasks; 3 = one narrow family; 5 = many common tasks
    across domains.
  - distinctiveness: 1 = interchangeable with a look-alike icon; 5 = unique.
- discard: true only if the icon can never be the subject of a task: it shows only a
  property, quantity or setting (a speed like 1.5x, a megapixel count, signal bars); it
  is interface chrome with no subject (chevrons, drag handles, layout toggles); it is
  only letters or digits; or it cannot be read at a glance. Still fill every field.
- reasoning: 1-3 sentences: what it shows, what it is good for, what it could be
  confused with.

Return only JSON that follows the schema."""


def _arr(items: dict, lo: int, hi: int) -> dict:
    # Vertex rejects this schema (400 INVALID_ARGUMENT) once it carries minItems /
    # maxItems / minimum / maximum, so the counts live in the prompt and in
    # soft_problems(); enums are accepted. lo / hi are kept for the reader.
    return {"type": "array", "items": items}


_S = {"type": "string"}
_SCORE = {"type": "integer"}   # 1-5, checked in hard_problems()

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "depiction": {"type": "object", "properties": {
            "primary_object": _S, "modifiers": _arr(_S, 0, 8), "composition": _S},
            "required": ["primary_object", "modifiers", "composition"]},
        "readings": _arr({"type": "object", "properties": {
            "meaning": _S, "strength": {"type": "string", "enum": ["primary", "secondary"]}},
            "required": ["meaning", "strength"]}, *N["readings"]),
        "ambiguity": _S,
        "modifier_semantics": _arr({"type": "object", "properties": {
            "modifier": _S, "meaning": _S}, "required": ["modifier", "meaning"]},
            *N["modifier_semantics"]),
        "domains": _arr({"type": "string", "enum": list(DOMAINS)}, *N["domains"]),
        "task_intents": _arr(_S, *N["task_intents"]),
        "example_tasks": _arr({"type": "object", "properties": {
            "task": _S, "parent": _S, "domain": {"type": "string", "enum": list(DOMAINS)}},
            "required": ["task", "parent", "domain"]}, *N["example_tasks"]),
        "example_tasks_es": _arr(_S, *N["example_tasks_es"]),
        "example_tasks_ru": _arr(_S, *N["example_tasks_ru"]),
        "context_tasks": _arr({"type": "object", "properties": {
            "task": _S, "central": _S}, "required": ["task", "central"]}, *N["context_tasks"]),
        "near_misses": _arr({"type": "object", "properties": {
            "task": _S, "better": _S, "trap": {"type": "string", "enum": list(TRAPS)}},
            "required": ["task", "better", "trap"]}, *N["near_misses"]),
        "scores": {"type": "object", "properties": {
            "recognizability": _SCORE, "task_fit": _SCORE, "distinctiveness": _SCORE},
            "required": ["recognizability", "task_fit", "distinctiveness"]},
        "discard": {"type": "boolean"},
        "reasoning": _S,
    },
    "required": ["depiction", "readings", "ambiguity", "modifier_semantics", "domains",
                 "task_intents", "example_tasks", "example_tasks_es", "example_tasks_ru",
                 "context_tasks", "near_misses", "scores", "discard", "reasoning"],
}

_CYR = re.compile(r"[Ѐ-ӿ]")


def hard_problems(d) -> List[str]:
    """Problems that make a response unusable (it is retried)."""
    if not isinstance(d, dict):
        return ["not an object"]
    out = [f"missing {k}" for k in RESPONSE_SCHEMA["required"] if k not in d]
    if out:
        return out
    dep = d["depiction"]
    if not isinstance(dep, dict) or not str(dep.get("primary_object", "")).strip():
        out.append("depiction.primary_object empty")
    for k in N:
        if not isinstance(d[k], list):
            out.append(f"{k} not a list")
    if out:
        return out
    sc = d["scores"]
    for k in ("recognizability", "task_fit", "distinctiveness"):
        v = sc.get(k) if isinstance(sc, dict) else None
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5:
            out.append(f"scores.{k} invalid")
    if not isinstance(d["discard"], bool):
        out.append("discard not boolean")
    if not all(isinstance(e, dict) and str(e.get("task", "")).strip() for e in d["example_tasks"]):
        out.append("example_tasks malformed")
    if len(d["example_tasks"]) < 8:
        out.append("fewer than 8 example_tasks")
    # Runaway generation (seen once in the pilot: a "meaning" listing hundreds of terms
    # in dozens of languages). Every field is short by design; reasoning is 1-3 sentences.
    longest = max((len(s) for s in _strings(d)), default=0)
    if longest > MAX_STRING:
        out.append(f"runaway string of {longest} characters")
    return out


MAX_STRING = 600


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)


def _parse_or_none(text: str):
    """The JSON object in a response's text, or None."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except ValueError:
        return None


def words(text: str) -> int:
    return len(str(text).split())


def soft_problems(d) -> List[str]:
    """Departures from the requested counts and rules; recorded, not retried."""
    out = []
    for k, (lo, hi) in N.items():
        if not lo <= len(d[k]) <= hi:
            out.append(f"{k}: {len(d[k])} not in {lo}-{hi}")
    ex = d["example_tasks"]
    if len(ex) != sum(LENGTH_MIX.values()):
        out.append(f"example_tasks: {len(ex)} != 16")
    n_par = sum(1 for e in ex if str(e.get("parent", "")).strip())
    if n_par != N_WITH_PARENT:
        out.append(f"example_tasks with parent: {n_par} != {N_WITH_PARENT}")
    doms = {e.get("domain") for e in ex}
    if len(doms) < 3:
        out.append(f"example_tasks span {len(doms)} domains (< 3)")
    if any(_CYR.search(t) for t in d["example_tasks_es"]):
        out.append("example_tasks_es contains Cyrillic")
    if not all(_CYR.search(t) for t in d["example_tasks_ru"]):
        out.append("example_tasks_ru not all Cyrillic")
    traps = {n.get("trap") for n in d["near_misses"]}
    if len(traps) < 3:
        out.append(f"near_misses use {len(traps)} traps (< 3)")
    if any(not str(n.get("better", "")).strip() for n in d["near_misses"]):
        out.append("near_misses with empty 'better'")
    return out


# --- Reference titles (style only; training-only pool, never eval) -----------------

def norm(text: str) -> str:
    """Normalized title for exact-duplicate checks: lowercase, punctuation to spaces."""
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def eval_titles() -> List[str]:
    """Every title of every eval dataset (with Public-expanded's parallel versions).
    Personal is read locally and only compared against; it is never sent anywhere."""
    from .datasets import DATASETS, read_jsonl
    out = []
    for ds in DATASETS.values():
        if not ds.path.exists():
            continue
        for r in read_jsonl(ds.path):
            out.append(r["title"])
            # Parallel versions: {lang: title} (Public-short) or {lang: {title, body}}.
            for v in ((r.get("meta") or {}).get("parallel") or {}).values():
                t = v.get("title") if isinstance(v, dict) else v
                if t:
                    out.append(t)
    return out


# --- Contamination guard (§4) -------------------------------------------------------

GUARD_COSINE = 0.95


@lru_cache(maxsize=1)
def _eval_index():
    """(normalized eval titles, their e5 embeddings). Personal stays on this machine."""
    from .index import get_model
    evals = eval_titles()
    pre = config.query_prefix_for(config.MULTILINGUAL_MODEL)
    E = get_model(config.MULTILINGUAL_MODEL).encode(
        [pre + t for t in evals], batch_size=128, normalize_embeddings=True)
    return frozenset(norm(t) for t in evals), E


def contamination_hits(texts: Sequence[str], threshold: float = GUARD_COSINE,
                       embed: bool = True) -> Tuple[set, set]:
    """Indices of `texts` that equal an eval title after norm() (exact), and of those
    within e5 cosine `threshold` of one (near; exact matches usually fall here too)."""
    banned, E = _eval_index() if embed else (frozenset(norm(t) for t in eval_titles()), None)
    exact = {i for i, t in enumerate(texts) if norm(t) in banned}
    near = set()
    if embed and len(texts):
        from .index import get_model
        pre = config.query_prefix_for(config.MULTILINGUAL_MODEL)
        X = get_model(config.MULTILINGUAL_MODEL).encode(
            [pre + t for t in texts], batch_size=256, normalize_embeddings=True)
        for s in range(0, len(texts), 4096):
            best = (X[s:s + 4096] @ E.T).max(axis=1)
            near.update(int(s + i) for i in (best >= threshold).nonzero()[0])
    return exact, near


@lru_cache(maxsize=1)
def reference_pool() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(short, long) reference titles: MS-LaTTE titles that no public dataset uses, and
    the long style exemplars; minus anything equal to an eval title."""
    from .datasets import DATASETS, read_jsonl
    used = set()
    for key in ("public_short", "public_expanded"):
        if DATASETS[key].path.exists():
            for r in read_jsonl(DATASETS[key].path):
                m = r.get("meta") or {}
                if m.get("source") == "mslatte":
                    used.add(str(m.get("source_id")))
    banned = {norm(t) for t in eval_titles()}
    short, seen = [], set()
    raw = json.loads((config.RAW_DIR / "MS-LaTTE.json").read_text(encoding="utf-8"))
    for r in raw:
        t = " ".join(str(r.get("TaskTitle", "")).split())
        n = norm(t)
        if (str(r.get("ID")) in used or n in banned or n in seen
                or not 2 <= words(t) <= 8 or not re.search(r"[a-zA-Z]", t)
                or re.search(r"https?://|@|\d{3,}", t)):
            continue
        seen.add(n)
        short.append(t)
    long_ = []
    path = config.DATASETS_DIR / "style_exemplars.jsonl"
    for r in read_jsonl(path) if path.exists() else []:
        if r.get("lang", "en") == "en" and norm(r["title"]) not in banned and words(r["title"]) >= 6:
            long_.append(r["title"])
    return tuple(short), tuple(long_)


def references_for(key: str, n_short: int = 5, n_long: int = 3) -> List[str]:
    """Rotating reference titles for one glyph, the same for every model (seeded by
    `key`, normally the glyph hash)."""
    short, long_ = reference_pool()
    rng = random.Random(hashlib.sha1(("refs-v1" + key).encode()).hexdigest())
    picked = rng.sample(short, min(n_short, len(short))) + rng.sample(long_, min(n_long, len(long_)))
    rng.shuffle(picked)
    return picked


# --- The glyph -------------------------------------------------------------------

def _codepoints() -> Dict[str, int]:
    out = {}
    for line in (config.DATA_DIR / "codepoints.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = int(parts[1], 16)
    return out


def _font_path():
    from pathlib import Path
    for d in FONT_DIRS:
        p = Path(d).expanduser() / FONT_NAME
        if p.exists():
            return p
    return None


def glyph_png(name: str, px: int = 256) -> bytes:
    """The glyph as PNG bytes, black on white, re-rendered crisply at `px` from the font
    (the same scale as scripts/render_icons.py: 3/4 of the canvas) and centred on its
    ink; the 128 px render if the font is missing."""
    path = GLYPH_CACHE / f"{px}" / f"{name}.png"
    if path.exists():
        return path.read_bytes()
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    font_path = _font_path()
    if font_path is None:
        return (config.ICON_PNG_DIR / f"{name}.png").read_bytes()
    font = ImageFont.truetype(str(font_path), round(px * 0.75))
    glyph = chr(_codepoints()[name])
    big = Image.new("L", (px * 2, px * 2), 255)
    ImageDraw.Draw(big).text((px // 2, px // 2), glyph, fill=0, font=font)
    box = ImageOps.invert(big).getbbox()
    img = Image.new("RGB", (px, px), "white")
    if box:
        ink = big.crop(box)
        img.paste(ink.convert("RGB"), ((px - ink.width) // 2, (px - ink.height) // 2))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return buf.getvalue()


def build_request(name: str, key: Optional[str] = None) -> Tuple[str, str, List[Tuple[bytes, str]]]:
    """(system, prompt, images) for one glyph. `key` seeds the reference titles
    (default: the glyph's pixel hash, so aliases get the same request)."""
    if key is None:
        from .data import glyph_hashes
        key = glyph_hashes()[name] or name
    refs = references_for(key)
    prompt = ("Describe the attached icon.\n\n"
              "For STYLE ONLY, here are some real task titles from other people's lists "
              "(unrelated to this icon). Match their naturalness and variety; do not copy "
              "or paraphrase them:\n" + "\n".join(f"- {t}" for t in refs))
    return SYSTEM, prompt, [(glyph_png(name), "image/png")]
