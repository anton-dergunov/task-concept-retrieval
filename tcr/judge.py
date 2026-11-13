"""The LLM judge: picks icons for a task the way the author labels them
(docs/icon-selection-principles.md, "How to choose"; docs/fine-tuning-plan.md §6).

Stages, each one single request (no growing chat, so the tokens per task stay small):

  A  scan and plan   the task + ~100 candidate glyphs on numbered contact sheets with a
                     one-line description each → analysis (topic, associations,
                     qualifiers), every somewhat relevant candidate, and up to 4 searches.
  —  search          local: M3 on the v2 descriptions, 10 new icons per query.
  B  decide          the analysis + the collected icons + the search results → a graded
                     ranking or "no good icon"; may search once more (B runs at most twice).

A task's progress is a JSON-serializable state dict, advanced one response at a time, so
the online runner (`judge_online`) and the batch driver (scripts/judge_batch.py) make
exactly the same requests. The model sees glyphs and descriptions only, never icon names
(CLAUDE.md hard rule 1). Prompt examples come from the author's labels and are private
(data/private/judge/fewshot.json).
"""

from __future__ import annotations

import io
import json
import math
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from . import config
from .datasets import DATASETS, read_jsonl
from .org_query import normalize_text

PROMPT_VERSION = "j4"
# What each prompt version changed (general rules only; the error analysis that led to
# them is in data/private/judge/iterations.md).
PROMPT_HISTORY = {
    "j1": "Procedure (understand → scan all → search → prune and order), principles, "
          "abstract-topics rule, avoid the domain icon, 5 labelled examples.",
    "j2": "The subject is the head of the title in its intended sense; not the action, side "
          "nouns or tool names; no puns; the brain / AI icon is right for tasks about AI "
          "models themselves.",
    "j3": "Things about a topic are about the topic (notes, videos, lists → what they are "
          "about); collect icons of the parent heading's domain too; be more inclusive.",
    "j4": "Ranking order: the plain icon of the central object first, then its context, then "
          "icons of specific aspects, operations or metaphors.",
}
JUDGE_DIR = config.PRIVATE_DIR / "judge"
FEWSHOT_PATH = JUDGE_DIR / "fewshot.json"
SPEND_LOG = JUDGE_DIR / "spend.jsonl"
BUNDLE_TASKS = config.ROOT / "labeller" / "bundle" / "tasks"

DEFAULT_MODEL = "gemini-3.8-flash"
GEN = {"thinking_level": "low", "media_resolution": "high"}
TEMPERATURE = 1.0
SHEET = 25            # glyphs per contact sheet (5 × 5)
CELL_PX = 200
MAX_QUERIES = 4       # per search step
PER_QUERY = 10        # new icons per query
SEARCH_ID0 = 101      # search results are numbered from here; the pool is 1–100
MAX_B_ROUNDS = 2
MAX_SHORTLIST = 60
N_SIBLINGS = 10

# USD per 1M tokens: (online in, online out, batch in, batch out); output includes
# thinking. Read 2026-11-05 (scripts/v2_pilot.py PRICES); promotions: 3.8 Flash until
# 2026-12-31, Argon until 2026-11-30, both double afterwards.
PRICES = {
    "gemini-3-flash-preview": (0.50, 3.00, 0.25, 1.50),
    "gemini-3.8-flash": (0.75, 3.75, 0.375, 1.875),
    "gemini-3.1-pro-preview": (2.00, 12.00, 1.00, 6.00),
    "gemini-4-argon": (2.00, 10.00, 1.00, 5.00),
}


def usd(usage: dict, model: str, batch: bool = False) -> float:
    p = PRICES.get(model, (2.0, 12.0, 1.0, 6.0))
    pin, pout = (p[2], p[3]) if batch else (p[0], p[1])
    return (usage.get("prompt", 0) * pin
            + (usage.get("output", 0) + usage.get("thoughts", 0)) * pout) / 1e6


_SPEND_LOCK = threading.Lock()


def log_spend(model: str, usage: dict, what: str, batch: bool = False) -> float:
    cost = usd(usage, model, batch)
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model,
           "what": what, "batch": batch, "usd": round(cost, 6),
           **{k: usage.get(k, 0) for k in ("prompt", "output", "thoughts")}}
    with _SPEND_LOCK, open(SPEND_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return cost


def spent() -> float:
    if not SPEND_LOG.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in SPEND_LOG.read_text().splitlines() if l.strip())


# --- prompts ---------------------------------------------------------------------------

SYSTEM = """\
You choose icons for tasks in one person's task list. In their agenda, each task gets one \
small black-and-white icon: a glanceable reminder of what the task is about. You see the \
candidate icons as images with red numbers, each with a one-line description of what it \
draws.

What a good icon is:
- It cues the one thing the task is about, so that icon + a glance at the title recall \
the task at once.
- Priority: (1) the central subject of the task; (2) its context, domain or kind of \
activity; (3) the action, only when the verb is the point (call, pay, fix, read, sign).
- The subject is the head of the title, in the sense the task means it: "Serving MoE \
models" is about serving (servers), "Train a tone classifier" (in a Mandarin speech \
project) is about Mandarin speech, "Read a book on building LLMs" is about LLMs. When the \
title's words are generic or figurative (ramp, notes, list, lenses, landscape), the \
parent heading's topic is the subject: "Decide the application ramp" under "Job search" \
is about the job search.
- Things about a topic are about the topic. When the title does something to generic \
things (merge the notes, recover the videos, fold a list, set up, decide, write down), \
the subject is what those things are about, often named by a qualifier or by the parent \
heading: "Fold the ML interview notes into the list" is about interviews; "Recover the \
vegan videos from backups" under "Cooking" is about cooking; "Dependency pinning and \
provenance" under "Security for developers" is about security.
- Not the subject: the action (decide, order, merge, fold, delete, write up, compare) \
unless doing it is the whole point; side nouns such as examples in parentheses, outputs \
("report the confusion matrix"), or tool names; and properties that only describe \
(small, urgent, narrow, great, waterproof), quantities, dates or people's names.
- No puns. Never depict another meaning of a word (anchoring → ship anchor, distillation \
→ chemistry flask, judging → gavel, ramp → a physical ramp, lenses → glasses) unless the \
task means that thing.
- Abstract topics. Many tasks are about ideas: things to learn, research directions, \
career decisions. When the head noun is generic (methods, approaches, overview, \
landscape, notes, niche), the qualifier that names the topic is the subject: \
"evolutionary strategies" is about evolution. Find the association a person would recall \
the topic by: a literal object if the topic has one, otherwise a visual metaphor of its \
core idea (auction → gavel, objective → target, variance reduction → shrinking arrows).
- Domain icons. This list is dominated by machine learning and software. Prefer an icon \
of what distinguishes this task when one exists. But when the task is about AI models \
themselves (how they work, training or fine-tuning them, a book or course on them) and \
nothing more specific can be drawn, the brain / AI icon is the right answer, not a \
failure.
- Specific beats generic; recognizable at a glance beats technically correct.
- Imperfect is normal. No icon will fit exactly; a ranked list of several partial fits is \
a good answer. Answer "no good icon" only when nothing beyond generic or property-level \
icons fits.

How to work (the procedure a careful human labeller follows):
1. Understand the task: its main topic at the level that sets it apart, the visual \
associations of that topic, and the qualifiers that define it.
2. Scan every candidate once and collect each icon that is at least somewhat relevant. \
Look at the image itself, not only its description.
3. Search the icon descriptions for associations that no candidate shows well.
4. Prune and order: the best reminder first."""

STAGE_A = """\
Do steps 1–3 for the TASK above.
- analysis.main_topic: the central subject: the head of the title in the sense the task \
means it (the parent's topic when the title is generic or figurative), in a few words.
- analysis.distinguishing_topic: what sets it apart from its sibling tasks and from the \
list's general domain.
- analysis.associations: 5–10 concrete things or visual metaphors that could depict the \
central subject (first) and its context, most specific first. No puns.
- analysis.qualifiers: the qualifiers that name the topic, if any.
- relevant: the numbers of EVERY candidate that is at least somewhat relevant to the task \
or one of its associations, including icons of the parent heading's domain (security → \
shield, cooking → food, job search → briefcase) even when the title does not name it. Be \
inclusive (typically 10–25).
- search: up to 4 short queries (2–5 words naming a drawn object or idea, e.g. "gavel \
auction hammer") for associations that no candidate shows well. Leave it empty when the \
candidates already hold strong fits."""

STAGE_B = """\
Do step 4 for the TASK above. You see your analysis, the icons you collected, and the \
results of your searches (all with their numbers).
- ranking: every icon that works as a reminder for this task, best first, each with a \
grade (3 = shows the central subject; 2 = shows its context, domain or activity; 1 = a \
partial or loose association) and a short reason. Usually 3–8 icons; the top matters most.
  Order: when the subject is a thing that can be drawn (an image, a table, a server, a \
test tube, a watch, a shield for security), its plain icon comes first; then its context \
(the parent's domain); then icons of a specific aspect, operation or metaphor (crop marks \
for preprocessing, a join diagram for joins, routing arrows for expert placement). A \
clever icon of an aspect does not beat the plain icon of the object. Only when the \
subject is an idea with no drawable object does its best metaphor come first.
- no_good_icon: true only when nothing here is better than a generic or property-level \
icon; the ranking may then be empty.
- search: {search_rule}"""

SEARCH_RULE_OPEN = ("up to 3 more queries only if a key association is still missing; "
                    "otherwise empty.")
SEARCH_RULE_CLOSED = "leave empty (no more searches)."

_STR = {"type": "string"}
SCHEMA_A = {
    "type": "object",
    "properties": {
        "analysis": {"type": "object", "properties": {
            "main_topic": _STR, "distinguishing_topic": _STR,
            "associations": {"type": "array", "items": _STR},
            "qualifiers": {"type": "array", "items": _STR}},
            "required": ["main_topic", "distinguishing_topic", "associations", "qualifiers"]},
        "relevant": {"type": "array", "items": {"type": "integer"}},
        "search": {"type": "array", "items": _STR},
    },
    "required": ["analysis", "relevant", "search"],
}
SCHEMA_B = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "grade": {"type": "integer"}, "why": _STR},
            "required": ["id", "grade", "why"]}},
        "no_good_icon": {"type": "boolean"},
        "search": {"type": "array", "items": _STR},
    },
    "required": ["ranking", "no_good_icon", "search"],
}


# --- icons as the model sees them ----------------------------------------------------

@lru_cache(maxsize=None)
def icon_line(name: str) -> str:
    """What the glyph draws and its main reading, from the v2 description (v1 if the
    glyph has none). Never the name."""
    p = config.ICON_DESC_V2_DIR / f"{name}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        reads = "; ".join(r["meaning"] for r in d.get("readings", [])[:2])
        return f"draws {d['depiction']['primary_object']}; reads as {reads}"
    d = json.loads((config.ICON_DESC_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return "shows " + "; ".join(d.get("visual_concepts", [])[:3])


def sheet_png(items: Sequence[tuple]) -> bytes:
    """[(number, icon name)] → one PNG, 5 per row, each glyph under its red number."""
    from PIL import Image, ImageDraw, ImageFont
    from .describe_v2 import glyph_png
    cols, pad = 5, 46
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * CELL_PX, rows * (CELL_PX + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=38)
    except TypeError:
        font = ImageFont.load_default()
    for i, (num, name) in enumerate(items):
        x, y = (i % cols) * CELL_PX, (i // cols) * (CELL_PX + pad)
        draw.text((x + 10, y + 4), str(num), fill=(210, 0, 0), font=font)
        img = Image.open(io.BytesIO(glyph_png(name, 256))).convert("RGB").resize(
            (CELL_PX - 20, CELL_PX - 20), Image.LANCZOS)
        sheet.paste(img, (x + 10, y + pad))
        draw.rectangle([x, y, x + CELL_PX - 1, y + CELL_PX + pad - 1], outline=(190, 190, 190))
    # A 16-colour palette keeps the red numbers and the anti-aliasing and makes the PNG
    # about 5× smaller than RGB, which matters for batch input (4 sheets per request).
    buf = io.BytesIO()
    sheet.quantize(colors=16).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def sheets_parts(numbered: Sequence[tuple], title: str) -> list:
    """Interleaved parts: a caption, then its sheet, for every 25 icons; then the lines."""
    parts = []
    for i in range(0, len(numbered), SHEET):
        chunk = numbered[i:i + SHEET]
        parts.append(f"{title}, sheet {i // SHEET + 1} (numbers {chunk[0][0]}–{chunk[-1][0]}):")
        parts.append((sheet_png(chunk), "image/png"))
    parts.append("Descriptions:\n" + "\n".join(f"{n}: {icon_line(name)}" for n, name in numbered))
    return parts


# --- task context ------------------------------------------------------------------------

def _clean(text: str) -> str:
    return normalize_text(text or "")


def task_block(rec: dict, siblings: Sequence[str]) -> str:
    parents = [p for p in (_clean(p) for p in rec.get("parents") or []) if p]
    lines = ["TASK", f"Title: {_clean(rec['title'])}"]
    if parents:
        lines.append("Parent headings (outermost first): " + " > ".join(parents))
    body = _clean((rec.get("body") or "").replace("\n", " "))[:400]
    if body:
        lines.append(f"Body (start): {body}")
    if siblings:
        lines.append("Other tasks under the same parent:")
        lines += [f"- {s}" for s in siblings]
    return "\n".join(lines)


def fewshot_block(exclude: Sequence[str] = ()) -> str:
    """The author's labelled examples, in words (private: data/private/judge/fewshot.json)."""
    if not FEWSHOT_PATH.exists():
        return ""
    shots = [s for s in json.loads(FEWSHOT_PATH.read_text()) if s["task_id"] not in exclude]
    if not shots:
        return ""
    recs = {r["id"]: r for r in read_jsonl(DATASETS["personal"].path)}
    out = ["EXAMPLES of how this person labels (their ranked picks, described in words):"]
    for s in shots:
        rec = recs[s["task_id"]]
        parents = " > ".join(_clean(p) for p in rec.get("parents") or [])
        out.append(f"\nTask: {_clean(rec['title'])}" + (f"  [{parents}]" if parents else ""))
        a = s["analysis"]
        out.append(f"Analysis: {a['main_topic']} — distinguishing: {a['distinguishing_topic']}; "
                   f"associations: {', '.join(a['associations'])}")
        if s.get("none"):
            out.append(f"Answer: no good icon ({s['why_none']})")
        else:
            out.append("Their picks: " + " | ".join(
                f"{i + 1}) {icon_line(n).split(';')[0].removeprefix('draws ')}"
                for i, n in enumerate(s["picks"])))
            if s.get("note"):
                out.append(f"Note: {s['note']}")
    return "\n".join(out)


class Context:
    """Per-dataset helpers: candidate pools, sibling titles, and the search engine."""

    def __init__(self, dataset: str, records: Optional[List[dict]] = None,
                 fewshot_exclude: Sequence[str] = ()):
        self.dataset = dataset
        self.records = records if records is not None else read_jsonl(DATASETS[dataset].path)
        self.by_id = {r["id"]: r for r in self.records}
        groups: Dict[tuple, List[str]] = {}
        for r in self.records:
            groups.setdefault(tuple(r.get("parents") or []), []).append(r["id"])
        self._groups = groups
        bundle = BUNDLE_TASKS / f"{dataset}.jsonl"
        self._bundle = {r["id"]: r for r in read_jsonl(bundle)} if bundle.exists() else {}
        self._pool_builder = None
        self._search = None
        self._lock = threading.Lock()
        self.fewshot = fewshot_block(exclude=fewshot_exclude)

    def siblings(self, rec: dict) -> List[str]:
        key = tuple(rec.get("parents") or [])
        if not key:
            return []
        ids = [i for i in self._groups.get(key, []) if i != rec["id"]][:N_SIBLINGS]
        return [_clean(self.by_id[i]["title"]) for i in ids]

    def pool(self, rec: dict) -> List[str]:
        """The ~100 candidates: the labelling bundle's grid if the task has one (what the
        human saw), else the same construction (tcr.candidates)."""
        b = self._bundle.get(rec["id"])
        if b and b.get("more"):
            return list(b["icons"]) + list(b["more"])
        with self._lock:
            if self._pool_builder is None:
                from .candidates import PoolBuilder
                self._pool_builder = PoolBuilder()
            first, more, _ = self._pool_builder.pool(rec)
        return first + more

    def search(self, query: str, exclude: set) -> List[str]:
        with self._lock:
            if self._search is None:
                from .data import load_icons
                from .methods import build_method
                self._search = build_method("M3", load_icons(version="v2"))
            ranked = self._search.rank(query, top_k=80)
        return [n for n, _ in ranked if n not in exclude][:PER_QUERY]


# --- the state machine -----------------------------------------------------------------

def start(ctx: Context, rec: dict) -> dict:
    pool = ctx.pool(rec)
    return {"task_id": rec["id"], "dataset": ctx.dataset, "prompt_version": PROMPT_VERSION,
            "ids": {str(i + 1): n for i, n in enumerate(pool)}, "next_id": SEARCH_ID0,
            "pool_size": len(pool), "stage": "A", "b_round": 0, "analysis": None,
            "relevant": [], "searches": [], "shortlist": [], "final": None, "calls": []}


def _header(ctx: Context, rec: dict) -> List[str]:
    parts = [ctx.fewshot] if ctx.fewshot else []
    parts.append(task_block(rec, ctx.siblings(rec)))
    return parts


def request(ctx: Context, state: dict):
    """(system, parts, schema) of the state's next call; the last part is the instruction."""
    rec = ctx.by_id[state["task_id"]]
    ids = state["ids"]
    if state["stage"] == "A":
        numbered = [(int(k), v) for k, v in ids.items() if int(k) < SEARCH_ID0]
        parts = _header(ctx, rec) + [f"CANDIDATES: {len(numbered)} icons."]
        parts += sheets_parts(numbered, "Candidates")
        return SYSTEM, parts + [STAGE_A], SCHEMA_A
    a = state["analysis"] or {}
    numbered = [(i, ids[str(i)]) for i in state["shortlist"]]
    searched = "; ".join(f"\"{s['query']}\" → {', '.join(map(str, s['ids'])) or 'nothing new'}"
                         for s in state["searches"]) or "none"
    parts = _header(ctx, rec) + [
        "YOUR ANALYSIS: " + json.dumps(a, ensure_ascii=False),
        f"Searches so far: {searched}",
        f"COLLECTED ICONS AND SEARCH RESULTS: {len(numbered)} icons."]
    parts += sheets_parts(numbered, "Icons") if numbered else ["(none)"]
    rule = SEARCH_RULE_OPEN if state["b_round"] < MAX_B_ROUNDS - 1 else SEARCH_RULE_CLOSED
    return SYSTEM, parts + [STAGE_B.format(search_rule=rule)], SCHEMA_B


def _run_searches(ctx: Context, state: dict, queries: Sequence[str], limit: int) -> List[int]:
    seen = set(state["ids"].values())
    new_ids = []
    for q in [q.strip() for q in queries if q and q.strip()][:limit]:
        found = ctx.search(q, seen)
        got = []
        for name in found:
            n = state["next_id"]
            state["ids"][str(n)] = name
            state["next_id"] += 1
            seen.add(name)
            got.append(n)
        state["searches"].append({"query": q, "ids": got, "stage": state["stage"]})
        new_ids += got
    return new_ids


def advance(ctx: Context, state: dict, data: dict, call: dict) -> dict:
    """Apply one response (`data`) and move to the next stage."""
    state["calls"].append({"stage": state["stage"], **call})
    valid = state["ids"]
    if state["stage"] == "A":
        state["analysis"] = data.get("analysis")
        rel = [i for i in data.get("relevant", []) if str(i) in valid]
        state["relevant"] = list(dict.fromkeys(rel))
        new = _run_searches(ctx, state, data.get("search") or [], MAX_QUERIES)
        state["shortlist"] = (state["relevant"] + new)[:MAX_SHORTLIST]
        state["stage"] = "B"
        if not state["shortlist"]:
            state["final"] = {"ranking": [], "no_good_icon": True, "reason": "nothing collected"}
            state["stage"] = "done"
        return state
    ranking = []
    for r in data.get("ranking", []):
        if str(r.get("id")) in valid and r["id"] not in [x["id"] for x in ranking]:
            ranking.append({"id": r["id"], "icon": valid[str(r["id"])],
                            "grade": int(r.get("grade", 1)), "why": r.get("why", "")})
    state["b_round"] += 1
    more = data.get("search") or []
    if more and state["b_round"] < MAX_B_ROUNDS:
        new = _run_searches(ctx, state, more, 3)
        keep = [x["id"] for x in ranking] + [i for i in state["shortlist"]
                                             if i not in [x["id"] for x in ranking]]
        state["shortlist"] = (keep + new)[:MAX_SHORTLIST]
        state["draft"] = ranking
        if new:
            return state
    state["final"] = {"ranking": ranking, "no_good_icon": bool(data.get("no_good_icon"))}
    state["stage"] = "done"
    return state


def judge_online(ctx: Context, rec: dict, model: str = DEFAULT_MODEL, tag: str = "",
                 private: bool = True, gen: Optional[dict] = None) -> dict:
    """Run one task through every stage with single online calls (cached on disk)."""
    from .llm import complete_json
    state = start(ctx, rec)
    while state["stage"] != "done":
        system, parts, schema = request(ctx, state)
        t0 = time.time()
        res = complete_json(parts[-1], images=parts[:-1], system=system, schema=schema,
                            backend="vertex", models=[model], temperature=TEMPERATURE,
                            private=private, tag=f"{tag}|{state['stage']}{state['b_round']}",
                            gen=gen or GEN, retries=3)
        usage = res.get("usage", {})
        if not res.get("cached"):
            log_spend(model, usage, f"{ctx.dataset}:{rec['id']}:{state['stage']}")
        advance(ctx, state, res["data"], {"model": res["model"], "usage": usage,
                                          "usd": round(usd(usage, model), 6),
                                          "cached": bool(res.get("cached")),
                                          "seconds": round(time.time() - t0, 1)})
    state["model"] = model
    state["usd"] = round(sum(c["usd"] for c in state["calls"]), 6)
    return state
