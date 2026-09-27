#!/usr/bin/env python3
"""Build the Public-expanded dataset (D5): real to-do concepts, written the way the
user writes tasks (longer titles, bodies, org markup), in EN / ES / RU.

1. Sample ~2,000 MS-LaTTE titles disjoint from Public-short (same stratified sampler),
   so the labeller never sees the same underlying item twice.
2. Triage (Gemini, free tier): 0–3 "is there substance worth expanding?" — drops
   "buy milk"-type items that no one would describe in a body.
3. Keep ~500 with score >= 2, stratified again by place.
4. Expand into the user's style. Each item gets a SHAPE drawn from the user's
   aggregate style statistics (empty / short / medium / long body, markup, links), and
   each batch sees 6 rotating style exemplars. The prompt never contains private text.
   The expansion must keep the core intent: `meta.concept` stores the original title.
5. Translate to ES and RU. One random display language per item; the others ride
   along in meta.parallel and show up as title translations in the labeller.

Steps 3–5 use Vertex Gemini Flash (public text only; paid trial credits, no daily cap).
Run: python scripts/datasets/build_public_expanded.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr.datasets import DATASETS, read_jsonl, write_jsonl  # noqa: E402
from tcr.llm import batched, complete_json  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_public_short import sample_mslatte, source_id  # noqa: E402

SEED = 20260927
N_POOL = 2000
N_KEEP = 500
EXEMPLARS = config.DATASETS_DIR / "style_exemplars.jsonl"
FLASH = ["gemini-3.5-flash", "gemini-2.5-flash"]

# Body shapes, weighted to follow the user's own distribution (style_stats: ~4% empty,
# median ~90 chars, p75 ~320, p90 ~560, max ~2,500; bold 19%, verbatim 18%,
# obsidian links 14%, bare URLs 13%, italic 6%, lists 2%).
SHAPES = [
    (4, "no body: return an empty body"),
    (38, "one short body sentence, 50–120 characters"),
    (28, "a body of 150–350 characters: motivation plus the concrete next step"),
    (16, "a body of 350–650 characters"),
    (5, "a long body of 700–1400 characters, possibly two paragraphs"),
    (4, "a body with a short lead sentence and a '- ' bullet list of 2–4 items"),
    (5, "a short body (60–150 characters) followed by a relevant well-known public URL alone on its own line"),
]
MARKUP = [
    (19, "use *bold* once on the key point"),
    (18, "use =verbatim= once for a name, file, place or number"),
    (14, "end the body with a reference to a personal note as [[obsidian:<Plausible Note Title>]]"),
    (6, "use /italic/ once"),
    (4, "put =verbatim= or *bold* markup in the TITLE"),
    (39, "no markup"),
]


# Short codes stored in meta.shape / meta.markup (the prompt text above is for the LLM).
SPEC_CODES = dict(zip([t for _, t in SHAPES],
                      ["empty", "short", "medium", "long", "very_long", "list", "short_url"]))
SPEC_CODES.update(zip([t for _, t in MARKUP],
                      ["bold", "verbatim", "obsidian_link", "italic", "title_markup", "none"]))


def weighted(rng, options):
    return rng.choices([o for _, o in options], weights=[w for w, _ in options])[0]


TRIAGE_SYSTEM = """You judge short to-do list entries written by real people.
Score each entry 0–3 for how much real substance it has for a longer description:
3 = a meaningful project, errand or decision with context worth writing down;
2 = a clear task a thoughtful person might add a sentence or two about;
1 = trivial or self-explanatory (a single shopping item, "laundry");
0 = unintelligible, a name only, or not a task."""

TRIAGE_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["id", "score"]}}}, "required": ["items"]}

EXPAND_SYSTEM = """You rewrite terse to-do entries into the personal task style shown in the
examples: a specific, natural title (usually 6–13 words) and, when asked, a body that
explains the motivation or the concrete next step, in the same voice. Markup is org-mode
(*bold*, /italic/, =verbatim=, [[obsidian:Note]], [[https://…][desc]]), never Markdown.

Hard rules:
- keep the CORE INTENT of the original entry: whatever icon best fits the original must
  still fit your version; add context, never change what the task is about;
- invent plausible, ordinary details (no real private persons; any names are fictional);
- web links must be real, well-known public pages;
- follow each item's SHAPE and MARKUP instructions exactly."""

ITEM_SCHEMA = {"type": "object", "properties": {"id": {"type": "string"}, "title": {"type": "string"},
                                                "body": {"type": "string"}},
               "required": ["id", "title", "body"]}
EXPAND_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": ITEM_SCHEMA}},
                 "required": ["items"]}

TRANSLATE_SYSTEM = """You translate personal to-do tasks from English into {lang_name}, as the same
person would write them natively in {lang_name}: natural, informal task phrasing, not a
literal translation. Keep org-mode markup (*bold*, /italic/, =verbatim=, links) and all
URLs exactly; translate the description part of [[url][description]] and the note title in
[[obsidian:…]] links. Keep empty bodies empty."""
LANG_NAMES = {"es": "Spanish", "ru": "Russian"}


def ids_match(want):
    return lambda d: {x["id"] for x in d.get("items", [])} == set(want)


def triage(items):
    scores = {}
    for batch in batched(items, 50):
        lines = "\n".join(json.dumps({"id": r["id"], "entry": r["title"]}, ensure_ascii=False)
                          for r in batch)
        res = complete_json("Score these entries:\n" + lines, system=TRIAGE_SYSTEM,
                            schema=TRIAGE_SCHEMA, backend="gemini", temperature=0.0,
                            validate=ids_match([r["id"] for r in batch]))
        scores.update({x["id"]: x["score"] for x in res["data"]["items"]})
        print(f"  triage {len(scores)}/{len(items)}", end="\r")
    print()
    return scores


def format_exemplars(exemplars):
    return "\n\n".join(f"TITLE: {e['title']}\nBODY:\n{e['body'] or '(empty)'}" for e in exemplars)


def expand(items, exemplars, rng):
    out = {}
    en_ex = [e for e in exemplars if e["lang"] == "en"]
    for bi, batch in enumerate(batched(items, 10)):
        shots = random.Random(bi).sample(en_ex, 6)
        lines = "\n".join(json.dumps({"id": r["id"], "entry": r["title"], "shape": r["shape"],
                                      "markup": r["markup"]}, ensure_ascii=False) for r in batch)
        prompt = (f"Style examples (same person, unrelated topics):\n\n{format_exemplars(shots)}\n\n"
                  f"Rewrite each entry below in that style, following its shape and markup:\n{lines}\n\n"
                  'Return JSON {"items": [{"id", "title", "body"}]}.')
        res = complete_json(prompt, system=EXPAND_SYSTEM, schema=EXPAND_SCHEMA, backend="vertex",
                            models=FLASH, temperature=0.8, validate=ids_match([r["id"] for r in batch]))
        out.update({x["id"]: x for x in res["data"]["items"]})
        print(f"  expand {len(out)}/{len(items)}", end="\r")
    print()
    return out


def translate(expanded, lang):
    out = {}
    items = list(expanded.values())
    for batch in batched(items, 10):
        lines = "\n".join(json.dumps({"id": x["id"], "title": x["title"], "body": x["body"]},
                                     ensure_ascii=False) for x in batch)
        res = complete_json("Translate these tasks:\n" + lines + '\nReturn JSON {"items": [{"id", "title", "body"}]}.',
                            system=TRANSLATE_SYSTEM.format(lang_name=LANG_NAMES[lang]),
                            schema=EXPAND_SCHEMA, backend="vertex", models=FLASH, temperature=0.3,
                            validate=ids_match([x["id"] for x in batch]))
        out.update({x["id"]: x for x in res["data"]["items"]})
        print(f"  translate {lang} {len(out)}/{len(items)}", end="\r")
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=N_POOL)
    ap.add_argument("--n-keep", type=int, default=N_KEEP)
    args = ap.parse_args()
    rng = random.Random(SEED)

    taken = {r["meta"]["source_id"] for r in read_jsonl(DATASETS["public_short"].path)
             if r["meta"]["source"] == "mslatte"}
    # Oversample, then drop anything already in Public-short.
    pool = [r for r in sample_mslatte(args.n_pool + len(taken), rng)
            if r["meta"]["source_id"] not in taken][:args.n_pool]
    print(f"pool: {len(pool)} MS-LaTTE titles disjoint from Public-short")

    scores = triage(pool)
    print("triage scores:", dict(sorted(Counter(scores.values()).items())))
    good = [r for r in pool if scores.get(r["id"], 0) >= 2]
    rng.shuffle(good)
    # Keep the place mix: round-robin over strata so no place dominates.
    by_place = {}
    for r in good:
        by_place.setdefault(r["meta"]["place"], []).append(r)
    kept = []
    while len(kept) < args.n_keep and any(by_place.values()):
        for place in sorted(by_place):
            if by_place[place] and len(kept) < args.n_keep:
                kept.append(by_place[place].pop())
    for r in kept:
        r["shape"], r["markup"] = weighted(rng, SHAPES), weighted(rng, MARKUP)
        if r["shape"].startswith("no body"):
            r["markup"] = "no markup"
    print(f"kept {len(kept)} of {len(good)} with score >= 2")

    exemplars = read_jsonl(EXEMPLARS)
    en = expand(kept, exemplars, rng)
    tr = {lang: translate(en, lang) for lang in LANG_NAMES}

    records = []
    for r in kept:
        x = en[r["id"]]
        parallel = {"en": {"title": x["title"].strip(), "body": x["body"].strip()}}
        for lang in LANG_NAMES:
            y = tr[lang][r["id"]]
            parallel[lang] = {"title": y["title"].strip(), "body": y["body"].strip()}
        lang = rng.choice(["en", "es", "ru"])
        records.append({
            "id": source_id("mslatte-exp", r["meta"]["source_id"]).replace("pub-", "exp-"),
            "dataset": "public_expanded", "title": parallel[lang]["title"],
            "body": parallel[lang]["body"], "lang": lang,
            "meta": {"source": "mslatte", "source_id": r["meta"]["source_id"], "concept": r["title"],
                     "list": r["meta"]["list"], "place": r["meta"]["place"],
                     "triage": scores[r["id"]], "shape": SPEC_CODES[r["shape"]],
                     "markup": SPEC_CODES[r["markup"]],
                     "parallel": parallel}})
    ds = DATASETS["public_expanded"]
    n = write_jsonl(ds.path, records)
    print(f"{n} tasks (display langs {dict(Counter(r['lang'] for r in records))}) -> {ds.path}")


if __name__ == "__main__":
    main()
