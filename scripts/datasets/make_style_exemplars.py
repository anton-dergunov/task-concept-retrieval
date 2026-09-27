#!/usr/bin/env python3
"""Write fictional tasks in the user's style: the few-shot set for the
Personal-synth and Public-expanded generators.

The model sees aggregate style stats plus a handful of real tasks as a STYLE-ONLY
reference, and must write about unrelated topics. Only the generated exemplars
are committed (after the user reviews them), so later generation prompts never
contain private text. Exemplars sharing any 5-word sequence with a private task
are rejected.

Backend: claude-cli (private text → never the free Gemini tier).
Run: python scripts/datasets/make_style_exemplars.py [--n 12] [--model opus]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr.datasets import DATASETS, read_jsonl, write_jsonl  # noqa: E402
from tcr.llm import complete_json  # noqa: E402

OUT = config.DATASETS_DIR / "style_exemplars.jsonl"
STYLE_STATS = config.PRIVATE_DIR / "style_stats.json"

# One slot per exemplar: an everyday topic (never the user's own) and a SHAPE, so the
# few-shot set covers the whole range of the user's habits, not only the typical task.
SLOTS = [
    ("gardening", "en", "title only — the body is EMPTY"),
    ("cycling", "en", "deliberately terse: 2–4 word title, one short body sentence"),
    ("home repair", "en", "multi-line body: one lead sentence, then a '- ' bullet list of 3–5 items"),
    ("cooking", "en", "title contains =verbatim= markup (e.g. a file or tool name)"),
    ("learning an instrument", "en", "very long body, 1500–2200 characters, 2–4 paragraphs"),
    ("astronomy", "en", "medium body ending with a bare URL alone on its last line"),
    ("car maintenance", "en", "body with an org link that has a description: [[https://…][description]]"),
    ("volunteering", "es", "written entirely in Spanish, medium body"),
    ("photography", "ru", "written entirely in Russian, short body"),
    ("board games", "en", "title contains *bold* markup on the key word"),
    ("birdwatching", "en", "body uses /italic/ and ends with a [[obsidian:Note Title]] reference"),
    ("personal finance paperwork", "en",
     "body contains a small #+begin_src … #+end_src block (a shell command or a few lines of Python)"),
    ("hiking trip planning", "en", "two short paragraphs separated by a blank line, with a bare URL"),
    ("woodworking", "en", "title contains /italic/ markup; body uses =verbatim= for a measurement or part"),
    ("pet care", "en", "short body with an [[obsidian:…]] link"),
    ("local history", "en", "medium plain-prose body, no markup"),
]

SYSTEM = """You write fictional to-do tasks that imitate one person's WRITING STYLE while
sharing none of their CONTENT. The reference tasks are private: never reuse their topics,
names, projects, links, or any phrase of four or more words from them."""

SCHEMA = {"type": "object", "properties": {"tasks": {"type": "array", "items": {
    "type": "object", "properties": {"topic": {"type": "string"}, "title": {"type": "string"},
                                     "body": {"type": "string"}},
    "required": ["topic", "title", "body"]}}}, "required": ["tasks"]}


def words(text: str):
    return re.findall(r"[^\W_]+", text.casefold())


_URL_RE = re.compile(r"https?://\S+")


def urls(text: str):
    return {u.rstrip(".,;)") for u in _URL_RE.findall(text)}


def ngrams(text: str, n: int = 5):
    """Word n-grams of the prose; URLs are compared whole (see `urls`), since
    "https en wikipedia org wiki" would otherwise match every Wikipedia link."""
    w = words(_URL_RE.sub(" ", text))
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def pick_references(records, rng, k=8):
    """Style references spread over body lengths and markup habits."""
    def has(r, pat):
        return re.search(pat, r["title"] + "\n" + r["body"])
    buckets = [
        lambda r: 0 < len(r["body"]) <= 100,
        lambda r: 100 < len(r["body"]) <= 350,
        lambda r: 350 < len(r["body"]) <= 900,
        lambda r: has(r, r"\[\[obsidian:"),
        lambda r: has(r, r"https?://"),
        lambda r: has(r, r"(?<!\w)\*\S[^*]*\*(?!\w)"),
        lambda r: has(r, r"(?<!\w)=\S[^=]*=(?!\w)"),
        lambda r: len(r["body"]) <= 100,
    ]
    refs = []
    for pred in buckets[:k]:
        cand = [r for r in records if pred(r) and r not in refs]
        refs.append(rng.choice(cand))
    return refs


def build_prompt(stats, refs, slots):
    fr = stats["feature_rates"]
    ref_block = "\n\n".join(f"TITLE: {r['title']}\nBODY:\n{r['body'] or '(empty)'}" for r in refs)
    slot_block = "\n".join(f"{i + 1}. topic: {t}; language: {lang}; shape: {shape}"
                           for i, (t, lang, shape) in enumerate(slots))
    return f"""Style statistics of the person's ~{stats['n_tasks']} tasks:
- title length in words: median {stats['title_words']['median']}, 10th–90th percentile {stats['title_words']['p10']}–{stats['title_words']['p90']}
- {100 - round(stats['share_empty_body'] * 100)}% of tasks have a body; body length in characters: median {stats['body_chars_nonempty']['median']}, 75th pct {stats['body_chars_nonempty']['p75']}, 90th pct {stats['body_chars_nonempty']['p90']}, max ~{stats['body_chars_nonempty']['max']}
- share of tasks using: *bold* {fr['bold']:.0%}, =verbatim= {fr['verbatim']:.0%}, [[obsidian:Note Title]] links {fr['obsidian_link']:.0%}, bare URLs {fr['bare_url']:.0%}, /italic/ {fr['italic']:.0%}, lists {fr['list']:.0%}; markup appears in titles too
- markup is org-mode (not Markdown); a bare URL usually sits alone on its own line

Reference tasks (style only — private, do not reuse content):

{ref_block}

Write exactly one task per slot below, in this order. Each slot fixes the topic, the
language and a SHAPE the task must have. Otherwise match the person's style: title
phrasing, how the body explains motivation or the concrete next step, sentence rhythm.
Web links must be real, well-known public URLs (e.g. Wikipedia); obsidian links use a
plausible note title. Non-English tasks must read as written by the same person
natively in that language (not translated from English).

{slot_block}

Return JSON: {{"tasks": [{{"topic", "title", "body"}}]}} with body "" when it is empty."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opus")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tries", type=int, default=3, help="re-asks for slots that failed the leak check")
    args = ap.parse_args()

    private = read_jsonl(DATASETS["personal"].path)
    stats = json.loads(STYLE_STATS.read_text())
    refs = pick_references(private, random.Random(args.seed))
    private_grams, private_urls = set(), set()
    for r in private:
        private_grams |= ngrams(r["title"] + " " + r["body"])
        private_urls |= urls(r["body"])

    done = {}
    for attempt in range(args.tries):
        todo = [i for i in range(len(SLOTS)) if i not in done]
        if not todo:
            break
        slots = [SLOTS[i] for i in todo]
        res = complete_json(build_prompt(stats, refs, slots), system=SYSTEM, schema=SCHEMA,
                            backend="claude-cli", models=[args.model], private=True,
                            tag=f"try{attempt}" if attempt else "",
                            validate=lambda d: len(d.get("tasks", [])) == len(slots))
        for i, t in zip(todo, res["data"]["tasks"]):
            text = t["title"] + " " + t["body"]
            if ngrams(text) & private_grams or urls(text) & private_urls:
                print(f"  slot {i + 1} ({SLOTS[i][0]}): overlaps private text, re-asking")
                continue
            topic, lang, shape = SLOTS[i]
            done[i] = {"id": f"sty-{i + 1:02d}", "topic": topic, "lang": lang, "shape": shape,
                       "title": t["title"].strip(), "body": t["body"].strip(), "model": res["model"]}
    out = [done[i] for i in sorted(done)]
    write_jsonl(OUT, out)
    print(f"{len(out)}/{len(SLOTS)} exemplars -> {OUT}")


if __name__ == "__main__":
    main()
