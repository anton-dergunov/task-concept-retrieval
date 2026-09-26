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

# Everyday areas for the exemplars; the prompt also forbids the reference topics.
TOPICS = ["gardening", "cycling", "home repair", "cooking", "learning an instrument",
          "astronomy", "car maintenance", "volunteering", "photography", "board games",
          "birdwatching", "personal finance paperwork", "hiking trip planning",
          "woodworking", "pet care", "local history"]

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


def build_prompt(stats, refs, n):
    fr = stats["feature_rates"]
    ref_block = "\n\n".join(f"TITLE: {r['title']}\nBODY:\n{r['body'] or '(empty)'}" for r in refs)
    return f"""Style statistics of the person's ~{stats['n_tasks']} tasks:
- title length in words: median {stats['title_words']['median']}, 10th–90th percentile {stats['title_words']['p10']}–{stats['title_words']['p90']}
- {100 - round(stats['share_empty_body'] * 100)}% of tasks have a body; body length in characters: median {stats['body_chars_nonempty']['median']}, 75th pct {stats['body_chars_nonempty']['p75']}, 90th pct {stats['body_chars_nonempty']['p90']}; usually one paragraph
- share of tasks using: *bold* {fr['bold']:.0%}, =verbatim= {fr['verbatim']:.0%}, [[obsidian:Note Title]] links {fr['obsidian_link']:.0%}, bare URLs {fr['bare_url']:.0%}, /italic/ {fr['italic']:.0%}, lists {fr['list']:.0%}
- markup is org-mode (not Markdown)

Reference tasks (style only — private, do not reuse content):

{ref_block}

Write {n} new tasks, one per topic, on these topics: {", ".join(TOPICS[:n])}.
Match the style: title length and phrasing, how the body explains motivation or the
concrete next step, sentence rhythm, and markup/link habits at roughly the rates above.
Body lengths should follow the distribution above (several short ~60–110 chars, a few
~200–350, one or two ~450–650). Links: use [[obsidian:<Plausible Note Title>]] for
internal notes and real, well-known public URLs (e.g. Wikipedia) for web links; the
person usually puts a bare URL on its own line at the end of the body — do that in at
least two tasks.
English only. Return JSON: {{"tasks": [{{"topic", "title", "body"}}]}}."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    private = read_jsonl(DATASETS["personal"].path)
    stats = json.loads(STYLE_STATS.read_text())
    refs = pick_references(private, random.Random(args.seed))
    res = complete_json(build_prompt(stats, refs, args.n), system=SYSTEM, schema=SCHEMA,
                        backend="claude-cli", models=[args.model], private=True,
                        validate=lambda d: len(d.get("tasks", [])) >= args.n)

    private_grams, private_urls = set(), set()
    for r in private:
        private_grams |= ngrams(r["title"] + " " + r["body"])
        private_urls |= urls(r["body"])
    out, rejected = [], 0
    for i, t in enumerate(res["data"]["tasks"]):
        text = t["title"] + " " + t["body"]
        if ngrams(text) & private_grams or urls(text) & private_urls:
            rejected += 1
            continue
        out.append({"id": f"sty-{i:02d}", "topic": t["topic"], "title": t["title"].strip(),
                    "body": t["body"].strip(), "model": res["model"]})
    write_jsonl(OUT, out)
    print(f"{len(out)} exemplars ({rejected} rejected for 5-gram overlap) -> {OUT}")


if __name__ == "__main__":
    main()
