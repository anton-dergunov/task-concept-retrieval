#!/usr/bin/env python3
"""Build the Public-short dataset (D4): short real-world to-do titles.

  ~600 MS-LaTTE titles  real Wunderlist tasks. Stratified by the crowd-judged place
                        where a task gets done (home / work / a public-place type),
                        with square-root allocation so the long tail of topics is
                        represented and huge strata (groceries) are down-weighted.
  ~300 MASSIVE items    calendar_set / lists_createoradd voice commands in parallel
                        en-US / es-ES / ru-RU, rewritten by Gemini into to-do titles
                        per language. Each group is shown in ONE random language;
                        the other two ride along in meta.parallel so a single label
                        yields a three-language eval item.

Run after fetch_public.py:  python scripts/datasets/build_public_short.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr.datasets import DATASETS, write_jsonl  # noqa: E402
from tcr.llm import GEMINI_MODELS, batched, complete_json  # noqa: E402

SEED = 20260926
N_MSLATTE = 600
N_MASSIVE = 300
MASSIVE_INTENTS = {"calendar_set": 0.75, "lists_createoradd": 0.25}  # share of N_MASSIVE
LANGS = {"en": "en-US", "es": "es-ES", "ru": "ru-RU"}
_JUNK_TITLES = {"to do", "todo", "list", "stuff", "things", "misc", "inbox", "to do list"}


# MS-LaTTE list names are the tasks' natural parent heading ("groceries", "home depot",
# "house projects"), except these generic ones.
_UNINFORMATIVE_LISTS = {"default list", "to do", "todo", "to do list", "to-do", "to-do list",
                        "today", "to do today", "things to do", "tasks", "inbox", "list",
                        "my list", "reminders", "misc", "tomorrow", "this week", "stuff",
                        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
                        "sunday", "weekend"}


def list_parent(list_title: str):
    t = re.sub(r"\s+", " ", list_title).strip()
    if not t or t.casefold() in _UNINFORMATIVE_LISTS or not re.search(r"[a-z]{3}", t.casefold()):
        return []
    return [t[:1].upper() + t[1:]]


def source_id(source: str, sid: str) -> str:
    return "pub-" + hashlib.sha1(f"{source}:{sid}".encode()).hexdigest()[:10]


# --- MS-LaTTE ------------------------------------------------------------------

def _majority(values):
    values = [v for v in values if v]
    return Counter(values).most_common(1)[0][0] if values else ""


def location_stratum(rec: dict) -> str:
    known = [j for j in rec.get("LocJudgements", []) if j.get("Known") == "yes"]
    loc = _majority(j.get("Locations", "") for j in known)
    if loc == "public":
        place = _majority(j.get("PublicLocations", "") for j in known)
        return f"public:{place or 'other'}"
    return loc or "unknown"


def clean_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t[:1].upper() + t[1:]


def sample_mslatte(n: int, rng: random.Random):
    recs = json.loads((config.RAW_DIR / "MS-LaTTE.json").read_text(encoding="utf-8"))
    seen, pool = set(), []
    for r in recs:
        title = clean_title(r["TaskTitle"])
        key = title.casefold()
        if (len(title) < 3 or not re.search(r"[a-z]{2}", key) or key in _JUNK_TITLES
                or re.match(r"^\d", title)       # "0 cucumbers": number-replacement artifacts
                or key in seen):
            continue
        seen.add(key)
        pool.append((location_stratum(r), title, r))

    strata = defaultdict(list)
    for s, title, r in pool:
        strata[s].append((title, r))
    # Square-root allocation: a compromise between proportional and uniform.
    weights = {s: math.sqrt(len(v)) for s, v in strata.items()}
    total = sum(weights.values())
    alloc = {s: min(len(strata[s]), max(1, round(n * w / total))) for s, w in weights.items()}
    picked = []
    for s in sorted(strata):
        picked += [(s, t, r) for t, r in rng.sample(strata[s], alloc[s])]
    # Rounding leaves the total a little off n: top up uniformly or trim.
    chosen = {t for _, t, _ in picked}
    rest = [(s, t, r) for s, t, r in pool if t not in chosen]
    picked += rng.sample(rest, max(0, n - len(picked)))
    rng.shuffle(picked)
    picked = picked[:n]

    out = [{"id": source_id("mslatte", r["ID"]), "dataset": "public_short", "title": t, "body": "",
            "parents": list_parent(r["ListTitle"]), "lang": "en", "meta": {"source": "mslatte", "source_id": r["ID"],
                                   "list": r["ListTitle"], "place": s}}
           for s, t, r in picked]
    print(f"MS-LaTTE: {len(pool)} clean titles in {len(strata)} strata -> {len(out)} sampled")
    return out


# --- MASSIVE -------------------------------------------------------------------

REWRITE_SYSTEM = """You turn voice-assistant commands into to-do list entries.
For each item you get the SAME command in English (en), Spanish (es) and Russian (ru).
Rewrite each language's command, in that language, into the task title a person would
jot into their own to-do list:
- imperative or noun phrase, as short as a real to-do entry (usually 2-7 words);
- remove the assistant framing ("remind me to", "add … to my list", "set up a meeting");
- remove dates, times, recurrence and alarm details;
- keep the substance: people, places, objects, the actual activity;
- adding items to a shopping list becomes "Buy …" (in the item's language);
- use natural phrasing in each language; do not translate from English;
- start with a capital letter, no trailing period.
If the command has no concrete task content (e.g. "set a reminder for 5pm",
"create a new list"), set keep=false and leave the titles empty."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "keep": {"type": "boolean"},
                       "en": {"type": "string"}, "es": {"type": "string"}, "ru": {"type": "string"}},
        "required": ["id", "keep", "en", "es", "ru"]}}},
    "required": ["items"],
}


def load_massive():
    by_loc = {}
    for short, loc in LANGS.items():
        rows = [json.loads(l) for l in open(config.RAW_DIR / "massive" / f"{loc}.jsonl", encoding="utf-8")]
        by_loc[short] = {r["id"]: r for r in rows}
    ids = [i for i, r in by_loc["en"].items()
           if r["intent"] in MASSIVE_INTENTS and all(i in by_loc[l] for l in LANGS)]
    return by_loc, ids


def sample_massive(n: int, rng: random.Random):
    by_loc, ids = load_massive()
    groups = []
    for intent, share in MASSIVE_INTENTS.items():
        cand = sorted(i for i in ids if by_loc["en"][i]["intent"] == intent)
        # Oversample ~30%: some commands carry no task content and get dropped.
        groups += rng.sample(cand, min(len(cand), round(n * share * 1.3)))

    rewrites = {}
    for batch in batched(groups, 10):
        lines = [json.dumps({"id": i, **{l: by_loc[l][i]["utt"] for l in LANGS}}, ensure_ascii=False)
                 for i in batch]
        want = set(batch)
        res = complete_json(
            "Rewrite these commands:\n" + "\n".join(lines), system=REWRITE_SYSTEM,
            schema=REWRITE_SCHEMA, backend="gemini", models=GEMINI_MODELS, temperature=0.2,
            validate=lambda d: {x["id"] for x in d["items"]} == want)
        for x in res["data"]["items"]:
            rewrites[x["id"]] = x
        print(f"  MASSIVE rewrites: {len(rewrites)}/{len(groups)}", end="\r")
    print()

    out, seen = [], set()
    for i in groups:
        x = rewrites[i]
        parallel = {l: clean_title(x[l]) for l in LANGS}
        if not x["keep"] or not all(parallel.values()):
            continue
        lang = rng.choice(sorted(LANGS))
        key = parallel["en"].casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": source_id("massive", i), "dataset": "public_short",
                    "title": parallel[lang], "body": "", "lang": lang,
                    "meta": {"source": "massive", "source_id": i,
                             "intent": by_loc["en"][i]["intent"],
                             "utt": {l: by_loc[l][i]["utt"] for l in LANGS},
                             "parallel": parallel}})
    # Keep the intent mix after drops, then cap.
    per_intent = defaultdict(list)
    for r in out:
        per_intent[r["meta"]["intent"]].append(r)
    capped = []
    for intent, share in MASSIVE_INTENTS.items():
        capped += per_intent[intent][:round(n * share)]
    print(f"MASSIVE: {len(groups)} groups -> {len(out)} kept -> {len(capped)} "
          f"(display langs {dict(Counter(r['lang'] for r in capped))})")
    return capped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mslatte", type=int, default=N_MSLATTE)
    ap.add_argument("--n-massive", type=int, default=N_MASSIVE)
    args = ap.parse_args()

    rng = random.Random(SEED)
    records = sample_mslatte(args.n_mslatte, rng) + sample_massive(args.n_massive, rng)
    rng.shuffle(records)
    ds = DATASETS["public_short"]
    n = write_jsonl(ds.path, records)
    print(f"{n} tasks -> {ds.path}")


if __name__ == "__main__":
    main()
