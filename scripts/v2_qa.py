#!/usr/bin/env python3
"""Pass 3 of the v2 icon descriptions: blind QA of glyph readings, v1 vs v2
(docs/icon-descriptions-v2.md §5 step 4 and §6 "glyph-reading accuracy").

Two samples, judged together, blind (v1 and v2 shown as slots A/B in an order
shuffled per glyph, in one common format):

- doubtful: v2-kept glyphs with recognizability ≤ 2; the glyphs kept only because the
  manual review overrode v2's discard; and the glyphs whose v1 and v2 readings
  disagree most (lowest e5 cosine). Wrong or partly right v2 readings are fixed by
  `redescribe`.
- random: 200 glyphs kept by v1 or v2, for the v1 vs v2 error rates with CIs.

  sample      pick the glyphs                          -> results/v2_qa/sample.json
  sheets      contact sheets + packets for the judge   -> results/v2_qa/judging/
  redescribe  re-describe v2 readings judged wrong or partly, with the judge's fix
  report      error rates and counts                   -> results/v2_qa/report.md

Judges (Opus subagents, one per range of packets) read judging/README.md and write
judgments_<n>.jsonl.
After `redescribe`: rerun the contrast for the touched groups
(describe_v2_pass2.py test --limit 999), then v2_guard.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import describe_v2 as dv2  # noqa: E402
from tcr.data import load_icons  # noqa: E402
from tcr.judging import contact_sheet, wilson  # noqa: E402

OUT = config.RESULTS_DIR / "v2_qa"
JUDGING = OUT / "judging"
SEED = "v2-qa-1"
N_RANDOM = 200
N_DISAGREE = 130
PER_SHEET = 10
MODEL = "gemini-4-argon"
QA_VERSION = "v2.0-qa"


def read_v1(name):
    return json.loads((config.ICON_DESC_DIR / f"{name}.json").read_text(encoding="utf-8"))


def read_v2(name):
    return json.loads((config.ICON_DESC_V2_DIR / f"{name}.json").read_text(encoding="utf-8"))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# --- sample --------------------------------------------------------------------------

def cmd_sample(args):
    v1 = {ic.name for ic in load_icons(version="v1")}
    v2_all = {ic.name: ic for ic in load_icons(include_discarded=True, version="v2")}
    v2 = {ic.name for ic in load_icons(version="v2")}
    both = sorted(n for n in v2_all if (n in v1 or n in v2)
                  and (config.ICON_DESC_DIR / f"{n}.json").exists())
    low_rec = sorted(n for n in v2 if v2_all[n].scores.get("recognizability", 5) <= 2)
    override = sorted(n for n in v2 if read_v2(n)["discard"])

    # v1 vs v2 disagreement: what is drawn + what it means, in each version.
    from tcr.index import embed_texts
    cand = sorted(v2 - set(low_rec) - set(override))
    cand = [n for n in cand if (config.ICON_DESC_DIR / f"{n}.json").exists()]
    t1 = [", ".join(read_v1(n).get("visual_concepts", [])[:8] + read_v1(n).get("task_intents", [])[:6])
          for n in cand]
    t2 = [", ".join(v2_all[n].visual_concepts + v2_all[n].readings) for n in cand]
    E1 = embed_texts(config.MULTILINGUAL_MODEL, t1)
    E2 = embed_texts(config.MULTILINGUAL_MODEL, t2)
    sim = (E1 * E2).sum(axis=1)
    disagree = [cand[i] for i in sim.argsort()[:N_DISAGREE]]

    rng = random.Random(SEED)
    rand = sorted(rng.sample(both, N_RANDOM))
    items = {}
    for why, names in (("low_recognizability", low_rec), ("review_override", override),
                       ("v1_v2_disagree", disagree), ("random", rand)):
        for n in names:
            items.setdefault(n, []).append(why)
    sample = [{"icon": n, "why": w} for n, w in sorted(items.items())]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
    c = Counter(w for s in sample for w in s["why"])
    print(f"{len(sample)} glyphs: {dict(c)}")


def load_sample():
    return json.loads((OUT / "sample.json").read_text(encoding="utf-8"))


# --- sheets --------------------------------------------------------------------------

def slot_v1(name):
    d = read_v1(name)
    return {"shows": ", ".join(d.get("visual_concepts", [])[:10]),
            "means": "; ".join(d.get("task_intents", [])[:6]),
            "note": d.get("reasoning", "")}


def slot_v2(name):
    d = read_v2(name)
    dep = d["depiction"]
    shows = dep["primary_object"] + ("; modifiers: " + "; ".join(dep["modifiers"]) if dep.get("modifiers") else "")
    return {"shows": shows,
            "means": "; ".join(r["meaning"] for r in d["readings"]),
            "note": d.get("reasoning", "")}


README = """# Blind QA: glyph readings

Each glyph (q001 …) has two descriptions, slots A and B, from two different
description sets; which is which is shuffled per glyph. Look at the glyph on its
contact sheet (`sheet_NN.png`, labelled in red), then judge each slot. Judge what is
DRAWN and how a person would read it. You are not told the icon's official name and
it does not matter: a reading most people would share is correct even if the font
names the glyph differently. Small details (a corner badge, a slash, a fill, the
number of bars) change the meaning; look closely.

For each glyph and slot, a verdict on `shows` + `means` together:

- `correct`: the depiction is right and the main meaning is what most people would
  take it to mean (or a fair reading of a genuinely ambiguous glyph).
- `partly`: the main object is right but a meaningful modifier is missed or misread,
  or the main meaning is weak while a secondary one is right.
- `wrong`: the main object is misidentified, or the meanings would send the icon to the
  wrong tasks.

For `partly` and `wrong`, `fix` says in one or two sentences what is actually drawn
and what it would mean to most people. `note` is optional otherwise.

Write one JSON line per (glyph, slot) to the judgments file named in your
instructions (`results/v2_qa/judgments_<n>.jsonl`; append, one line at a time):
`{"glyph": "q007", "slot": "B", "verdict": "partly", "fix": "a bell with a small slash: notifications off", "note": ""}`

Do NOT open `results/v2_qa/blind_key.json`, `sample.json`, or any file under `data/`.
"""


def cmd_sheets(args):
    sample = load_sample()
    rng = random.Random(SEED + "-blind")
    order = sample[:]
    rng.shuffle(order)
    JUDGING.mkdir(parents=True, exist_ok=True)
    for old in JUDGING.glob("*"):
        old.unlink()
    key = {}
    for i, it in enumerate(order):
        gid = f"q{i + 1:03d}"
        versions = ["v1", "v2"]
        rng.shuffle(versions)
        key[gid] = {"icon": it["icon"], "slots": dict(zip("AB", versions)), "why": it["why"]}
    gids = list(key)
    for b in range(0, len(gids), PER_SHEET):
        chunk = gids[b:b + PER_SHEET]
        n = b // PER_SHEET + 1
        contact_sheet([(g, key[g]["icon"]) for g in chunk], JUDGING / f"sheet_{n:02d}.png", px=256, cols=5)
        lines = [f"# Packet {n:02d} (glyphs on sheet_{n:02d}.png)\n"]
        for g in chunk:
            lines.append(f"## {g}\n")
            for s, v in key[g]["slots"].items():
                d = (slot_v1 if v == "v1" else slot_v2)(key[g]["icon"])
                lines.append(f"**{s}**\n- shows: {d['shows']}\n- means: {d['means']}\n"
                             f"- note: {d['note']}\n")
        (JUDGING / f"packet_{n:02d}.md").write_text("\n".join(lines), encoding="utf-8")
    (JUDGING / "README.md").write_text(README, encoding="utf-8")
    (OUT / "blind_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(gids)} glyphs in {(len(gids) - 1) // PER_SHEET + 1} packets -> {JUDGING}")


# --- results -------------------------------------------------------------------------

def verdicts():
    """{(icon, version): judgment} — the last judgment per (glyph, slot) wins."""
    key = json.loads((OUT / "blind_key.json").read_text(encoding="utf-8"))
    out = {}
    for j in (j for p in sorted(OUT.glob("judgments*.jsonl")) for j in read_jsonl(p)):
        k = key.get(j["glyph"])
        if k and j.get("slot") in k["slots"]:
            out[(k["icon"], k["slots"][j["slot"]])] = {**j, "why": k["why"]}
    return key, out


def cmd_redescribe(args):
    """Re-describe the glyphs whose v2 reading was judged wrong or partly, with the
    judge's fix as a reviewer note (synchronous Argon calls)."""
    from tcr.data import glyph_hashes
    from tcr.llm import LLMError, complete_json
    _, v = verdicts()
    todo = sorted(n for (n, ver), j in v.items() if ver == "v2" and j["verdict"] in ("wrong", "partly"))
    if args.limit:
        todo = todo[:args.limit]
    hashes = glyph_hashes()
    aliases = defaultdict(list)
    for p in config.ICON_DESC_V2_DIR.glob("*.json"):
        a = json.loads(p.read_text(encoding="utf-8"))["_meta"].get("alias_of")
        if a:
            aliases[a].append(p.stem)
    print(f"{len(todo)} glyphs to re-describe")

    def one(name):
        j = v[(name, "v2")]
        system, prompt, images = dv2.build_request(name, key=hashes[name])
        old = read_v2(name)
        prompt += ("\n\nA reviewer checked an earlier description of this icon against the image "
                   f"and found it {j['verdict']}. The earlier description read it as: "
                   f"{old['depiction']['primary_object']}; meanings: "
                   + "; ".join(r["meaning"] for r in old["readings"])
                   + f". The reviewer says: {j['fix']}\nLook at the image again carefully and "
                   "describe it from scratch; follow the reviewer only where the image confirms it.")
        res = complete_json(prompt, system=system, schema=dv2.RESPONSE_SCHEMA, backend="vertex",
                            models=[MODEL], temperature=1.0, images=images,
                            validate=lambda d: not dv2.hard_problems(d), retries=4, tag="qa")
        data = res["data"]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for fname in [name] + sorted(aliases.get(name, [])):
            prev = read_v2(fname)["_meta"]
            meta = {"model": MODEL, "prompt_version": dv2.PROMPT_VERSION, "pass": "describe",
                    "passes": ["describe", "qa"], "mode": "online", "glyph": hashes[name],
                    "usage": res.get("usage"), "soft_problems": dv2.soft_problems(data),
                    "created": now, "alias_of": prev.get("alias_of"),
                    "qa": {"version": QA_VERSION, "verdict": j["verdict"], "fix": j["fix"],
                           "replaced": {"depiction": old["depiction"], "readings": old["readings"],
                                        "passes": prev.get("passes") or ["describe"]}}}
            doc = {"schema_version": 2, **data, "_meta": meta}
            (config.ICON_DESC_V2_DIR / f"{fname}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        return data

    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, n): n for n in todo}
        for f in as_completed(futs):
            n = futs[f]
            try:
                d = f.result()
                print(f"ok {n}: {d['depiction']['primary_object']} | "
                      + "; ".join(r["meaning"] for r in d["readings"][:2]))
            except LLMError as e:
                print(f"FAILED {n}: {str(e)[:120]}")


def cmd_report(args):
    key, v = verdicts()
    n_glyphs = len(key)
    judged = {k[0] for k in v}
    L = ["# Blind QA of glyph readings: v1 vs v2", "",
         f"{n_glyphs} glyphs, {len(v)} of {2 * n_glyphs} (glyph, version) judgments. Judge: a "
         "Claude Opus subagent, blind to the version (slots shuffled per glyph, one common "
         "format). v2 is judged after the contrast pass and before `redescribe`.", ""]
    for title, sel in (("Random sample (kept by v1 or v2)", "random"),
                       ("Doubtful glyphs", None)):
        rows = [k for k in key.values()
                if (sel in k["why"] if sel else set(k["why"]) - {"random"})]
        L += [f"## {title}: {len(rows)} glyphs", "",
              "| Version | correct | partly | wrong | wrong, 95% CI | wrong or partly, 95% CI |",
              "|---|---|---|---|---|---|"]
        for ver in ("v1", "v2"):
            c = Counter(v[(k["icon"], ver)]["verdict"] for k in rows if (k["icon"], ver) in v)
            n = sum(c.values())
            w, wp = wilson(c["wrong"], n), wilson(c["wrong"] + c["partly"], n)
            L.append(f"| {ver} | {c['correct']} | {c['partly']} | {c['wrong']} | "
                     f"{c['wrong'] / max(n, 1):.1%} ({w[0]:.1%}–{w[1]:.1%}) | "
                     f"{(c['wrong'] + c['partly']) / max(n, 1):.1%} ({wp[0]:.1%}–{wp[1]:.1%}) |")
        L.append("")
    by_why = defaultdict(Counter)
    for k in key.values():
        if (k["icon"], "v2") in v:
            for w in k["why"]:
                by_why[w][v[(k["icon"], "v2")]["verdict"]] += 1
    L += ["## v2 verdicts by reason for sampling", "", "| Reason | correct | partly | wrong |",
          "|---|---|---|---|"]
    for w, c in sorted(by_why.items()):
        L.append(f"| {w} | {c['correct']} | {c['partly']} | {c['wrong']} |")
    fixed = [n for n in judged if (read_v2(n)["_meta"].get("qa") or {}).get("version") == QA_VERSION]
    L += ["", f"Re-described after QA: {len(fixed)} glyphs (v2 judged wrong or partly).", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "sheets", "redescribe", "report"])
    ap.add_argument("--limit", type=int, default=0, help="redescribe: only the first N")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
