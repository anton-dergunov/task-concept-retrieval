#!/usr/bin/env python3
"""Statistics of the v2 icon descriptions next to v1 (docs/icon-descriptions.md §6).

One description per distinct glyph (font aliases collapsed). The text statistics are over
each version's KEPT glyphs (discard = false):

- counts, discard × the manual review level, v2 scores, soft problems, guard drops, cost;
- example tasks: count, length mix vs the 5 / 8 / 3 target, cross-icon reuse;
- negatives (v1 poor_matches, v2 near_misses): share taken by the 20 commonest strings;
- v2 domains: entropy of the example-task domains across the run.

Writes results/v2_stats/report.md and stats.json. Run: python scripts/v2_report.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import describe_v2 as dv2  # noqa: E402
from tcr.data import glyph_hashes, icon_labels, load_icons  # noqa: E402

OUT = config.RESULTS_DIR / "v2_stats"
V2_DIR = config.DATA_DIR / "icon_descriptions_v2"
PRICE = (1.00, 5.00)   # batch USD / 1M tokens (input, output incl. thinking), promo


def level_by_glyph():
    H = glyph_hashes()
    return {H.get(n): lab for n, lab in icon_labels().items()}


def load_v1():
    """{name: (discard, examples, negatives)} — one name per glyph."""
    out = {}
    for ic in load_icons(include_discarded=True, version="v1"):
        d = json.loads((config.ICON_DESC_DIR / f"{ic.name}.json").read_text(encoding="utf-8"))
        out[ic.name] = {"discard": bool(d.get("discard")), "examples": ic.example_tasks,
                        "negatives": ic.poor_matches}
    return out


def load_v2():
    out = {}
    for p in sorted(V2_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["_meta"].get("alias_of"):
            continue
        out[p.stem] = {"discard": d["discard"], "examples": [e["task"] for e in d["example_tasks"]],
                       "domains": [e.get("domain") for e in d["example_tasks"]],
                       "negatives": [n["task"] for n in d["near_misses"]], "doc": d}
    return out


def text_stats(rows):
    kept = [r for r in rows.values() if not r["discard"]]
    ex = [t for r in kept for t in r["examples"]]
    words = [dv2.words(t) for t in ex]
    buckets = Counter("2-5" if 2 <= w <= 5 else "6-12" if 6 <= w <= 12
                      else "13-16" if 13 <= w <= 16 else "other" for w in words)
    owners = Counter()
    for r in kept:
        for t in {dv2.norm(t) for t in r["examples"]}:
            owners[t] += 1
    neg = Counter(dv2.norm(t) for r in kept for t in r["negatives"])
    n_neg = sum(neg.values())
    return {
        "kept glyphs": len(kept),
        "example tasks": len(ex),
        "examples per glyph (median)": statistics.median(len(r["examples"]) for r in kept),
        "words per example (median / p90)": (statistics.median(words),
                                             sorted(words)[int(0.9 * len(words))]),
        "length mix 2-5 / 6-12 / 13-16 / other": tuple(
            round(buckets[b] / len(ex), 3) for b in ("2-5", "6-12", "13-16", "other")),
        "distinct examples": len(owners),
        "examples used by >1 glyph": sum(1 for c in owners.values() if c > 1),
        "most reused example (glyphs)": max(owners.values()),
        "negatives": n_neg,
        "distinct negatives": len(neg),
        "top-20 negatives' share": round(sum(c for _, c in neg.most_common(20)) / n_neg, 3),
    }


def discard_table(rows, levels):
    H = glyph_hashes()
    t = Counter((levels.get(H.get(n), "unsure"), r["discard"]) for n, r in rows.items())
    return {lev: {"kept": t[(lev, False)], "discarded": t[(lev, True)]}
            for lev in ("useful", "somewhat", "unsure", "reject")}


def entropy(counts):
    n = sum(counts.values())
    h = -sum(c / n * math.log2(c / n) for c in counts.values() if c)
    return h, h / math.log2(len(dv2.DOMAINS))


def main():
    levels = level_by_glyph()
    v1, v2 = load_v1(), load_v2()
    docs = [r["doc"] for r in v2.values()]
    s1, s2 = text_stats(v1), text_stats(v2)
    dom = Counter(x for r in v2.values() if not r["discard"] for x in r["domains"])
    h, hn = entropy(dom)
    usage = Counter()
    for d in docs:
        for k in ("prompt", "output", "thoughts"):
            usage[k] += (d["_meta"].get("usage") or {}).get(k, 0) or 0
    cost = (usage["prompt"] * PRICE[0] + (usage["output"] + usage["thoughts"]) * PRICE[1]) / 1e6
    soft = Counter(p.split(":")[0] for d in docs for p in d["_meta"]["soft_problems"])
    n_soft = sum(1 for d in docs if d["_meta"]["soft_problems"])
    guard = Counter()
    for d in docs:
        guard.update(((d["_meta"].get("guard") or {}).get("dropped") or {}))
    scores = {k: Counter(d["scores"][k] for d in docs if not d["discard"])
              for k in ("recognizability", "task_fit", "distinctiveness")}
    passes = Counter(tuple(d["_meta"].get("passes") or [d["_meta"].get("pass", "describe")]) for d in docs)

    L = ["# v2 icon descriptions: statistics vs v1", "",
         f"One description per distinct glyph. v1: {len(v1)} glyphs; v2: {len(v2)} glyphs "
         f"(the manual review's rejects were not described). Passes: "
         + ", ".join(f"{'+'.join(k)} {v}" for k, v in passes.items()) + ".", "",
         "## Kept vs discarded, by manual review level", "",
         "| Level | v1 kept | v1 discarded | v2 kept | v2 discarded |", "|---|---|---|---|---|"]
    t1, t2 = discard_table(v1, levels), discard_table(v2, levels)
    for lev in t1:
        L.append(f"| {lev} | {t1[lev]['kept']} | {t1[lev]['discarded']} | "
                 f"{t2[lev]['kept']} | {t2[lev]['discarded']} |")
    L += ["", "The loader keeps v2-discarded glyphs that the review calls useful or somewhat "
          "(the human label wins).", "", "## Text", "", "| Measure | v1 | v2 |", "|---|---|---|"]
    for k in s1:
        L.append(f"| {k} | {s1[k]} | {s2[k]} |")
    L += ["", "Target length mix: 5 / 8 / 3 of 16 = 0.31 / 0.50 / 0.19. v1 has no ES/RU "
          "examples; v2 adds 3 + 3 per glyph (not counted above).", "",
          "## v2 only", "",
          f"- Example-task domains: entropy {h:.2f} bits, {hn:.2f} of the maximum; "
          + ", ".join(f"{k} {v}" for k, v in dom.most_common()) + ".",
          "- Scores of kept glyphs (1–5): " + "; ".join(
              f"{k} " + " ".join(f"{s}:{scores[k][s]}" for s in range(1, 6)) for k in scores) + ".",
          f"- Soft problems: {n_soft} of {len(docs)} glyphs ({n_soft / len(docs):.0%}): "
          + ", ".join(f"{k} {v}" for k, v in soft.most_common()) + ".",
          f"- Contamination guard dropped {sum(guard.values())} strings: "
          + (", ".join(f"{k} {v}" for k, v in guard.most_common()) or "none") + ".",
          f"- Tokens: {usage['prompt']:,} in, {usage['output'] + usage['thoughts']:,} out; "
          f"≈ ${cost:.2f} at the batch price.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps(
        {"v1": s1, "v2": s2, "discard_v1": t1, "discard_v2": t2, "domains": dom,
         "domain_entropy": [h, hn], "soft": soft, "guard": guard, "usage": usage,
         "cost_usd": cost}, indent=1, default=list), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
