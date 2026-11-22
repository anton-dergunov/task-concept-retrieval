#!/usr/bin/env python3
"""Contrast set of the minimal pairs (docs/argon-experiments.md #3): does a method's top-1
follow a small edit of the title that changes the right icon?

The pairs come from validation tasks of the judged training pools (augment_pools.py
minpair, MP_VAL_SHARE): no F1 run trains on the original, its rewrites or the edit. Argon
judged both sides (the original in its pool run, the edit in mp-*-j4), so each side has its
own gold. A pair is a *flip* when the original's top icon is no longer a hit (grade < 2) on
the edit, otherwise a *keep* (the edit did not change the icon: an invariance pair).

    nice python scripts/contrast_eval.py [--methods M3,F1@all-large-all3k,...] [--versions v2]
                                         [--no-parent] [--tag TAG]

Metrics per method, on flips: hit@1 of the original and of the edit (each on its own gold),
both (pair accuracy), *follows* (the top-1 changed with the edit) and *stuck* (the edit's
top-1 is the method's top-1 of the original, and it is not a hit on the edit); nDCG@5 of
each side. On keeps: hit@1 on both sides and the same top-1. The gap between the original's
and the edit's hit@1 is the contrast-set drop (Gardner et al., 2020).

Output: results/contrast/<tag>.md and .json (aggregates only, no task text).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tcr import config, silver  # noqa: E402
from tcr.data import load_icons  # noqa: E402
from tcr.datasets import read_jsonl  # noqa: E402
from tcr.methods import build_method  # noqa: E402
from tcr.org_tasks import query_text  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "contrast"
# (edit judge run, original pool runs, the edits as judge input)
SOURCES = [
    ("mp-synth-j4", ["pool-synth-j4", "pool-synth2-j4"],
     config.PRIVATE_DIR / "train_pools" / "personal_synth_pool.mp_pool.jsonl"),
    ("mp-mslatte-j4", ["pool-mslatte-j4"], config.DATA_DIR / "train_pools" / "train_mslatte.mp_pool.jsonl"),
]
DEFAULT_METHODS = ("M3,F1@all-large-all3k,F1@all-large-e8,F1@all-kd-final,F1@all-lr-para-all3k,"
                   "F1@all-base-lr")
PARENT_METHODS = ("M1", "M3", "F1")
TOP_K = 10


def load_pairs() -> list:
    """The contrast-set pairs: {"set", "id", "kind" (flip / keep), "orig", "edit" (title,
    parents, gold)}. Both sides use the edit's parents (the original's, as minpair kept them)."""
    pairs = []
    for run, orig_runs, pool in SOURCES:
        if not pool.exists() or not silver.label_path(run).exists():
            continue
        g_edit = silver.load_gold([run])
        g_orig = silver.load_gold(orig_runs)
        orig_text = {t["id"]: t for r in orig_runs for t in silver.run_tasks(r)}
        for r in read_jsonl(pool):
            m = r["meta"]
            if not m.get("val"):
                continue
            oid = m["orig_task_id"]
            ge, go, ot = g_edit.get(r["id"]), g_orig.get(oid), orig_text.get(oid)
            if not (ge and go and ot and go.order and ge.order) or ge.no_good_icon:
                continue
            kind = "keep" if ge.grades.get(go.order[0], 0) >= 2 else "flip"
            par = list(r.get("parents") or [])
            pairs.append({"set": run, "id": r["id"], "kind": kind,
                          "orig": {"title": ot["title"], "parents": par, "gold": go},
                          "edit": {"title": r["title"], "parents": par, "gold": ge}})
    return pairs


def score(pairs: list, tops: dict) -> dict:
    """Aggregates for one method view; tops: {pair id: (orig top-k, edit top-k)}."""
    out = {}
    for kind in ("flip", "keep"):
        ps = [p for p in pairs if p["kind"] == kind]
        if not ps:
            continue
        acc = {k: 0.0 for k in ("hit_orig", "hit_edit", "both", "same_top", "stuck", "nd_orig", "nd_edit")}
        for p in ps:
            to, te = tops[p["id"]]
            mo = silver.task_metrics(p["orig"]["gold"], to)
            me = silver.task_metrics(p["edit"]["gold"], te)
            same = silver.canonical(to[0]) == silver.canonical(te[0])
            acc["hit_orig"] += mo["hit1"]
            acc["hit_edit"] += me["hit1"]
            acc["both"] += mo["hit1"] and me["hit1"]
            acc["same_top"] += same
            acc["stuck"] += same and not me["hit1"]
            acc["nd_orig"] += mo["ndcg5"]
            acc["nd_edit"] += me["ndcg5"]
        out[kind] = {"n": len(ps), **{k: round(v / len(ps), 4) for k, v in acc.items()}}
    return out


def run(args) -> dict:
    pairs = load_pairs()
    if not pairs:
        sys.exit("no contrast pairs: run vertex_round3_jobs.sh v3a first")
    sets = sorted({p["set"] for p in pairs})
    print(f"{len(pairs)} pairs: " + ", ".join(
        f"{s} {sum(p['set'] == s and p['kind'] == 'flip' for p in pairs)} flip / "
        f"{sum(p['set'] == s and p['kind'] == 'keep' for p in pairs)} keep" for s in sets), file=sys.stderr)
    keeps = [p for p in pairs if p["kind"] == "keep"]
    res = {"date": date.today().isoformat(), "pairs": {s: {k: sum(p["set"] == s and p["kind"] == k for p in pairs)
                                                          for k in ("flip", "keep")} for s in sets},
           # the judge's own top-1 on keeps: how often it names the same icon twice
           "judge_same_top_keep": round(sum(p["orig"]["gold"].order[0] == p["edit"]["gold"].order[0]
                                            for p in keeps) / max(1, len(keeps)), 4),
           "methods": {}}
    for ver in args.versions.split(","):
        icons = load_icons(version=ver)
        for key in args.methods.split(","):
            t0 = time.time()
            m = build_method(key, icons)
            views = [("", False)]
            if key.split("@")[0] in PARENT_METHODS and not args.no_parent:
                views.append(("+parent", True))
            for suffix, with_parent in views:
                def q(side):
                    return query_text(side["title"], parents=side["parents"] if with_parent else ())
                if hasattr(m, "prepare"):
                    m.prepare([q(p[s]) for p in pairs for s in ("orig", "edit")])
                tops = {p["id"]: tuple([n for n, _ in m.rank_and_signal(q(p[s]), top_k=TOP_K)[0]]
                                       for s in ("orig", "edit")) for p in pairs}
                label = f"{key}.{ver}{suffix}"
                res["methods"][label] = {"all": score(pairs, tops),
                                         **{s: score([p for p in pairs if p["set"] == s], tops) for s in sets}}
            print(f"{key}.{ver}: {time.time() - t0:.0f} s", file=sys.stderr)
    return res


def markdown(res: dict) -> str:
    n = res["pairs"]
    lines = [f"# Contrast set of the minimal pairs, {res['date']}", "",
             "Validation tasks of the judged training pools, each edited by Flash in 1–3 words so the right "
             "icon should change, the edit judged by Argon (argon-experiments.md #3; scripts/contrast_eval.py). "
             "*Flip*: the original's top icon is no longer a hit on the edit; *keep*: it still is. "
             "Pairs: " + "; ".join(f"{s}: {v['flip']} flip, {v['keep']} keep" for s, v in n.items()) + ". "
             f"For reference, the judge itself names the same top-1 on both sides of {res['judge_same_top_keep']:.0%} "
             "of keeps (a keep only needs the original's icon to stay a hit). With ~120 flips a rate's 95% "
             "interval is about ±0.09, so only large differences between methods count.", ""]
    for part in ["all"] + list(n):
        lines += [f"## {part}", "",
                  "| method | flips: hit@1 orig | hit@1 edit | both | top-1 follows | stuck | nDCG@5 orig / edit "
                  "| keeps: hit@1 both | same top-1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows = sorted(res["methods"].items(), key=lambda kv: -(kv[1][part].get("flip", {}).get("both", 0)))
        for label, r in rows:
            f, k = r[part].get("flip"), r[part].get("keep")
            if not f:
                continue
            kk = f"{k['both']:.2f} | {k['same_top']:.2f}" if k else "– | –"
            lines.append(f"| {label} | {f['hit_orig']:.2f} | {f['hit_edit']:.2f} | {f['both']:.2f} | "
                         f"{1 - f['same_top']:.2f} | {f['stuck']:.2f} | {f['nd_orig']:.3f} / {f['nd_edit']:.3f} | {kk} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--versions", default="v2")
    ap.add_argument("--no-parent", action="store_true")
    ap.add_argument("--tag", default=date.today().isoformat())
    args = ap.parse_args()
    res = run(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.tag}.json").write_text(json.dumps(res, indent=1))
    md = markdown(res)
    (OUT_DIR / f"{args.tag}.md").write_text(md)
    print(md)
    print(f"-> {OUT_DIR / args.tag}.md")


if __name__ == "__main__":
    main()
