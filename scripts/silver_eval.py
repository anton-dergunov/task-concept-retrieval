#!/usr/bin/env python3
"""Silver leaderboard: every method on the dev split of every eval set, scored against the
LLM judge's graded labels (docs/next-label-free-work.md step 1; gold and metrics in
tcr/silver.py).

    python scripts/silver_eval.py [--methods B0,B1,M1,M2,M3] [--versions v1,v2]
                                  [--datasets personal,...] [--ref M3.v2] [--no-parent]
                                  [--with latest|TAG,...|none] [--fresh] [--tag TAG]

Methods not run now are merged in from stored rankings (data/private/silver/<tag>/):
first the output tag's own, then --with (default: the most recent other tag), each
rescored against the current gold. So `--methods F1 --versions v2` adds F1 next to every
earlier method, with Δ vs --ref, instead of replacing the table. --fresh: no merging.

Metrics per method and dataset (dev tasks only; judge-dev and test never):
  ndcg5        graded nDCG@5 (gain = the judge's grade 3 / 2 / 1)
  hit1         top-1 graded ≥ 2, with its Wilson 95% interval
  hit1_ub      upper bound: an unjudged top-1 counted as a hit (for methods outside the
               judge's pool, e.g. F1: report [hit1, hit1_ub])
  judged1      the judge saw the top-1 (graded or left out); the rest is *unjudged*
  hit1_judged  hit1 over judged top-1s only
  rej1         top-1 is a glyph the author rejected in the icon review
  *_served     the same after the production filter (rejected glyphs removed)
Each method is also compared with --ref on the same tasks: ΔnDCG@5 with a bootstrap
interval, McNemar's exact test on hit@1.

Output: results/silver/<date>.json (aggregates, committed) and .md (tables);
per-task top-10s, scores and gate signals in data/private/silver/<date>/ (step 4).
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
from tcr.data import glyph_hashes, load_icons, rejected_names  # noqa: E402
from tcr.datasets import load, read_jsonl, split_of  # noqa: E402
from tcr.methods import build_method  # noqa: E402
from tcr.org_tasks import query_text  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "silver"
PRIVATE_OUT = config.PRIVATE_DIR / "silver"
PARENT_METHODS = ("M1", "M3", "F1")       # also scored with the nearest parent heading
TOP_K = 10
POOL_BUILDERS = ("M3 (title), M1, M2 and B1 (title + body) and M1 (parent + title), each on v1 "
                 "and v2 (tcr/candidates.py); B0 is not among them")


def dev_tasks(dataset: str, gold: dict) -> list:
    return [t for t in load(dataset) if split_of(t["id"]) == "dev" and t["id"] in gold]


def run(args) -> dict:
    datasets = args.datasets.split(",")
    golds = {d: silver.load_gold(silver.EVAL_RUNS[d]) for d in datasets}
    tasks = {d: dev_tasks(d, golds[d]) for d in datasets}
    for d in datasets:
        print(f"{d}: {len(tasks[d])} dev tasks with judge labels", file=sys.stderr)
    per_task: dict = {}       # label -> dataset -> task_id -> metrics
    ranked_out: dict = {}     # label -> dataset -> task_id -> {"top", "scores", "signal"}
    for ver in args.versions.split(","):
        icons = load_icons(version=ver)
        for key in args.methods.split(","):
            t0 = time.time()
            m = build_method(key, icons)
            views = [("title", False)]
            if key.split("@")[0] in PARENT_METHODS and not args.no_parent:
                views.append(("parent", True))
            for view, with_parent in views:
                label = f"{key}.{ver}" + ("+parent" if with_parent else "")
                for d in datasets:     # +parent: tasks without one get the title alone
                    pt, ro = {}, {}
                    for t in tasks[d]:
                        q = query_text(t["title"], parents=(t.get("parents") or []) if with_parent else ())
                        ranked, signal = m.rank_and_signal(q, top_k=TOP_K)
                        names = [n for n, _ in ranked]
                        pt[t["id"]] = silver.task_metrics(golds[d][t["id"]], names)
                        ro[t["id"]] = {"top": names, "scores": [round(s, 5) for _, s in ranked],
                                       "signal": round(signal, 4)}
                    per_task.setdefault(label, {})[d] = pt
                    ranked_out.setdefault(label, {})[d] = ro
            print(f"{key}.{ver}: {time.time() - t0:.0f} s", file=sys.stderr)
    res = {"datasets": datasets, "tasks": {d: len(v) for d, v in tasks.items()},
           "per_task": per_task, "ranked": ranked_out, "merged": {}}
    if not args.fresh:
        merge_stored(res, merge_tags(args), golds, tasks)
    return res


def merge_tags(args) -> list:
    tags = [args.tag]
    w = args.with_
    if w == "latest":
        others = [p for p in PRIVATE_OUT.iterdir() if p.is_dir() and p.name != args.tag] \
            if PRIVATE_OUT.exists() else []
        tags += [max(others, key=lambda p: p.stat().st_mtime).name] if others else []
    elif w and w != "none":
        tags += w.split(",")
    return tags


def merge_stored(res: dict, tags: list, golds: dict, tasks: dict) -> None:
    """Add every stored method not run now, rescoring its stored top-10s on the current gold
    (only tasks still in the dev sets). Earlier tags in `tags` win."""
    dev = {d: {t["id"] for t in ts} for d, ts in tasks.items()}
    for tag in tags:
        folder = PRIVATE_OUT / tag
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.jsonl")):
            label = p.stem
            if label in res["per_task"]:
                continue
            pt, ro = {}, {}
            for r in read_jsonl(p):
                d, tid = r["dataset"], r["task_id"]
                if d in dev and tid in dev[d]:
                    pt.setdefault(d, {})[tid] = silver.task_metrics(golds[d][tid], r["top"])
                    ro.setdefault(d, {})[tid] = {k: r[k] for k in ("top", "scores", "signal") if k in r}
            if pt:
                res["per_task"][label], res["ranked"][label] = pt, ro
                res["merged"][label] = tag
    if res["merged"]:
        print(f"merged from stored rankings: {res['merged']}", file=sys.stderr)


def summarize(res: dict, ref: str) -> dict:
    per_task, datasets = res["per_task"], res["datasets"]
    out = {"date": date.today().isoformat(), "ref": ref, "tasks": res["tasks"],
           "rejected_glyphs": len({glyph_hashes()[n] for n in rejected_names()}), "methods": {}}
    for label, by_d in per_task.items():
        row = {d: silver.aggregate(by_d[d]) for d in by_d}
        pooled = {f"{d}/{t}": m for d in by_d for t, m in by_d[d].items()}
        row["pooled"] = silver.aggregate(pooled)
        if set(by_d) == set(datasets):
            row["macro"] = {k: round(sum(row[d][k] for d in datasets) / len(datasets), 4)
                            for k in ("ndcg5", "hit1", "hit1_ub", "judged1", "rej1", "ndcg5_served",
                                      "hit1_served")}
        if ref in per_task and label != ref:
            cmp = {d: silver.paired(by_d[d], per_task[ref][d]) for d in by_d if d in per_task[ref]}
            ref_pooled = {f"{d}/{t}": m for d in by_d if d in per_task[ref]
                          for t, m in per_task[ref][d].items()}
            cmp["pooled"] = silver.paired(pooled, ref_pooled)
            row["vs_ref"] = cmp
        if label in res.get("merged", {}):
            row["from_tag"] = res["merged"][label]
        out["methods"][label] = row
    return out


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def markdown(s: dict) -> str:
    cols = [("ndcg5", "nDCG@5"), ("hit1", "hit@1"), ("hit1_ci", "95% CI"), ("hit1_ub", "hit@1 ub"),
            ("judged1", "judged@1"),
            ("hit1_judged", "hit@1 judged"), ("rej1", "rej@1"), ("ndcg5_served", "nDCG@5 served"),
            ("hit1_served", "hit@1 served")]
    lines = [f"# Silver leaderboard {s['date']}", "",
             "Dev split of each eval set, scored against the LLM judge's graded labels "
             "(prompt j4). Rejected glyphs are removed from gold "
             f"({s['rejected_glyphs']} in the review); *served* = after the production filter. "
             f"The judge's candidate pool was built by {POOL_BUILDERS}, so these methods are "
             "favoured; judged@1 shows how often a top-1 lies outside what the judge saw. "
             "*hit@1 ub* counts an unjudged top-1 as a hit (the upper bound for a method outside "
             "the pool). "
             f"Δ vs {s['ref']}: mean ΔnDCG@5 [bootstrap 95%], hit@1 wins/losses, McNemar p.", ""]

    def fmt(v):
        if isinstance(v, list):
            return f"{v[0]:.2f}–{v[1]:.2f}"
        return "" if v is None else f"{v:.3f}"

    for d in ["pooled"] + list(s["tasks"]):
        n = sum(s["tasks"].values()) if d == "pooled" else s["tasks"][d]
        lines += [f"## {d} (n = {n})", "",
                  "| method | " + " | ".join(c for _, c in cols) + f" | Δ vs {s['ref']} |",
                  "|---|" + "---:|" * (len(cols) + 1)]
        rows = [(lab, r[d], (r.get("vs_ref") or {}).get(d)) for lab, r in s["methods"].items() if d in r]
        for lab, a, c in sorted(rows, key=lambda x: -x[1]["ndcg5"]):
            if lab == s["ref"]:
                delta = "ref"
            elif c:
                delta = (f"{c['d_ndcg5']:+.3f} [{c['d_ndcg5_ci'][0]:+.3f}, {c['d_ndcg5_ci'][1]:+.3f}]; "
                         f"{c['hit1_only_a']}/{c['hit1_only_b']}, p{_p(c['mcnemar_p'])}")
            else:
                delta = ""
            lines.append(f"| {lab} | " + " | ".join(fmt(a.get(k)) for k, _ in cols) + f" | {delta} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--methods", default="B0,B1,M1,M2,M3")
    ap.add_argument("--versions", default="v1,v2")
    ap.add_argument("--datasets", default=",".join(silver.EVAL_RUNS))
    ap.add_argument("--ref", default="M3.v2")
    ap.add_argument("--no-parent", action="store_true", help="skip the +parent query rows")
    ap.add_argument("--tag", default=date.today().isoformat(), help="output file name")
    ap.add_argument("--with", dest="with_", default="latest",
                    help="stored rankings to merge: latest (default), TAG[,TAG], or none")
    ap.add_argument("--fresh", action="store_true", help="no merging with stored rankings")
    args = ap.parse_args()

    res = run(args)
    s = summarize(res, args.ref)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.tag}.json").write_text(json.dumps(s, indent=1))
    (OUT_DIR / f"{args.tag}.md").write_text(markdown(s))
    priv = PRIVATE_OUT / args.tag
    priv.mkdir(parents=True, exist_ok=True)
    for label, by_d in res["ranked"].items():
        with open(priv / f"{label}.jsonl", "w", encoding="utf-8") as f:
            for d, rows in by_d.items():
                for tid, r in rows.items():
                    f.write(json.dumps({"task_id": tid, "dataset": d, **r,
                                        **res["per_task"][label][d][tid]}) + "\n")
    print(f"-> {OUT_DIR / args.tag}.json/.md; per task in {priv}", file=sys.stderr)


if __name__ == "__main__":
    main()
