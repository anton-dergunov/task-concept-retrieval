#!/usr/bin/env python3
"""Paired comparisons of methods already on the silver leaderboard, from their stored
rankings (data/private/silver/<tag>/), without re-encoding anything:

    python scripts/silver_paired.py F1@1k-act:F1@1k-rand F1@all-lr-act1k:F1@all-lr-rand1k \
        [--views title,parent] [--extra-grades g1,g2,g3] [--tags 2026-11-21,2026-11-20]

For each pair A:B and view: mean ΔnDCG@5 (A − B) pooled over the eval sets with a bootstrap
95% interval, the same on Personal and on Realistic, and hit@1 wins/losses with McNemar's
exact p (tcr.silver.paired). Dev tasks only, as in silver_eval.py. A method's rankings come
from the first tag (default: newest first) that holds them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcr import config, silver  # noqa: E402
from tcr.datasets import split_of  # noqa: E402

DATASETS = ["personal", "personal_synth", "realistic", "public_short", "public_expanded"]
STORE = config.PRIVATE_DIR / "silver"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="A:B, method labels without the .v2 suffix")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--views", default="title,parent")
    ap.add_argument("--extra-grades", default="", help="e.g. g1,g2,g3 (silver_eval.py --extra-grades)")
    ap.add_argument("--tags", default="", help="stored-ranking tags to search (default: all, newest first)")
    a = ap.parse_args()

    tags = a.tags.split(",") if a.tags else sorted(
        (p.name for p in STORE.iterdir() if p.is_dir() and not p.name.endswith("-xling")), reverse=True)
    golds = {d: silver.eval_gold(d) for d in DATASETS}
    if a.extra_grades:
        golds = silver.with_extra(golds, a.extra_grades.split(","))
    dev = {d: {t["id"] for t in silver.eval_tasks(d) if split_of(t["id"]) == "dev" and t["id"] in golds[d]}
           for d in DATASETS}

    def per_task(label: str) -> dict:
        for tag in tags:
            p = STORE / tag / f"{label}.jsonl"
            if p.exists():
                out = {}
                for line in open(p, encoding="utf-8"):
                    r = json.loads(line)
                    d, t = r["dataset"], r["task_id"]
                    if d in dev and t in dev[d]:
                        out[f"{d}/{t}"] = silver.task_metrics(golds[d][t], r["top"])
                return out
        raise SystemExit(f"no stored rankings for {label} in {tags}")

    gold = f"judge + {a.extra_grades}" if a.extra_grades else "judge only"
    print(f"gold: {gold}; ΔnDCG@5 A − B [bootstrap 95%]; hit@1 A-only / B-only, McNemar p\n")
    print("| A vs B | view | ΔnDCG@5 [95%] | Personal | Realistic | hit@1 |")
    print("|---|---|---:|---:|---:|---:|")
    for view in a.views.split(","):
        sfx = f".{a.version}" + ("+parent" if view == "parent" else "")
        for pair in a.pairs:
            x, y = pair.split(":")
            A, B = per_task(x + sfx), per_task(y + sfx)
            p = silver.paired(A, B)
            sub = {d: silver.paired({k: v for k, v in A.items() if k.startswith(d + "/")},
                                    {k: v for k, v in B.items() if k.startswith(d + "/")})
                   for d in ("personal", "realistic")}
            pv = "< 0.001" if p["mcnemar_p"] < 0.001 else f"= {p['mcnemar_p']:.2f}"
            print(f"| {x} vs {y} | {view} | {p['d_ndcg5']:+.3f} [{p['d_ndcg5_ci'][0]:+.3f}, "
                  f"{p['d_ndcg5_ci'][1]:+.3f}] | {sub['personal']['d_ndcg5']:+.3f} | "
                  f"{sub['realistic']['d_ndcg5']:+.3f} | {p['hit1_only_a']} / {p['hit1_only_b']}, p {pv} |")


if __name__ == "__main__":
    main()
