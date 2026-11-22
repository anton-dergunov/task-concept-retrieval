#!/usr/bin/env python3
"""Judge-noise ceiling (docs/argon-experiments.md #6): re-judge dev tasks of the Argon-judged
eval sets in a fresh run with another candidate order, and measure how far the judge agrees
with itself.

    python scripts/noise_ceiling.py select [--n personal=150,personal_synth=75,realistic=75]
        -> data/private/judge/rejudge/<dataset>.ids (dev tasks, fixed seed)
    (scripts/vertex_round3_jobs.sh v3a judges them: run rejudge-<dataset>, --shuffle-seed 1)
    python scripts/noise_ceiling.py report
        -> results/noise_ceiling.md (metrics only)

Report, per eval set and pooled:
- top-1 the same glyph; top-1 a hit (grade >= 2) under the other run's grades, both ways;
  no-good-icon agreement;
- quadratic-weighted Cohen's kappa of the grades (0-3) over every icon either run graded;
- the ceiling: the re-judge's ranking scored as a method against the original gold
  (nDCG@5 / hit@1, tcr.silver.task_metrics), next to the best model's numbers on the same
  tasks. A model cannot be expected to beat the judge at agreeing with the judge.
- which Δs to trust: every stored silver ranking rescored against the re-judge's gold
  (Spearman ρ of the method orders), judge noise's share of a paired Δ's per-task
  variance, its SD on the full pooled dev set, and sign agreement by |Δ|.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config, silver  # noqa: E402
from tcr.datasets import split_of  # noqa: E402

DIR = config.PRIVATE_DIR / "judge" / "rejudge"
DEFAULT_N = "personal=150,personal_synth=75,realistic=75"
RUN = "rejudge-{}"          # judge_batch.py run per eval set
OUT = config.RESULTS_DIR / "noise_ceiling.md"
STORE = config.PRIVATE_DIR / "silver"
MODEL_LABEL = "F1@all-large-all3k.v2+parent"   # the best model, for comparison


def select(spec: str) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    for part in spec.split(","):
        d, n = part.split("=")
        gold = silver.eval_gold(d)
        ids = [t["id"] for t in silver.eval_tasks(d) if split_of(t["id"]) == "dev" and t["id"] in gold]
        ids.sort(key=lambda i: hashlib.sha1(f"rejudge:{i}".encode()).hexdigest())
        pick = ids[:int(n)]
        (DIR / f"{d}.ids").write_text("\n".join(pick) + "\n")
        print(f"{d}: {len(pick)} of {len(ids)} dev tasks -> {DIR / f'{d}.ids'}")


def kappa_w(pairs: list, k: int = 4) -> float:
    """Quadratic-weighted Cohen's kappa of integer ratings 0..k-1."""
    n = len(pairs)
    if not n:
        return float("nan")
    obs = [[0] * k for _ in range(k)]
    for a, b in pairs:
        obs[a][b] += 1
    ra = [sum(obs[i]) for i in range(k)]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2  # noqa: E731
    o = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    e = sum(w(i, j) * ra[i] * rb[j] for i in range(k) for j in range(k)) / n / n
    return 1 - o / e if e else float("nan")


def model_rankings(label: str) -> dict:
    """{dataset/task_id: top} of a stored silver ranking (newest tag holding it)."""
    for tag in sorted((p.name for p in STORE.iterdir() if p.is_dir() and not p.name.endswith("-xling")),
                      reverse=True):
        p = STORE / tag / f"{label}.jsonl"
        if p.exists():
            return {f"{r['dataset']}/{r['task_id']}": r["top"] for r in map(json.loads, open(p))}
    return {}


def all_rankings() -> dict:
    """{label: {dataset/task_id: top}} of every stored silver ranking (newest tag wins)."""
    out: dict = {}
    for tag in sorted((p.name for p in STORE.iterdir() if p.is_dir() and not p.name.endswith("-xling")),
                      reverse=True):
        for p in sorted((STORE / tag).glob("*.jsonl")):
            if p.stem not in out:
                out[p.stem] = {f"{r['dataset']}/{r['task_id']}": r["top"] for r in map(json.loads, open(p))}
    return out


def trust(pairs_of_gold: dict, n_full: int) -> list:
    """Does the leaderboard survive the re-judge? Every stored method scored on the
    re-judged tasks against both golds; for every pair of methods, how much of the per-task
    variance of their ΔnDCG@5 is judge noise. If the two runs are independent draws of the
    judge, d1 - d2 (the task's Δ under each gold) has twice the noise variance of one run."""
    import numpy as np
    keys = sorted(pairs_of_gold)
    runs = {lab: r for lab, r in all_rankings().items() if all(k in r for k in keys)}
    s1 = {lab: np.array([silver.task_metrics(pairs_of_gold[k][0], r[k])["ndcg5"] for k in keys])
          for lab, r in runs.items()}
    s2 = {lab: np.array([silver.task_metrics(pairs_of_gold[k][1], r[k])["ndcg5"] for k in keys])
          for lab, r in runs.items()}
    labs = sorted(runs, key=lambda x: -s1[x].mean())
    m1 = np.array([s1[x].mean() for x in labs])
    m2 = np.array([s2[x].mean() for x in labs])
    rank = lambda v: np.argsort(np.argsort(-v))  # noqa: E731
    rho = float(np.corrcoef(rank(m1), rank(m2))[0, 1])
    share, sd_noise, agree = [], [], {}
    bins = [(0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 1)]
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            d1 = s1[labs[i]] - s1[labs[j]]
            d2 = s2[labs[i]] - s2[labs[j]]
            var_noise = float(np.var(d1 - d2, ddof=1)) / 2
            var_tot = float(np.var(d1, ddof=1))
            if var_tot > 0:
                share.append(min(1.0, var_noise / var_tot))
            sd_noise.append((var_noise / n_full) ** 0.5)
            a = abs(d1.mean())
            for lo, hi in bins:
                if lo <= a < hi:
                    agree.setdefault((lo, hi), []).append(int(np.sign(d1.mean()) == np.sign(d2.mean())))
    med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
    lines = ["", "## Does the leaderboard survive the re-judge?", "",
             f"Every stored method ({len(labs)}) on the {len(keys)} re-judged tasks, nDCG@5 against the "
             f"original gold and against the re-judge's. Spearman ρ of the two method orders: **{rho:.3f}**.", "",
             "| method | nDCG@5 original | nDCG@5 re-judge | change |", "|---|---:|---:|---:|"]
    show = labs[:12] + [x for x in labs[12:] if x.split("@")[0].split(".")[0] in ("M3", "M1", "M2", "B1", "B0")]
    for x in show:
        k = labs.index(x)
        lines.append(f"| {x} | {m1[k]:.3f} | {m2[k]:.3f} | {m2[k] - m1[k]:+.3f} |")
    lines += ["", f"Over all {len(share)} pairs of methods: judge noise is a median **{med(share):.0%}** of the "
              "per-task variance of a paired ΔnDCG@5 (the rest is which tasks happen to be in the set). "
              "A bootstrap over tasks resamples each task with its one label, so its interval already "
              f"covers that noise. On the full pooled dev set (n = {n_full}), judge noise alone gives a "
              f"paired Δ a standard deviation of a median **{med(sd_noise):.4f}** "
              f"(90% of pairs ≤ {float(__import__('numpy').percentile(sd_noise, 90)):.4f}).", "",
              "Sign of the mean Δ on these tasks, original gold vs re-judge, by |Δ| under the original:", "",
              "| \\|Δ\\| | pairs | same sign |", "|---|---:|---:|"]
    for lo, hi in bins:
        v = agree.get((lo, hi), [])
        if v:
            lines.append(f"| {lo:.3f}–{hi:.3f} | {len(v)} | {sum(v) / len(v):.2f} |")
    return lines


def report() -> None:
    model = model_rankings(MODEL_LABEL)
    both: dict = {}             # dataset/task_id -> (original gold, re-judge gold)
    rows, pooled = [], {"n": 0}
    acc_all = {k: [] for k in ("same1", "hit_ab", "hit_ba", "ngi", "nd_j", "h1_j", "nd_m", "h1_m")}
    kp_all = []
    for d in silver.EVAL_RUNS:
        run = RUN.format(d)
        if not silver.label_path(run).exists():
            continue
        orig, new = silver.eval_gold(d), silver.load_gold([run])
        acc = {k: [] for k in acc_all}
        kp = []
        for t, g2 in new.items():
            g1 = orig.get(t)
            if g1 is None:
                continue
            both[f"{d}/{t}"] = (g1, g2)
            a, b =(g1.order[0] if g1.order else None), (g2.order[0] if g2.order else None)
            acc["same1"].append(int(a is not None and a == b))
            acc["hit_ab"].append(int(b is not None and g1.grades.get(b, 0) >= 2))
            acc["hit_ba"].append(int(a is not None and g2.grades.get(a, 0) >= 2))
            acc["ngi"].append(int(g1.no_good_icon == g2.no_good_icon))
            m = silver.task_metrics(g1, g2.order)
            acc["nd_j"].append(m["ndcg5"]); acc["h1_j"].append(m["hit1"])
            top = model.get(f"{d}/{t}")
            if top is not None:
                mm = silver.task_metrics(g1, top)
                acc["nd_m"].append(mm["ndcg5"]); acc["h1_m"].append(mm["hit1"])
            for icon in set(g1.grades) | set(g2.grades):
                kp.append((g1.grades.get(icon, 0), g2.grades.get(icon, 0)))
        if not acc["same1"]:
            continue
        mean = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
        rows.append((d, len(acc["same1"]), mean, kappa_w(kp)))
        for k in acc_all:
            acc_all[k] += acc[k]
        kp_all += kp
    if not rows:
        sys.exit("no re-judge labels yet (vertex_round3_jobs.sh v3a, then judge_batch.py export)")
    mean = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc_all.items()}
    rows.append(("pooled", len(acc_all["same1"]), mean, kappa_w(kp_all)))
    lines = ["# Judge-noise ceiling (argon-experiments.md #6)", "",
             "Dev tasks of the Argon-judged eval sets, re-judged by the same judge (prompt j4, "
             "gemini-4-argon) in a fresh run with the candidates in another order. *Judge as a "
             f"method*: the re-judge's ranking scored against the original gold; *model*: {MODEL_LABEL} "
             "on the same tasks, plain gold.", "",
             "| set | n | same top-1 | re-judge top-1 a hit | original top-1 a hit | no-good-icon agree | "
             "grade κ (quadratic) | judge as a method nDCG@5 / hit@1 | model nDCG@5 / hit@1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d, n, m, k in rows:
        lines.append(f"| {d} | {n} | {m['same1']:.2f} | {m['hit_ab']:.2f} | {m['hit_ba']:.2f} | "
                     f"{m['ngi']:.2f} | {k:.2f} | {m['nd_j']:.3f} / {m['h1_j']:.2f} | "
                     f"{m['nd_m']:.3f} / {m['h1_m']:.2f} |")
    n_full = sum(len([t for t in silver.eval_tasks(d) if split_of(t["id"]) == "dev" and t["id"] in g])
                 for d in silver.EVAL_RUNS for g in [silver.eval_gold(d)])
    lines += trust(both, n_full)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--n", default=DEFAULT_N)
    sub.add_parser("report")
    a = ap.parse_args()
    select(a.n) if a.cmd == "select" else report()


if __name__ == "__main__":
    main()
