#!/usr/bin/env python3
"""F1's learning curves (docs/f1-dual-encoder.md §6): silver nDCG@5 against the number of
judged pool tasks, and held-out nDCG@5 over training for every run.

    python scripts/f1_report.py [--silver results/silver/2026-11-15.json]

Reads results/f1/<run>.done.json + .train_log.jsonl and a silver leaderboard JSON; writes
results/f1/learning_curve.png and results/f1/learning_curve.md (the same numbers as tables).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F1_DIR = ROOT / "results" / "f1"
CURVE = [("F1@0", "F1-0"), ("F1@1k", "F1-1k"), ("F1@3k", "F1-3k"), ("F1@all", "F1-all")]
RUNS = [("F1", "F1")] + CURVE
DATASETS = ["personal", "personal_synth", "realistic", "public_short", "public_expanded"]
# validated categorical slots 1-5 (dataviz reference palette, light); ink for "pooled"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
# panel B encodes data size (a magnitude): one hue, light -> dark
SIZE_RAMP = {"F1@0": "#9ec5f0", "F1@1k": "#5d9ce6", "F1@3k": "#2a78d6", "F1@all": "#174d91",
             "F1": "#174d91"}


def spread(ys: list, gap: float) -> list:
    """Label positions near ys, at least `gap` apart (order kept)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        out[b] = max(out[b], out[a] + gap)
    shift = (sum(out) - sum(ys)) / len(ys)          # recentre the block
    return [y - shift for y in out]


def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load(silver_path: Path) -> dict:
    s = json.loads(silver_path.read_text())
    out = {"silver": s, "runs": {}}
    for name, stem in RUNS:
        done = json.loads((F1_DIR / f"{stem}.done.json").read_text())
        evals = [r for r in read_jsonl(F1_DIR / f"{stem}.train_log.jsonl") if "eval" in r]
        out["runs"][name] = {"done": done, "evals": evals}
    return out


def plot(d: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.42})
    meth = d["silver"]["methods"]

    # A: silver nDCG@5 vs pool tasks (log x; 0 drawn at 300 and labelled "0")
    xs = [max(d["runs"][n]["done"]["pool_tasks"], 300) for n, _ in CURVE]
    ref = meth["M3.v2"]["pooled"]["ndcg5"]
    a1.axhline(ref, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    a1.text(xs[-1] * 1.12, ref, "M3.v2, pooled", color=MUTED, va="center", fontsize=9)
    ends, labels = [], []
    for i, ds in enumerate(DATASETS):
        ys = [meth[f"{n}.v2"][ds]["ndcg5"] for n, _ in CURVE]
        a1.plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, zorder=2)
        ends.append(ys[-1]); labels.append((ds.replace("_", "-"), INK2, "normal"))
    ys = [meth[f"{n}.v2"]["pooled"]["ndcg5"] for n, _ in CURVE]
    a1.plot(xs, ys, color=INK, lw=2.6, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3)
    ends.append(ys[-1]); labels.append(("pooled", INK, "bold"))
    for y, (t, c, w) in zip(spread(ends, 0.013), labels):
        a1.text(xs[-1] * 1.12, y, t, color=c, va="center", fontsize=9, weight=w)
    a1.set_xscale("log")
    a1.set_xticks(xs, ["0", "1k", "3k", f"{xs[-1] / 1000:.1f}k"])
    a1.minorticks_off()
    a1.set_xlim(240, xs[-1] * 1.9)
    a1.set_xlabel("judged pool tasks in training (log scale)")
    a1.set_ylabel("silver nDCG@5 (eval dev sets)")
    a1.set_title("More judged tasks still help", loc="left", weight="bold")

    # B: held-out nDCG@5 over training
    ends, names = [], []
    for name, _ in RUNS:
        ev = d["runs"][name]["evals"]
        ex = [e["epoch"] for e in ev]
        ey = [e["all"]["ndcg5"] for e in ev]
        a2.plot(ex, ey, color=SIZE_RAMP[name], lw=2, zorder=2,
                ls=(0, (4, 2)) if name == "F1" else "-")
        ends.append((ex[-1], ey[-1])); names.append(name)
    for (ex, _), y, name in zip(ends, spread([e[1] for e in ends], 0.012), names):
        label = "F1 (no icons held out, dashed)" if name == "F1" else name
        a2.text(ex + 0.12, y, label, color=INK2, va="center", fontsize=9)
    a2.set_xlim(0, 8.6)
    a2.set_xticks(range(7))
    a2.set_xlabel("epoch")
    a2.set_ylabel("held-out pool nDCG@5")
    a2.set_title("Within a run: fast start, still rising at the end", loc="left", weight="bold")
    for a in (a1, a2):
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def markdown(d: dict) -> str:
    meth = d["silver"]["methods"]
    ref = meth["M3.v2"]
    lines = ["# F1 learning curve", "",
             f"Silver: `{d['silver']['date']}`, dev split, nDCG@5 (hit@1 lower bound in brackets). "
             "Held-out: 10% of pool tasks, all languages. Plot: learning_curve.png.", "",
             "| run | pool tasks | " + " | ".join(DATASETS) + " | pooled | held-out (best step / total) |",
             "|---|---:|" + "---:|" * (len(DATASETS) + 2)]
    for name, _ in RUNS:
        done = d["runs"][name]["done"]
        cells = [f"{meth[f'{name}.v2'][ds]['ndcg5']:.3f}" for ds in DATASETS]
        p = meth[f"{name}.v2"]["pooled"]
        lines.append(f"| {name} | {done['pool_tasks']:,} | " + " | ".join(cells)
                     + f" | {p['ndcg5']:.3f} ({p['hit1']:.3f}) | {done['best']['all']['ndcg5']:.3f} "
                     f"({done['best']['step']} / {done['steps']}) |")
    cells = [f"{ref[ds]['ndcg5']:.3f}" for ds in DATASETS]
    lines.append("| M3.v2 | – | " + " | ".join(cells)
                 + f" | {ref['pooled']['ndcg5']:.3f} ({ref['pooled']['hit1']:.3f}) | – |")
    lines += ["", "## Held-out nDCG@5 per evaluation", "",
              "| epoch | " + " | ".join(n for n, _ in RUNS) + " |", "|---:|" + "---:|" * len(RUNS)]
    table: dict = {}
    for name, _ in RUNS:
        for e in d["runs"][name]["evals"]:
            table.setdefault(round(e["epoch"] * 2) / 2, {})[name] = e["all"]["ndcg5"]
    for ep in sorted(table):
        lines.append(f"| {ep:g} | " + " | ".join(f"{table[ep][n]:.3f}" if n in table[ep] else ""
                                                 for n, _ in RUNS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--silver", default=str(ROOT / "results" / "silver" / "2026-11-15.json"))
    a = ap.parse_args()
    d = load(Path(a.silver))
    plot(d, F1_DIR / "learning_curve.png")
    (F1_DIR / "learning_curve.md").write_text(markdown(d))
    print(f"-> {F1_DIR / 'learning_curve.png'}, learning_curve.md", file=sys.stderr)


if __name__ == "__main__":
    main()
