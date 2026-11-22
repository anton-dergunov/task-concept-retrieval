"""Export a matcher as a standalone bundle for the agenda (see tcr/bundle.py).

    python scripts/export_bundle.py --method M3 --out DIR [--check] [--keep-rejected]
    python scripts/export_bundle.py --method F1 --run all-large-all3k --out DIR [--check]
        [--precision fp16|fp32|int8] [--min-prob 0.2] [--hf-repo USER/NAME] [--revision REV]

DIR is typically agentic-org-planner's `icons/task-matcher/`. With --check, the
exported bundle is compared with the in-repo method on the Realistic tasks: top-1
agreement, gate agreement, and latency.

The bundle leaves out every glyph the manual icon review rejected (the production hard
rule; evaluation does not filter, see tcr/silver.py), unless --keep-rejected.

F1 (a fine-tuned dual encoder, docs/f1-dual-encoder.md; docs/deployment.md "F1 bundles"):
- the checkpoint `.cache/models/F1-<run>/` is exported to ONNX in `--onnx-dir` (default
  `.cache/onnx/F1-<run>/`, the folder to upload to Hugging Face as --hf-repo), and the
  files are linked where the agenda's runner and this repo's `fetch` look for them, so
  neither downloads a file that is not uploaded yet;
- the gate: P(the shown top-1 is acceptable | z-signal) is fitted on the silver dev
  split (the stored rankings of `F1@<run>.v2+parent`, gold with extra grades g1–g4),
  and the threshold is the z where it equals --min-prob;
- --check compares the bundle (ONNX query side) with those stored PyTorch rankings.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tcr import bundle, config, datasets, silver  # noqa: E402
from tcr.data import load_icons, rejected_names  # noqa: E402
from tcr.datasets import read_jsonl, split_of  # noqa: E402
from tcr.methods import build_method  # noqa: E402
from tcr.org_tasks import query_text  # noqa: E402

SILVER_RANKINGS = config.PRIVATE_DIR / "silver"
SILVER_TAG = "2026-11-22-extra"          # holds every current F1 run's +parent rankings
EXTRA_GRADES = ("g1", "g2", "g3", "g4")
RUNNER_CACHE = Path.home() / ".cache" / "ps-task-icons"     # the agenda runner's CACHE_DIR
HF_USER = "AntonDergunov"


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def export_icons(keep_rejected: bool = False, version: str = None) -> tuple:
    """The icons a production bundle may show."""
    icons = load_icons(version=version)
    if keep_rejected:
        return icons
    rej = rejected_names()
    return tuple(ic for ic in icons if ic.name not in rej)


def check(method: str, out: Path, dataset: str, keep_rejected: bool = False) -> None:
    tasks = datasets.load(dataset)
    exported = bundle.BundleMatcher(out)
    reference = build_method(method, export_icons(keep_rejected))
    same_top1 = same_gate = shown = 0
    lat = []
    disagreements = []
    for t in tasks:
        ref = reference.match_one(query_text(t["title"]))
        t0 = time.perf_counter()
        got = exported.match_one(exported.query_for(t))
        lat.append((time.perf_counter() - t0) * 1000)
        same_top1 += ref.icon == got.icon
        same_gate += ref.shown == got.shown
        shown += got.shown
        if ref.icon != got.icon:
            disagreements.append((t["title"], ref.icon, got.icon))
    n = len(tasks)
    print(f"\n{dataset}: {n} tasks, bundle vs {method}")
    print(f"  top-1 agreement  {same_top1 / n:.1%}")
    print(f"  gate agreement   {same_gate / n:.1%}   (bundle shows an icon for {shown / n:.1%})")
    print(f"  latency p50 {statistics.median(lat):.1f} ms, "
          f"p95 {sorted(lat)[int(0.95 * (n - 1))]:.1f} ms")
    for title, a, b in disagreements[:10]:
        print(f"    {title!r}: {a} -> {b}")


# --- F1 ----------------------------------------------------------------------------

def stored_rankings(run: str, tag: str = SILVER_TAG) -> list:
    """silver_eval.py's stored +parent rankings of F1@RUN on the dev split (PyTorch fp32,
    all v2 icons: rejected glyphs are still in `top`)."""
    p = SILVER_RANKINGS / tag / f"F1@{run}.v2+parent.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"no stored rankings at {p} (scripts/silver_eval.py)")
    return [r for r in read_jsonl(p) if split_of(r["task_id"]) == "dev"]


def gate_from_silver(run: str, min_prob: float, tag: str = SILVER_TAG):
    """The z-threshold where P(shown top-1 is acceptable) = MIN_PROB, fitted on silver.

    "Acceptable" is silver's hit@1 (the judge's grade ≥ 2) of the top-1 the bundle shows,
    i.e. the first glyph not rejected by the manual review. The z-signal is the stored
    standardized confidence. Returns (threshold, gate record, table rows)."""
    from sklearn.linear_model import LogisticRegression

    rows = stored_rankings(run, tag)
    ds = sorted({r["dataset"] for r in rows})
    golds = silver.with_extra({d: silver.eval_gold(d) for d in ds}, EXTRA_GRADES)
    x, y = [], []
    for r in rows:
        g = golds[r["dataset"]].get(r["task_id"])
        if g is not None:
            x.append(r["signal"])
            y.append(silver.task_metrics(g, r["top"])["hit1_served"])
    x, y = np.array(x), np.array(y)
    lr = LogisticRegression().fit(x.reshape(-1, 1), y)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])

    def z_at(p: float) -> float:
        if p <= 0 or a <= 0:
            return 0.0                    # z is never negative: the top-1 is ≥ the mean
        return max(0.0, (math.log(p / (1 - p)) - b) / a)

    table = []
    for p in sorted({0.0, 0.1, 0.2, 0.3, 0.5, 0.7, min_prob}):
        z = z_at(p)
        shown = x >= z
        table.append({"min_prob": p, "z": round(z, 3), "coverage": round(float(shown.mean()), 3),
                      "precision": round(float(y[shown].mean()), 3) if shown.any() else None})
    z = z_at(min_prob)
    row = next(t for t in table if t["min_prob"] == min_prob)
    gate = {"min_prob": min_prob, "fitted_on": f"silver dev {tag}, F1@{run}.v2+parent, "
            f"{len(x)} tasks, acceptable = judge grade >= 2",
            "platt": [round(a, 4), round(b, 4)], "coverage": row["coverage"],
            "precision": row["precision"]}
    return z, gate, table


def check_f1(run: str, out: Path, dataset_keys: list, tag: str = SILVER_TAG) -> None:
    """The bundle (ONNX query side, as the agenda runs it) against the stored PyTorch
    rankings on the dev tasks of DATASET_KEYS."""
    exported = bundle.BundleMatcher(out)
    want = set(dataset_keys)
    rows = [r for r in stored_rankings(run, tag) if r["dataset"] in want]
    by_id = {t["id"]: t for d in want for t in silver.eval_tasks(d)}
    rej = silver.rejected_canonical()
    golds = silver.with_extra({d: silver.eval_gold(d) for d in want}, EXTRA_GRADES)
    same = shown = 0
    dz, lat, diffs = [], [], []
    quality = {"bundle": [], "stored": []}       # (ndcg5_served, hit1_served) per task
    for r in rows:
        t = by_id[r["task_id"]]
        ref = next((n for n in (silver.canonical(n) for n in r["top"]) if n not in rej), None)
        q = exported.query_for(t)
        t0 = time.perf_counter()
        ranked, conf = exported.rank_and_signal(q, top_k=10)
        d = exported.gate.decide(ranked, conf)
        lat.append((time.perf_counter() - t0) * 1000)
        g = golds[r["dataset"]].get(r["task_id"])
        if g is not None:
            for side, top in (("bundle", [n for n, _ in ranked]), ("stored", r["top"])):
                m = silver.task_metrics(g, top)
                quality[side].append((m["ndcg5_served"], m["hit1_served"]))
        same += silver.canonical(d.icon) == ref
        shown += d.shown
        dz.append(abs(d.confidence - r["signal"]))
        if silver.canonical(d.icon) != ref:
            diffs.append((q, ref, d.icon))
    n = len(rows)
    print(f"\n{'+'.join(dataset_keys)} dev: {n} tasks, bundle (ONNX) vs stored F1@{run} (PyTorch)")
    print(f"  top-1 agreement  {same / n:.1%}")
    print(f"  shown            {shown / n:.1%}   (threshold z ≥ {exported.gate.threshold})")
    print(f"  |Δz| median {statistics.median(dz):.3f}, max {max(dz):.3f}")
    for side, label in (("bundle", "bundle (ONNX)"), ("stored", "stored (PyTorch)")):
        q_ = np.array(quality[side])
        print(f"  silver {label:17s} nDCG@5 {q_[:, 0].mean():.3f}  hit@1 {q_[:, 1].mean():.3f}"
              f"  ({len(q_)} tasks, served: rejected glyphs removed)")
    print(f"  latency p50 {statistics.median(lat):.1f} ms, p95 {sorted(lat)[int(0.95 * (n - 1))]:.1f} ms")
    for q, a, b in diffs[:10]:
        print(f"    {q!r}: {a} -> {b}")


def export_f1_bundle(args) -> dict:
    from tcr.finetune import FineTunedMatcher, model_dir

    path = model_dir(args.run)
    meta = json.loads((path / "f1_meta.json").read_text())
    size = meta["base_model"].rsplit("-", 1)[-1]                      # large / small / base
    repo = args.hf_repo or f"{HF_USER}/task-icon-encoder-e5-{size}-interim"
    onnx_dir = args.onnx_dir or (config.ROOT / ".cache" / "onnx" / f"F1-{args.run}")
    t0 = time.time()
    spec = bundle.export_f1_onnx(path, onnx_dir, args.precision, hf_repo=repo,
                                 revision=args.revision)
    print(f"ONNX encoder in {onnx_dir} ({spec['model']['size'] / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f} s)")
    # where this repo's OnnxEncoder looks, and the agenda runner's cache when the bundle
    # leaves this repo (hard links: no extra disk)
    caches = [config.CACHE_DIR / "onnx"]
    if not args.out.resolve().is_relative_to(config.ROOT):
        caches.append(RUNNER_CACHE)
    for key, fname in (("model", spec["model"]["url"].rsplit("/", 1)[-1]), ("tokenizer", "tokenizer.json")):
        for cache in caches:
            bundle.place_in_cache(spec[key], onnx_dir / fname, cache)

    icons = export_icons(args.keep_rejected, version="v2")
    names = [ic.name for ic in icons]
    full = FineTunedMatcher(load_icons(version="v2"), path, name=f"F1@{args.run}")
    row = {n: i for i, n in enumerate(full.names)}
    matrix = full.matrix[[row[n] for n in names]]
    del full

    z, gate, table = gate_from_silver(args.run, args.min_prob)
    print("gate on silver dev (shown = z ≥ threshold; precision = judge grade ≥ 2 among shown):")
    for t in table:
        mark = "  <-" if t["min_prob"] == args.min_prob else ""
        print(f"  P ≥ {t['min_prob']:.1f}: z ≥ {t['z']:.2f}  shown {t['coverage']:.1%}  "
              f"precision {t['precision']:.1%}{mark}")
    return bundle.export_f1(args.out, names, matrix, spec, z, gate,
                            source=f"task-concept-retrieval@{_git_rev()}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="M3", choices=["M1", "M3", "F1"])
    ap.add_argument("--run", help="F1 run tag, e.g. all-large-all3k (.cache/models/F1-<run>/)")
    ap.add_argument("--out", type=Path, default=config.BUNDLE_DIR)
    ap.add_argument("--check", action="store_true", help="compare the bundle with the in-repo method")
    ap.add_argument("--dataset", default="realistic",
                    help="dataset(s) used by --check (F1: comma-separated eval sets, dev split)")
    ap.add_argument("--keep-rejected", action="store_true",
                    help="also export the glyphs the manual icon review rejected")
    ap.add_argument("--precision", default="fp16", choices=bundle.F1_PRECISIONS, help="F1's ONNX encoder")
    ap.add_argument("--min-prob", type=float, default=0.2,
                    help="F1 gate: show the top-1 when P(acceptable) on silver is at least this")
    ap.add_argument("--onnx-dir", type=Path, default=None)
    ap.add_argument("--hf-repo", default=None,
                    help=f"Hugging Face repo the encoder is uploaded to (default {HF_USER}/task-icon-encoder-e5-<size>-interim)")
    ap.add_argument("--revision", default="main", help="the repo revision the manifest's URLs pin")
    args = ap.parse_args(argv)

    if args.method == "F1":
        if not args.run:
            ap.error("--method F1 needs --run")
        manifest = export_f1_bundle(args)
    else:
        manifest = bundle.export(args.out, export_icons(args.keep_rejected), bundle.OnnxEncoder(),
                                 method=args.method, source=f"task-concept-retrieval@{_git_rev()}")
    size = sum(p.stat().st_size for p in args.out.iterdir()) / 1e6
    print(f"wrote {args.out} ({manifest['tag']}, {size:.1f} MB)")
    if args.check:
        if args.method == "F1":
            check_f1(args.run, args.out, args.dataset.split(","))
        else:
            check(args.method, args.out, args.dataset, args.keep_rejected)


if __name__ == "__main__":
    main()
