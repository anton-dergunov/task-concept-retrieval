"""Export a matcher as a standalone bundle for the agenda (see tcr/bundle.py).

    python scripts/export_bundle.py --method M3 --out DIR [--check]

DIR is typically agentic-org-planner's `icons/task-matcher/`. With --check, the
exported bundle is compared with the in-repo method (fp32 sentence-transformers)
on the Realistic tasks: top-1 agreement, gate agreement, and latency.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tcr import bundle, config, datasets  # noqa: E402
from tcr.data import load_icons  # noqa: E402
from tcr.methods import build_method  # noqa: E402
from tcr.org_tasks import query_text  # noqa: E402


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def check(method: str, out: Path, dataset: str) -> None:
    tasks = datasets.load(dataset)
    exported = bundle.BundleMatcher(out)
    reference = build_method(method)
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="M3", choices=["M1", "M3"])
    ap.add_argument("--out", type=Path, default=config.BUNDLE_DIR)
    ap.add_argument("--check", action="store_true", help="compare the bundle with the in-repo method")
    ap.add_argument("--dataset", default="realistic", help="dataset used by --check")
    args = ap.parse_args(argv)

    manifest = bundle.export(args.out, load_icons(), bundle.OnnxEncoder(),
                             method=args.method, source=f"task-concept-retrieval@{_git_rev()}")
    size = sum(p.stat().st_size for p in args.out.iterdir()) / 1e6
    print(f"wrote {args.out} ({manifest['tag']}, {size:.1f} MB)")
    if args.check:
        check(args.method, args.out, args.dataset)


if __name__ == "__main__":
    main()
