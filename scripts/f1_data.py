#!/usr/bin/env python3
"""Build F1's training data (docs/f1-dual-encoder.md §2) into one tarball for Colab.

    python scripts/f1_data.py build [--smoke]
    python scripts/f1_data.py import <runs dir or f1_runs.zip>   # Kaggle / Colab results back

`import` copies each finished run's best/ checkpoint to .cache/models/F1[-<tag>]/ (where
the methods F1 and F1@<tag> load it) and its done.json + train_log.jsonl (metrics only) to
results/f1/; glyph vectors from scripts/i1_embed.py (any i1/*.npz in it) go to .cache/i1/.

Output: data/private/f1/f1_data[_smoke].tar.gz (private: it holds the Personal-style pool
text), unpacking to f1_data[_smoke]/ with
  manifest.json   counts per source / split / language, build notes
  icons.jsonl     one icon per line (row i = icon i): name (an id, never matched on), doc,
                  holdout, lookalikes (icon indices), near_misses (query texts)
  queries.jsonl   one training / validation query per line: task_id, source, split,
                  lang, text, text_parent (optional), grades {icon index: 3/2/1},
                  ambiguous (masked icon indices), hard_negs (icon indices)
  train_f1.py     the training script, so the tarball is all Colab / Kaggle needs
  i1_embed.py     glyph-image encoder for method I1 (tcr/methods/image.py)
  glyphs/<name>.png  each icon's glyph, 256 px, black on white (public)
No eval-set task or label is in it (asserted in tcr.finetune.build).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import finetune  # noqa: E402


def write(data: dict, smoke: bool) -> Path:
    name = "f1_data_smoke" if smoke else "f1_data"
    out = finetune.OUT_DIR / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps(data["manifest"], indent=1, ensure_ascii=False))
    for key in ("icons", "queries"):
        with open(out / f"{key}.jsonl", "w", encoding="utf-8") as f:
            for r in data[key]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for script in ("train_f1.py", "i1_embed.py"):
        shutil.copy(ROOT / "scripts" / script, out / script)
    from tcr.describe_v2 import glyph_png
    (out / "glyphs").mkdir()
    for r in data["icons"]:
        (out / "glyphs" / f"{r['name']}.png").write_bytes(glyph_png(r["name"], 256))
    tar = finetune.OUT_DIR / f"{name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=name)
    return tar


def import_runs(src: Path) -> None:
    import tempfile
    import zipfile
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="f1_runs_"))
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        base = tmp
    else:
        base = src
    found = [p.parent for p in base.rglob("done.json")]
    i1 = [p for p in base.rglob("*.npz") if p.parent.name == "i1"]
    if i1:
        from tcr.methods.image import I1_DIR
        I1_DIR.mkdir(parents=True, exist_ok=True)
        for p in i1:
            shutil.copy(p, I1_DIR / p.name)
            print(f"glyph vectors {p.name} -> {I1_DIR}")
    res = ROOT / "results" / "f1"
    res.mkdir(parents=True, exist_ok=True)
    for run in sorted(found):
        if not (run / "best").is_dir():
            print(f"skip {run.name}: no best/")
            continue
        dst = finetune.MODELS_DIR / run.name          # F1, F1-all, F1-0, F1-1k, F1-3k
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(run / "best", dst)
        done = json.loads((run / "done.json").read_text())
        done.get("args", {}).pop("out", None)        # a Drive path, not a result
        done.get("args", {}).pop("data", None)
        (res / f"{run.name}.done.json").write_text(json.dumps(done, indent=1))
        shutil.copy(run / "train_log.jsonl", res / f"{run.name}.train_log.jsonl")
        b, base = done.get("best") or {}, done.get("base") or {}
        print(f"{run.name} -> {dst}: val nDCG@5 {b.get('all', {}).get('ndcg5')} "
              f"(base {base.get('all', {}).get('ndcg5')}), step {b.get('step')} of {done.get('steps')}")
    if src.suffix == ".zip":
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "import"])
    ap.add_argument("src", nargs="?", help="import: the runs directory or f1_runs.zip")
    ap.add_argument("--smoke", action="store_true", help="200 pool tasks, 1 in 10 icons' free signal")
    args = ap.parse_args()
    if args.cmd == "import":
        import_runs(Path(args.src))
        return
    data = finetune.build(smoke=args.smoke)
    tar = write(data, args.smoke)
    print(json.dumps(data["manifest"], indent=1, ensure_ascii=False))
    icons = data["icons"]
    rng = random.Random(0)
    pool = [q for q in data["queries"] if q["source"].startswith("pool")]
    for q in rng.sample(pool, min(3, len(pool))):
        top = sorted(q["grades"].items(), key=lambda kv: -kv[1])[:4]
        print(f"\n[{q['source']} {q['lang']} {q['split']}] {q.get('text_parent') or q['text']}\n"
              f"  graded: {[(icons[int(i)]['name'], g) for i, g in top]}  "
              f"ambiguous: {len(q['ambiguous'])}  hard negatives: {len(q['hard_negs'])}")
    print(f"\n-> {tar} ({tar.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
