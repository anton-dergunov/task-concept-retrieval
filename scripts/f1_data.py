#!/usr/bin/env python3
"""Build F1's training data (docs/f1-dual-encoder.md §2) into one tarball for Colab.

    python scripts/f1_data.py build [--smoke]
    python scripts/f1_data.py repack [--smoke] [--teachers T,..]   # same data, current scripts (+ teacher scores)
    python scripts/f1_data.py import <runs dir, f1_runs.tar.zst or .zip>   # Kaggle / Colab results back

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
                  ambiguous (masked icon indices), hard_negs (icon indices). Sources:
                  pool-* (judged), free-* (v2 free signal), para-* (rewrites of judged
                  tasks, with orig_task_id), distill-* (unlabelled: distill-mslatte = MS-LaTTE
                  tasks in no pool or eval set, distill-cand = generated candidates; used
                  only by runs with a teacher, docs/next-data-and-distillation.md)
  vlm.jsonl       round 6's VLM labels (tcr/finetune.py VLM_LABELS): task_id, set (train /
                  cand), order (icon indices, the VLM's ranking of the big model's top 20),
                  grades {icon index: 0-3}; only tasks with rows here
  teacher/<run>.npz  teacher scores for this queries.jsonl, when an earlier build's rows are
                  an exact prefix of it: that build's scores (data/private/f1/teacher/,
                  from Kaggle) extended on the Mac to the new rewrite rows (--teachers), so
                  a Kaggle session needs no teacher checkpoint for the rewrite filter
  train_f1.py     the training script, so the tarball is all Colab / Kaggle needs
  i1_embed.py     glyph-image encoder for method I1 (tcr/methods/image.py)
  glyphs/<name>.png  each icon's glyph, 256 px, black on white (public)
No eval-set task or label is in it (asserted in tcr.finetune.build).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import finetune  # noqa: E402


def distill_sources() -> dict:
    """Unlabelled text for distillation: MS-LaTTE tasks in no pool or eval set (public) and,
    once generated, the Personal-style candidates (private, finetune.CANDIDATES_FILE)."""
    sys.path.insert(0, str(ROOT / "scripts" / "datasets"))
    from build_train_pools import mslatte_unused
    from tcr import silver
    out = {"distill-mslatte": mslatte_unused()}
    if finetune.CANDIDATES_FILE.exists():
        from tcr.datasets import read_jsonl
        out["distill-cand"] = read_jsonl(finetune.CANDIDATES_FILE)
    # Tasks a judge run has labelled since (round 3's selection) are pool rows, not
    # unlabelled text as well.
    judged = {t["id"] for runs in silver.POOL_RUNS.values() for run in runs for t in silver.run_tasks(run)}
    for k, v in out.items():
        out[k] = [t for t in v if t["id"] not in judged]
        print(f"{k}: {len(out[k])} tasks ({len(v) - len(out[k])} judged since, left out)", file=sys.stderr)
    return out


def write(data: dict, smoke: bool, teachers=(), threads: int = 2) -> Path:
    name = "f1_data_smoke" if smoke else "f1_data"
    out = finetune.OUT_DIR / name
    prev = out.with_name(name + ".prev")      # the last build, kept until the next one
    if out.exists():
        if prev.exists():
            shutil.rmtree(prev)
        out.rename(prev)
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps(data["manifest"], indent=1, ensure_ascii=False))
    for key in ("icons", "queries"):
        with open(out / f"{key}.jsonl", "w", encoding="utf-8") as f:
            for r in data[key]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_vlm(data, out)
    from tcr.describe_v2 import glyph_png
    (out / "glyphs").mkdir()
    for r in data["icons"]:
        (out / "glyphs" / f"{r['name']}.png").write_bytes(glyph_png(r["name"], 256))
    if teachers and not smoke:
        extend_teachers(prev / "queries.jsonl", out, teachers, threads)
    return pack(out)


def write_vlm(data: dict, out: Path) -> None:
    """Round 6's VLM labels as icon indices, for the tasks that have rows in this build."""
    index = {r["name"]: i for i, r in enumerate(data["icons"])}
    ids = {q["task_id"] for q in data["queries"]}
    from tcr.datasets import read_jsonl
    n = 0
    with open(out / "vlm.jsonl", "w", encoding="utf-8") as f:
        for s, path in finetune.VLM_LABELS.items():
            if not path.exists():
                continue
            for r in read_jsonl(path):
                if r["task_id"] not in ids:
                    continue
                order = [index[x] for x in r["order"] if x in index]
                grades = {str(index[x]): int(g) for x, g in r["grades"].items() if x in index}
                f.write(json.dumps({"task_id": r["task_id"], "set": s, "order": order, "grades": grades}) + "\n")
                n += 1
    print(f"vlm.jsonl: {n} tasks", file=sys.stderr)


def extend_teachers(old_q: Path, out: Path, tags, threads: int) -> None:
    """Teacher scores for the new build without a Kaggle session: if the last build's
    queries.jsonl is an exact prefix of this one, copy each named teacher's scores for it
    (data/private/f1/teacher/<tag>.npz, from Kaggle) and score only the appended rewrite rows
    (title form) on the Mac with the imported checkpoint (.cache/models/<tag>/, its
    icon_matrix.npz)."""
    import numpy as np
    new_q = out / "queries.jsonl"
    if not old_q.exists():
        sys.exit(f"no earlier build at {old_q}: cannot extend teacher scores")
    old_b, new_b = old_q.read_bytes(), new_q.read_bytes()
    if not new_b.startswith(old_b):
        sys.exit("the earlier build's queries.jsonl is not a prefix of the new one: its teacher "
                 "scores do not carry over; score on Kaggle (train_f1.py --score) instead")
    old_sha, new_sha = hashlib.sha1(old_b).hexdigest(), hashlib.sha1(new_b).hexdigest()
    n_old = old_b.count(b"\n")
    qs = [json.loads(l) for l in new_b.decode("utf-8").splitlines() if l.strip()]
    man = json.loads((out / "manifest.json").read_text())
    icons = [json.loads(l) for l in (out / "icons.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [man["passage_prefix"] + r["doc"] for r in icons]
    # Only what a run reads without distillation: the rewrite filter (Data.teacher_agrees)
    # looks at the title form of rewrite rows. Other new rows stay at -1 (no teacher target,
    # train_f1.Teacher.target returns None), so a KD run that should cover them scores the
    # whole build on Kaggle instead (~0.5 h per e5-large teacher, vs ~8.7 texts/s here).
    jobs = [(r, 0, qs[r]["text"]) for r in range(n_old, len(qs))
            if qs[r]["split"] == "train" and qs[r]["source"].startswith("para")]
    print(f"teacher scores: {n_old} rows carried over, {len(qs) - n_old} new ({len(jobs)} texts to score)",
          file=sys.stderr)
    (out / "teacher").mkdir(exist_ok=True)
    for tag in tags:
        src = finetune.OUT_DIR / "teacher" / f"{tag}.npz"
        d = np.load(src, allow_pickle=False)
        if str(d["queries_sha1"]) != old_sha:
            sys.exit(f"{src} belongs to another queries.jsonl than the last build")
        ck = finetune.MODELS_DIR / tag
        im = np.load(ck / "icon_matrix.npz", allow_pickle=False)
        doc_sha = [hashlib.sha1(x.encode()).hexdigest() for x in docs]
        if list(map(str, im["doc_sha1"])) != doc_sha:
            sys.exit(f"{ck}/icon_matrix.npz was built for other icon documents")
        idx = np.full((len(qs), 2, d["idx"].shape[2]), -1, d["idx"].dtype)
        val = np.zeros((len(qs), 2, d["val"].shape[2]), d["val"].dtype)
        idx[:n_old], val[:n_old] = d["idx"], d["val"]
        if jobs:
            import time
            import torch
            sys.path.insert(0, str(ROOT / "scripts"))
            from train_f1 import Encoder
            torch.set_num_threads(threads)
            t0 = time.time()
            enc = Encoder(str(ck), "cpu", False, False)
            I = torch.from_numpy(im["matrix"])
            k = d["idx"].shape[2]
            for s in range(0, len(jobs), 256):
                chunk = jobs[s:s + 256]
                Qv = enc.encode([man["query_prefix"] + t for _, _, t in chunk], man["max_query_len"], batch=64)
                v, ix = (float(d["scale"]) * Qv @ I.T).topk(k, dim=1)
                for j, (r, f, _) in enumerate(chunk):
                    idx[r, f], val[r, f] = ix[j].numpy(), v[j].numpy()
            print(f"  {tag}: {len(jobs)} texts in {time.time() - t0:.0f} s", file=sys.stderr)
            del enc
        np.savez_compressed(out / "teacher" / f"{tag}.npz", run=str(d["run"]), names=d["names"], idx=idx,
                            val=val, queries_sha1=new_sha, scale=d["scale"])


def pack(out: Path) -> Path:
    """Copy the current scripts into a built data dir and tar it. `repack` uses this alone:
    queries.jsonl stays byte-identical, so teacher scores keyed by its hash stay valid."""
    for script in ("train_f1.py", "i1_embed.py"):
        shutil.copy(ROOT / "scripts" / script, out / script)
    tar = finetune.OUT_DIR / f"{out.name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=out.name)
    return tar


def _unpack_tar_zst(src: Path, dst: Path) -> None:
    """f1_runs.tar.zst (Kaggle cell 4): macOS tar and Python 3.11's tarfile can't read zstd,
    so the Homebrew `zstd` binary decompresses; tarfile's "data" filter keeps every member
    inside dst (the archive is a download)."""
    import subprocess
    import tarfile
    if not shutil.which("zstd"):
        sys.exit("needs zstd: brew install zstd")
    tar = dst / "f1_runs.tar"
    with open(tar, "wb") as f:
        subprocess.run(["zstd", "-dc", str(src)], stdout=f, check=True)
    with tarfile.open(tar) as t:
        t.extractall(dst, filter="data")
    tar.unlink()


def import_runs(src: Path) -> None:
    import tempfile
    import zipfile
    packed = src.suffix == ".zip" or src.name.endswith(".tar.zst")
    if packed:
        tmp = Path(tempfile.mkdtemp(prefix="f1_runs_"))
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
        else:
            _unpack_tar_zst(src, tmp)
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
    teach = [p for p in base.rglob("*.npz") if p.parent.name == "teacher"]
    if teach:                     # teacher scores per training row: input of select_active.py
        tdir = finetune.OUT_DIR / "teacher"
        tdir.mkdir(parents=True, exist_ok=True)
        for p in teach:
            shutil.copy(p, tdir / p.name)
            print(f"teacher scores {p.name} -> {tdir}")
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
    if packed:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "repack", "import"])
    ap.add_argument("src", nargs="?", help="import: the runs directory, f1_runs.tar.zst or f1_runs.zip")
    ap.add_argument("--smoke", action="store_true", help="200 pool tasks, 1 in 10 icons' free signal")
    ap.add_argument("--no-distill", action="store_true", help="leave out the unlabelled distillation text")
    ap.add_argument("--teachers", default="",
                    help="build, repack: extend these teachers' scores to the new rows and ship them "
                         "(e.g. F1-all-large-e8,F1-all-large-all3k; see extend_teachers)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for --teachers")
    args = ap.parse_args()
    if args.cmd == "import":
        import_runs(Path(args.src))
        return
    if args.cmd == "repack":
        out = finetune.OUT_DIR / ("f1_data_smoke" if args.smoke else "f1_data")
        sha = hashlib.sha1((out / "queries.jsonl").read_bytes()).hexdigest()
        if args.teachers and not args.smoke:
            # Teachers scored on the build before this one (f1_data.prev, a prefix of it)
            # join the shipped ones: e.g. round 6b's session B ships F1-all-large-all3k,
            # the big model's rewrite filter, beside F1-all-large-e8.
            extend_teachers(finetune.OUT_DIR / "f1_data.prev" / "queries.jsonl", out,
                            [t for t in args.teachers.split(",") if t], args.threads)
        tar = pack(out)
        print(f"-> {tar} ({tar.stat().st_size / 1e6:.1f} MB); queries.jsonl sha1 {sha[:12]} unchanged")
        return
    data = finetune.build(smoke=args.smoke, distill=None if args.no_distill else distill_sources())
    tar = write(data, args.smoke, [t for t in args.teachers.split(",") if t], args.threads)
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
