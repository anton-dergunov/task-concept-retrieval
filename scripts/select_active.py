#!/usr/bin/env python3
"""Choose the tasks for the next Argon judging (docs/next-data-and-distillation.md step 4):
2,000 where the models are unsure or disagree, plus 1,000 at random as the control that
says whether selection beat random picking.

Input: the unlabelled candidate rows of the F1 data (distill-cand: generated
Personal-style tasks, distill-mslatte: MS-LaTTE tasks in no pool or eval set) and teacher
scores of those rows written on Kaggle by `train_f1.py --score <teacher>,<student>[,...]`
and brought back by `f1_data.py import` (data/private/f1/teacher/<run>.npz).

Priority of a candidate (title form):
  1 − p₁ (the teacher's top-1 probability over its top 64 at the training scale)
  + 0.5 · [the teacher's top-1 is not in the student's top 5]
  + 0.5 · [two teachers disagree on the top-1], when a second teacher is given.
Filters: the 15% of candidates with the lowest teacher top-1 score (junk, nothing fits)
are not chosen actively; at most --per-icon tasks share a teacher top-1; a task whose
teacher top-10 overlaps a chosen one's by ≥ 7 is a near-duplicate topic and skipped.

    python scripts/select_active.py --teacher F1-all-base-lr --student F1-all-lr \
        [--teacher2 F1-all-large] [--active 2000 --random 1000]

Output (pool files for scripts/judge_batch.py prepare --pool …):
  data/private/train_pools/personal_synth_pool3.jsonl   Personal-style (private)
  data/train_pools/mslatte2.jsonl                       MS-LaTTE (public)
each task with meta.selection = active | random and meta.priority, plus
<file>.extra.json: the teacher's top 15 per task, which judge_batch adds to the judge's
candidate pool (--extra-candidates), so the icons the models favour are judged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "datasets"))

from tcr import config, finetune  # noqa: E402
from tcr.datasets import read_jsonl, write_jsonl  # noqa: E402

TEACHER_DIR = finetune.OUT_DIR / "teacher"
DATA_DIR = finetune.OUT_DIR / "f1_data"
OUT = {"distill-cand": config.PRIVATE_DIR / "train_pools" / "personal_synth_pool3.jsonl",
       "distill-mslatte": config.DATA_DIR / "train_pools" / "mslatte2.jsonl"}
SEED = 20261120


def load_scores(tag: str, sha: str) -> dict:
    p = TEACHER_DIR / f"{tag}.npz"
    if not p.exists():
        sys.exit(f"no teacher scores {p} (train_f1.py --score on Kaggle, then f1_data.py import)")
    d = np.load(p)
    if str(d["queries_sha1"]) != sha:
        sys.exit(f"{p} was scored on another queries.jsonl than {DATA_DIR} (rebuild or re-score)")
    return {"idx": d["idx"], "val": d["val"].astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True, help="teacher score tag (file stem in teacher/)")
    ap.add_argument("--student", required=True, help="the deployed-size model's score tag")
    ap.add_argument("--teacher2", default=None, help="a second teacher (disagreement)")
    ap.add_argument("--active", type=int, default=2000)
    ap.add_argument("--random", type=int, default=1000)
    ap.add_argument("--per-icon", type=int, default=4)
    ap.add_argument("--junk-quantile", type=float, default=0.15)
    ap.add_argument("--mslatte-max", type=float, default=0.33, help="largest MS-LaTTE share")
    args = ap.parse_args()

    qp = DATA_DIR / "queries.jsonl"
    sha = hashlib.sha1(qp.read_bytes()).hexdigest()
    qs = read_jsonl(qp)
    cand = [i for i, q in enumerate(qs) if q["source"] in OUT]
    if not cand:
        sys.exit("no candidate rows in the F1 data; generate them and rerun f1_data.py build")
    T, S = load_scores(args.teacher, sha), load_scores(args.student, sha)
    T2 = load_scores(args.teacher2, sha) if args.teacher2 else None

    tv, ti = T["val"][cand, 0], T["idx"][cand, 0]
    p = np.exp(tv - tv.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    top1 = ti[:, 0]
    pri = 1 - p[:, 0]
    pri += 0.5 * np.array([t not in set(s) for t, s in zip(top1, S["idx"][cand, 0, :5])])
    if T2 is not None:
        pri += 0.5 * (T2["idx"][cand, 0, 0] != top1)
    floor = np.quantile(tv[:, 0], args.junk_quantile)

    order = np.argsort(-pri)
    chosen, per_icon, sigs, n_ms = [], Counter(), [], 0
    for k in order:
        if len(chosen) >= args.active:
            break
        if tv[k, 0] < floor or per_icon[int(top1[k])] >= args.per_icon:
            continue
        is_ms = qs[cand[k]]["source"] == "distill-mslatte"
        if is_ms and n_ms >= args.mslatte_max * args.active:
            continue
        sig = set(ti[k, :10].tolist())
        if any(len(sig & s) >= 7 for s in sigs[-400:]):
            continue
        chosen.append(k)
        sigs.append(sig)
        per_icon[int(top1[k])] += 1
        n_ms += is_ms
    rest = sorted(set(range(len(cand))) - set(chosen))
    rnd = random.Random(SEED).sample(rest, min(args.random, len(rest)))

    sys.path.insert(0, str(ROOT / "scripts" / "datasets"))
    from build_train_pools import mslatte_unused
    recs = {r["id"]: r for r in read_jsonl(finetune.CANDIDATES_FILE)} if finetune.CANDIDATES_FILE.exists() else {}
    recs.update({r["id"]: r for r in mslatte_unused()})
    names = [r["name"] for r in read_jsonl(DATA_DIR / "icons.jsonl")]
    out = {src: [] for src in OUT}
    extra = {src: {} for src in OUT}
    for sel, ks in (("active", chosen), ("random", rnd)):
        for k in ks:
            q = qs[cand[k]]
            r = dict(recs[q["task_id"]])
            r["meta"] = {**(r.get("meta") or {}), "selection": sel, "priority": round(float(pri[k]), 4)}
            out[q["source"]].append(r)
            extra[q["source"]][r["id"]] = [names[int(i)] for i in ti[k, :15] if i >= 0]
    for src, rows in out.items():
        if rows:
            write_jsonl(OUT[src], rows)
            OUT[src].with_suffix(".extra.json").write_text(json.dumps(extra[src], indent=0))
            sel = Counter(r["meta"]["selection"] for r in rows)
            print(f"{src}: {len(rows)} tasks ({dict(sel)}) -> {OUT[src]}")
    print(f"candidates {len(cand)}; active {len(chosen)} (mean priority {pri[chosen].mean():.3f}, "
          f"{len(set(int(top1[k]) for k in chosen))} distinct teacher top-1s; all candidates "
          f"{pri.mean():.3f}); random {len(rnd)}")


if __name__ == "__main__":
    main()
