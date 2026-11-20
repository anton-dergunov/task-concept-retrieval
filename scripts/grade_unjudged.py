#!/usr/bin/env python3
"""Grade the picks the judge never saw (docs/f1-dual-encoder.md §7,
docs/next-data-and-distillation.md step 5).

F1 and its variants put an icon outside the judge's candidate pool at top-1 on 12–15% of
dev tasks; silver counts those as misses, so close variants cannot be told apart. This
asks the same judge (each eval set's own model, prompt j4's rubric) to grade them, as an
add-on to its own labels, not a rerun: one request per task with the task, the judge's
earlier analysis and graded ranking (glyphs + grades, as the scale), and the new icons.
About 50 calibration icons whose grade the judge already gave (a ranked icon removed
from the reference list, or an icon it saw and left out: grade 0) are mixed in blind, to
measure agreement with its own labels.

    python scripts/grade_unjudged.py prepare --name g1 \
        --methods F1@all-base.v2+parent,F1@all-lr.v2+parent,F1@all-kd.v2+parent
    python scripts/grade_unjudged.py prepare --name g2 --methods ... --skip g1   # only new picks
    python scripts/grade_unjudged.py run --name g1 --max-usd <cap>
    python scripts/grade_unjudged.py status --name g1
    python scripts/silver_eval.py --methods ... --extra-grades g1   # gold + these grades

Output: data/private/judge/extra_grades/<name>.jsonl (private: Personal tasks), one line
per graded icon: dataset, task_id, icon, grade 0–3, why, model, calib (+ judge_grade).
State: data/private/judge/grade/<name>/ (one Vertex batch job per judge model).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config, silver  # noqa: E402
from tcr import judge as J  # noqa: E402
from tcr import vertex_batch as vb  # noqa: E402
from tcr.datasets import read_jsonl, split_of, write_jsonl  # noqa: E402

WORK = J.JUDGE_DIR / "grade"
SILVER_PRIVATE = config.PRIVATE_DIR / "silver"
NEW_ID0 = 101
MAX_REF = 8

GRADE = """\
Grade each NEW icon as a reminder for the TASK above, on the scale of your earlier \
ranking: 3 = shows the central subject; 2 = shows its context, domain or activity; \
1 = a partial or loose association; 0 = not a useful reminder for this task.
- Judge each new icon by the same rules as if it had been among your candidates; the \
ranked icons only show the scale. A new icon may deserve a higher grade than any you \
ranked, or none.
- Look at the image itself, not only its description.
- Return one entry per new icon, with its number, its grade and a short reason."""

SCHEMA = {"type": "object", "properties": {"grades": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "grade": {"type": "integer"},
                                     "why": {"type": "string"}},
    "required": ["id", "grade", "why"]}}}, "required": ["grades"]}


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def stored(label: str) -> Path:
    """The most recent stored ranking of a method label (silver_eval's per-task output)."""
    hits = sorted(SILVER_PRIVATE.glob(f"*/{label}.jsonl"), key=lambda p: p.stat().st_mtime)
    if not hits:
        sys.exit(f"no stored ranking for {label} in {SILVER_PRIVATE} (run silver_eval.py first)")
    return hits[-1]


def judge_model(dataset: str) -> str:
    run = silver.EVAL_RUNS[dataset][0]
    return json.loads((silver.JUDGE_BATCH / run / "meta.json").read_text())["model"]


# --- prepare --------------------------------------------------------------------------

def collect_pairs(methods, depth: int, datasets, skip=()) -> dict:
    """{(dataset, task_id): set of icons} the judge never saw, among each method's top
    `depth` on dev tasks; plus per-method counts. Icons graded in the earlier grade sets
    `skip` (EXTRA_GRADES/<name>.jsonl) are left out."""
    golds = {d: silver.load_gold(silver.EVAL_RUNS[d]) for d in datasets}
    rej = silver.rejected_canonical()
    done = {(r["dataset"], r["task_id"], silver.canonical(r["icon"]))
            for name in skip for r in read_jsonl(silver.EXTRA_GRADES / f"{name}.jsonl")}
    pairs, counts = defaultdict(set), Counter()
    for label in methods:
        for r in read_jsonl(stored(label)):
            d, tid = r["dataset"], r["task_id"]
            g = golds.get(d, {}).get(tid)
            if g is None or split_of(tid) != "dev":
                continue
            for n in (silver.canonical(x) for x in r["top"][:depth]):
                if n not in g.seen and n not in rej and (d, tid, n) not in done:
                    pairs[(d, tid)].add(n)
                    counts[label] += 1
    return pairs, counts, golds


def cmd_prepare(args) -> None:
    wd = WORK / args.name
    if (wd / "meta.json").exists():
        sys.exit(f"{args.name} exists; pick another name")
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    skip = [x for x in (args.skip or "").split(",") if x]
    pairs, counts, golds = collect_pairs(methods, args.depth, datasets, skip)
    for m in methods:
        print(f"{m}: {counts[m]} unjudged picks in the top {args.depth}", file=sys.stderr)
    # calibration: on a hash-chosen subset of the tasks, one icon with a known verdict
    keys = sorted(pairs, key=lambda k: _h("calib:" + k[1]))
    calib = {}
    for d, tid in keys:
        if len(calib) >= args.calib:
            break
        g = golds[d][tid]
        if _h("kind:" + tid) % 2 and len(g.order) >= 2:
            n = g.order[_h("pick:" + tid) % min(len(g.order), 4)]
            calib[(d, tid)] = (n, g.grades[n])
        else:
            left = sorted(g.seen - g.relevant - set(g.grades) - pairs[(d, tid)])
            if left:
                calib[(d, tid)] = (left[_h("pick:" + tid) % len(left)], 0)

    labels = {d: {r["task_id"]: r for r in read_jsonl(silver.label_path(silver.EVAL_RUNS[d][0]))}
              for d in datasets}
    ctxs = {d: J.Context(d, records=silver.run_tasks(silver.EVAL_RUNS[d][0])) for d in datasets}
    by_model = defaultdict(list)
    for (d, tid), new in sorted(pairs.items()):
        by_model[judge_model(d)].append((d, tid, sorted(new)))
    wd.mkdir(parents=True)
    meta = {"name": args.name, "methods": methods, "depth": args.depth, "datasets": datasets,
            "runs": {}, "pairs": sum(len(v) for v in pairs.values()), "tasks": len(pairs),
            "calib": len(calib)}
    est_total = 0.0
    for model, items in by_model.items():
        tag = model.replace(".", "-")
        run = vb.Run(wd / "rounds", tag, f"judge-grade/{args.name}", model)
        manifest = {}
        n_in = 0
        with open(run.input, "w") as f:
            for d, tid, new in items:
                g, ctx = golds[d][tid], ctxs[d]
                c = calib.get((d, tid))
                ref = [n for n in g.order if not (c and n == c[0])][:MAX_REF]
                newlist = list(new) + ([c[0]] if c else [])
                newlist.sort(key=lambda n: _h(tid + n))          # calibration icon not last
                key = f"{d}|{tid}"
                manifest[key] = {"dataset": d, "task_id": tid,
                                 "new": {str(NEW_ID0 + i): n for i, n in enumerate(newlist)},
                                 "calib": {"icon": c[0], "grade": c[1]} if c else None}
                parts = request_parts(ctx, ctx.by_id[tid], labels[d].get(tid, {}), g, ref, newlist)
                req = vb.request_json(J.SYSTEM, parts, SCHEMA, J.TEMPERATURE, J.GEN)
                f.write(json.dumps({"key": key, "request": req}, ensure_ascii=False) + "\n")
                n_in += 1
        run.save_manifest(manifest)
        est = n_in * PER_TASK.get(model, 0.012)
        est_total += est
        meta["runs"][tag] = {"model": model, "tasks": n_in, "estimate": round(est, 2)}
        print(f"{model}: {n_in} tasks, estimate ${est:.2f}", file=sys.stderr)
    (wd / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"prepared {args.name}: {meta['pairs']} unjudged picks on {meta['tasks']} tasks, "
          f"{len(calib)} calibration icons; estimate ${est_total:.2f} (spent so far ${J.spent():.2f})")


# USD per task in batch (one request with two contact sheets); refined after a first run
PER_TASK = {"gemini-4-argon": 0.012, "gemini-3.8-flash": 0.005}


def request_parts(ctx, rec: dict, lab: dict, g, ref: list, new: list) -> list:
    parts = J._header(ctx, rec)
    if lab.get("analysis"):
        parts.append("YOUR EARLIER ANALYSIS: " + json.dumps(lab["analysis"], ensure_ascii=False))
    if ref:
        numbered = [(i + 1, n) for i, n in enumerate(ref)]
        parts.append(f"YOUR EARLIER RANKING: {len(ref)} icons, best first.")
        parts += J.sheets_parts(numbered, "Ranked earlier")
        parts.append("Grades you gave: " + "; ".join(f"{i}: {g.grades[n]}" for i, n in numbered))
    else:
        parts.append("YOUR EARLIER RANKING: no icon fitted.")
    numbered = [(NEW_ID0 + i, n) for i, n in enumerate(new)]
    parts.append(f"NEW ICONS: {len(new)} icons you did not see before.")
    parts += J.sheets_parts(numbered, "New icons")
    return parts + [GRADE]


# --- run --------------------------------------------------------------------------------

def runs_of(name: str):
    wd = WORK / name
    meta = json.loads((wd / "meta.json").read_text())
    return wd, meta, {tag: vb.Run(wd / "rounds", tag, f"judge-grade/{name}", r["model"])
                      for tag, r in meta["runs"].items()}


def cmd_run(args) -> None:
    wd, meta, runs = runs_of(args.name)
    for tag, run in runs.items():
        if not run.state().get("job"):
            est = meta["runs"][tag]["estimate"]
            if args.max_usd is not None and J.spent() + est > args.max_usd:
                sys.exit(f"refusing: spent ${J.spent():.2f} + estimate ${est:.2f} > --max-usd {args.max_usd}")
            run.submit(f"grade-{args.name}-{tag}")
    rows = []
    for tag, run in runs.items():
        job = run.wait(args.poll)
        rows += collect(run, meta["runs"][tag]["model"], job, args.name)
    out = silver.EXTRA_GRADES / f"{args.name}.jsonl"
    write_jsonl(out, rows)
    report(rows)
    print(f"{len(rows)} grades -> {out}; total logged ${J.spent():.2f}")


def collect(run, model: str, job, name: str) -> list:
    manifest = run.manifest()
    records = run.download() if vb.succeeded(job) or "PARTIALLY" in str(job.state) else []
    rows, spent, bad = [], 0.0, 0
    for rec in records:
        m = manifest.get(rec.get("key"))
        if not m:
            continue
        resp = rec.get("response") or {}
        usage = vb.usage_of(resp) if resp else {}
        if usage:
            spent += J.log_spend(model, usage, f"batch:grade-{name}", batch=True)
        try:
            items = {int(x["id"]): x for x in json.loads(vb.response_text(resp))["grades"]}
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        c = m.get("calib")
        for k, icon in m["new"].items():
            x = items.get(int(k))
            if x is None or not 0 <= int(x["grade"]) <= 3:
                bad += 1
                continue
            row = {"dataset": m["dataset"], "task_id": m["task_id"], "icon": icon,
                   "grade": int(x["grade"]), "why": x.get("why", ""), "model": model,
                   "calib": bool(c and c["icon"] == icon)}
            if row["calib"]:
                row["judge_grade"] = c["grade"]
            rows.append(row)
    vb.log(f"{model}: {len(rows)} grades, {bad} unusable answers; ${spent:.2f}")
    return rows


def report(rows: list) -> None:
    cal = [r for r in rows if r["calib"]]
    if cal:
        exact = sum(r["grade"] == r["judge_grade"] for r in cal)
        near = sum(abs(r["grade"] - r["judge_grade"]) <= 1 for r in cal)
        hit = sum((r["grade"] >= 2) == (r["judge_grade"] >= 2) for r in cal)
        print(f"calibration ({len(cal)}): same grade {exact}, within 1: {near}, "
              f"same hit (grade ≥ 2) {hit}")
    new = [r for r in rows if not r["calib"]]
    if new:
        dist = Counter(r["grade"] for r in new)
        print("new picks by grade: " + ", ".join(f"{g}: {dist[g]}" for g in (3, 2, 1, 0)) +
              f"; hit (≥ 2) {sum(r['grade'] >= 2 for r in new) / len(new):.2f}")


def cmd_status(args) -> None:
    wd, meta, runs = runs_of(args.name)
    print(f"{args.name}: {meta['pairs']} picks on {meta['tasks']} tasks, {meta['calib']} calibration")
    for tag, run in runs.items():
        print(f"  {tag}: " + (vb.describe_job(run.job()) if run.state().get("job") else "not submitted")
              + f" (estimate ${meta['runs'][tag]['estimate']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--name", required=True)
    p.add_argument("--methods", required=True, help="method labels as in the silver tables")
    p.add_argument("--depth", type=int, default=1, help="grade unjudged picks in the top N")
    p.add_argument("--calib", type=int, default=50)
    p.add_argument("--skip", default="", help="earlier grade sets NAME[,NAME]: their icons are not graded again")
    p.add_argument("--datasets", default=",".join(silver.EVAL_RUNS))
    for name in ("run", "status"):
        q = sub.add_parser(name)
        q.add_argument("--name", required=True)
        if name == "run":
            q.add_argument("--max-usd", type=float, default=None)
            q.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
