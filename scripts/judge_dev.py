#!/usr/bin/env python3
"""Develop the LLM judge on the judge-dev tasks (docs/fine-tuning-plan.md §6).

The judge-dev set is the 51 Personal tasks the author labelled first
(data/private/judge/judge_dev_ids.txt): 5 are examples in the prompt
(data/private/judge/fewshot.json), the other 46 validate it.

  run    judge the validation tasks online (cached; spend logged) → runs/<name>.jsonl
  score  metrics of every run (or the named ones) against the author's rankings, with
         the matchers as baselines → data/private/judge/scores.json (per task, private)
         and results/judge_dev/summary.json (metrics only, committable)

Run: python scripts/judge_dev.py run --model gemini-3.8-flash --rep 1 [--max-usd 20]
     python scripts/judge_dev.py score
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import judge as J  # noqa: E402
from tcr.datasets import judge_dev_ids, load  # noqa: E402
from tcr.judging import wilson  # noqa: E402

RUNS = J.JUDGE_DIR / "runs"
LABELS = config.PRIVATE_DIR / "labels" / "personal.jsonl"
SUMMARY = config.RESULTS_DIR / "judge_dev" / "summary.json"
BASELINES = [("M3.v2", "M3", "title"), ("M1.v2", "M1", "body"), ("M1p.v2", "M1", "parent")]


def human_labels() -> dict:
    """{task id: {"ranking": [...], "none": bool}} for the judge-dev tasks."""
    last = {}
    for line in LABELS.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            last[e["task_id"]] = e
    ids = judge_dev_ids()
    return {t: {"ranking": e["ranking"], "none": e["status"] == "none"}
            for t, e in last.items() if t in ids and e["status"] in ("labelled", "none")}


def validation_ids() -> list:
    shots = {s["task_id"] for s in json.loads(J.FEWSHOT_PATH.read_text())}
    return sorted(t for t in human_labels() if t not in shots)


def run_name(model: str, rep: int, suffix: str = "") -> str:
    return f"{J.PROMPT_VERSION}__{model}__rep{rep}" + (f"__{suffix}" if suffix else "")


def cmd_run(args) -> None:
    ids = validation_ids()[: args.limit or None]
    name = run_name(args.model, args.rep, args.suffix)
    out = RUNS / f"{name}.jsonl"
    est = len(ids) * args.est_per_task
    if args.max_usd is not None and J.spent() + est > args.max_usd:
        sys.exit(f"refusing: spent ${J.spent():.2f} + estimate ${est:.2f} > --max-usd {args.max_usd}")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "prompts").mkdir(exist_ok=True)
    (RUNS / "prompts" / f"{J.PROMPT_VERSION}.json").write_text(json.dumps(
        {"system": J.SYSTEM, "stage_a": J.STAGE_A, "stage_b": J.STAGE_B, "gen": J.GEN},
        ensure_ascii=False, indent=1))
    recs = {r["id"]: r for r in load("personal")}
    ctx = J.Context("personal")
    gen = dict(J.GEN, **({"thinking_level": args.thinking} if args.thinking else {}))
    tag = f"{J.PROMPT_VERSION}|{args.model}|rep{args.rep}|{args.suffix}"
    t0, done, states = time.time(), 0, {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(J.judge_online, ctx, recs[t], args.model, tag, True, gen): t
                for t in ids}
        for f in as_completed(futs):
            t = futs[f]
            try:
                states[t] = f.result()
            except Exception as e:  # noqa: BLE001 — report and continue
                print(f"  {t}: FAILED {str(e)[:200]}")
                continue
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(ids)} ({time.time() - t0:.0f}s)")
    with open(out, "w") as f:
        for t in ids:
            if t in states:
                f.write(json.dumps(states[t], ensure_ascii=False) + "\n")
    cost = sum(s["usd"] for s in states.values() if not all(c["cached"] for c in s["calls"]))
    print(f"{name}: {len(states)}/{len(ids)} tasks -> {out}; new spend ${cost:.2f}; "
          f"total logged ${J.spent():.2f}")


# --- scoring -----------------------------------------------------------------------------

def gain(user: list, icon: str) -> float:
    if icon not in user:
        return 0.0
    i = user.index(icon)
    return 3.0 if i == 0 else 2.0 if i == 1 else 1.0


def ndcg(user: list, ranked: list, k: int = 5) -> float:
    dcg = sum(gain(user, n) / math.log2(i + 2) for i, n in enumerate(ranked[:k]))
    ideal = sorted((gain(user, n) for n in user), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def task_metrics(lab: dict, ranked: list, abstained: bool, seen=None, collected=None) -> dict:
    u = lab["ranking"]
    if lab["none"]:
        return {"none_task": 1, "agree_none": int(abstained or not ranked)}
    top = [] if abstained else ranked
    m = {"none_task": 0, "hit1": int(bool(top) and top[0] in u),
         "exact1": int(bool(top) and top[0] == u[0]), "ndcg5": ndcg(u, top),
         "abstained": int(abstained), "u1_rank": (ranked.index(u[0]) + 1) if u[0] in ranked else None}
    if seen is not None:
        m["u1_seen"] = int(u[0] in seen)
        m["u1_collected"] = int(u[0] in collected)
        m["any_collected"] = int(any(x in collected for x in u))
    return m


_USER_SAW = None


def user_saw(task_id: str) -> set:
    """The icons the author saw for a task: the bundle's 100 plus what they searched."""
    global _USER_SAW
    if _USER_SAW is None:
        bundle = {r["id"]: r for r in map(json.loads, (J.BUNDLE_TASKS / "personal.jsonl")
                                           .read_text().splitlines())}
        last = {}
        for line in LABELS.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                last[e["task_id"]] = e
        _USER_SAW = {t: set(b["icons"]) | set(b.get("more", []))
                     | set((last.get(t) or {}).get("search_added") or []) for t, b in bundle.items()}
    return _USER_SAW.get(task_id, set())


def state_metrics(lab: dict, s: dict) -> dict:
    ids = s["ids"]
    ranked = [r["icon"] for r in s["final"]["ranking"]]
    seen = set(ids.values())
    collected = {ids[str(i)] for i in s["shortlist"]} | set(ranked)
    m = task_metrics(lab, ranked, s["final"]["no_good_icon"] and not ranked, seen, collected)
    m["usd"] = s["usd"]
    m["n_searches"] = len(s["searches"])
    if not lab["none"] and ranked and not m["abstained"]:
        # hit@1 only where the author saw the judge's top-1 (else it is unjudged)
        judged = ranked[0] in user_saw(s["task_id"])
        m["unjudged1"] = int(not judged)
        m["hit1_judged"] = m["hit1"] if judged else None
    return m


def aggregate(per_task: dict) -> dict:
    lab = [m for m in per_task.values() if not m["none_task"]]
    out = {"n": len(lab)}
    for k in ("hit1", "hit1_judged", "unjudged1", "exact1", "ndcg5", "u1_seen", "u1_collected", "any_collected",
              "abstained", "n_searches", "usd"):
        vals = [m[k] for m in lab if m.get(k) is not None]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    if "hit1" in out:
        k = sum(m["hit1"] for m in lab)
        out["hit1_ci"] = [round(x, 3) for x in wilson(k, len(lab))]
    nones = [m for m in per_task.values() if m["none_task"]]
    if nones:
        out["none_agree"] = f"{sum(m['agree_none'] for m in nones)}/{len(nones)}"
    if lab and "usd" in lab[0]:
        out["usd_total"] = round(sum(m["usd"] for m in per_task.values() if "usd" in m), 4)
    return out


def baseline_rankings(ids: list) -> dict:
    from tcr.data import load_icons
    from tcr.methods import build_method
    from tcr.org_tasks import query_text
    icons = load_icons(version="v2")
    built = {k: build_method(k, icons) for k in {k for _, k, _ in BASELINES}}
    recs = {r["id"]: r for r in load("personal")}
    out = {}
    for label, key, view in BASELINES:
        out[label] = {}
        for t in ids:
            r = recs[t]
            q = query_text(r["title"], r.get("body", ""), with_body=(view == "body"),
                           parents=(r.get("parents") or []) if view == "parent" else ())
            out[label][t] = [n for n, _ in built[key].rank(q, top_k=10)]
    return out


def cmd_score(args) -> None:
    labels = human_labels()
    ids = validation_ids()
    files = sorted(RUNS.glob("*.jsonl"))
    if args.runs:
        files = [f for f in files if any(r in f.stem for r in args.runs)]
    results, per_task_all = {}, {}
    for f in files:
        states = {s["task_id"]: s for s in map(json.loads, f.read_text().splitlines()) if s}
        per = {t: state_metrics(labels[t], states[t]) for t in ids if t in states}
        results[f.stem] = aggregate(per)
        per_task_all[f.stem] = per
    for label, ranked in baseline_rankings(ids).items():
        per = {t: task_metrics(labels[t], ranked[t], False) for t in ids}
        results[label] = aggregate(per)
        per_task_all[label] = per

    cols = ["n", "hit1", "hit1_judged", "unjudged1", "exact1", "ndcg5", "u1_collected", "any_collected", "u1_seen",
            "abstained", "n_searches", "none_agree", "usd"]
    print(f"{'run':52}" + "".join(f"{c[:9]:>10}" for c in cols))
    for name, agg in results.items():
        print(f"{name[:52]:52}" + "".join(
            f"{(agg.get(c) if agg.get(c) is not None else ''):>10}" for c in cols))
    # Run-to-run noise: the same config's two reps, per task.
    by_cfg = {}
    for name in per_task_all:
        if "__rep" in name:
            cfg = name.replace("__rep1", "").replace("__rep2", "")
            by_cfg.setdefault(cfg, []).append(name)
    for cfg, names in by_cfg.items():
        if len(names) == 2:
            a, b = (per_task_all[n] for n in names)
            common = [t for t in a if t in b and not a[t]["none_task"]]
            flips = sum(a[t]["hit1"] != b[t]["hit1"] for t in common)
            print(f"noise {cfg}: hit@1 differs between reps on {flips}/{len(common)} tasks")
    J.JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    (J.JUDGE_DIR / "scores.json").write_text(json.dumps(per_task_all, indent=0))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(results, indent=1))
    print(f"-> {SUMMARY}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default=J.DEFAULT_MODEL)
    r.add_argument("--rep", type=int, default=1)
    r.add_argument("--suffix", default="", help="names a variant (e.g. thinking level)")
    r.add_argument("--thinking", choices=["minimal", "low", "medium", "high"])
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--max-usd", type=float, default=None)
    r.add_argument("--est-per-task", type=float, default=0.03)
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="*")
    args = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
