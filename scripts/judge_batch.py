#!/usr/bin/env python3
"""Run the LLM judge (tcr/judge.py) over a whole dataset with Vertex batch jobs.

The judge's stages become rounds: round 1 sends every task's stage A, round 2 its stage
B (after the searches ran locally), round 3 the B of tasks that searched again. Each
round is split into jobs of at most --chunk tasks, so partial results land early. The
requests are exactly those of the online runner, at half the price.

State lives in data/private/judge/batch/<run>/ (private: requests carry the prompt's
examples), so every command resumes in a later session:

  prepare   build the task states and round 1        (--dataset KEY | --pool FILE --name N)
  estimate  the expected cost, from the online runs' measured cost per task
  submit    submit the current round's jobs that are not submitted yet
  status    the state of the current round's jobs
  collect   when every job of the round ended: apply the answers, run the searches,
            prepare the next round (or finish)
  run       submit → wait → collect until done (for a terminal; polls every --poll s)
  export    final labels: private datasets → data/private/judge/labels/<run>.jsonl,
            public ones → results/judge_labels/<run>.jsonl

Run: python scripts/judge_batch.py prepare --run personal-j3 --dataset personal
     python scripts/judge_batch.py run --run personal-j3 --max-usd 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import judge as J  # noqa: E402
from tcr import vertex_batch as vb  # noqa: E402
from tcr.datasets import DATASETS, read_jsonl, write_jsonl  # noqa: E402

BATCH_DIR = J.JUDGE_DIR / "batch"
MAX_ERRORS = 2          # a task whose answer fails this often is finished as "failed"
DEFAULT_EST = 0.0075    # USD per task in batch, if no online run measured it yet


class Batch:
    def __init__(self, run: str):
        self.run = run
        self.dir = BATCH_DIR / run
        self._ctx = None

    # --- files ---
    @property
    def meta(self) -> dict:
        return json.loads((self.dir / "meta.json").read_text())

    def save_meta(self, meta: dict) -> None:
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1))

    def states(self) -> list:
        return read_jsonl(self.dir / "states.jsonl")

    def save_states(self, states: list) -> None:
        write_jsonl(self.dir / "states.jsonl", states)

    def records(self) -> list:
        m = self.meta
        return read_jsonl(Path(m["source"]))

    def ctx(self) -> J.Context:
        if self._ctx is None:
            m = self.meta
            self._ctx = J.Context(m["dataset"], records=self.records())
        return self._ctx

    def job(self, name: str) -> vb.Run:
        return vb.Run(self.dir / "rounds", name, f"judge/{self.run}", self.meta["model"])

    # --- rounds ---
    def build_round(self, states: list) -> list:
        """Write the input of the next round for every unfinished task; return job names."""
        meta = self.meta
        k = meta.get("round", 0) + 1
        pending = [s for s in states if s["stage"] != "done"]
        if not pending:
            return []
        if meta["prompt_version"] != J.PROMPT_VERSION:
            sys.exit(f"this run uses prompt {meta['prompt_version']} but the code is at "
                     f"{J.PROMPT_VERSION}; finish it with that version checked out")
        ctx, names = self.ctx(), []
        for j in range(0, len(pending), meta["chunk"]):
            name = f"r{k}_c{j // meta['chunk'] + 1}"
            run = self.job(name)
            with open(run.input, "w") as f:
                for s in pending[j:j + meta["chunk"]]:
                    system, parts, schema = J.request(ctx, s)
                    req = vb.request_json(system, parts, schema, J.TEMPERATURE, meta["gen"])
                    key = f"{s['task_id']}|{s['stage']}{s['b_round']}"
                    f.write(json.dumps({"key": key, "request": req}, ensure_ascii=False) + "\n")
            names.append(name)
        meta.update(round=k, jobs=names)
        self.save_meta(meta)
        print(f"round {k}: {len(pending)} tasks in {len(names)} job(s)")
        return names

    def submit(self, max_usd=None) -> None:
        meta = self.meta
        todo = [n for n in meta.get("jobs", []) if not self.job(n).state().get("job")]
        if not todo:
            return
        if max_usd is not None:
            est = estimate(self, remaining_only=True)
            if J.spent() + est > max_usd:
                sys.exit(f"refusing: spent ${J.spent():.2f} + estimate ${est:.2f} "
                         f"> --max-usd {max_usd}")
        for name in todo:
            self.job(name).submit(f"judge-{self.run}-{name}")

    def status(self) -> bool:
        """Print the jobs' states; True when every job of the round has ended."""
        done = True
        for name in self.meta.get("jobs", []):
            run = self.job(name)
            if not run.state().get("job"):
                print(f"  {name}: not submitted")
                done = False
                continue
            job = run.job()
            print(f"  {name}: {vb.describe_job(job)}")
            done &= vb.ended(job)
        return done

    def collect(self) -> bool:
        """Apply the round's answers; prepare the next round. True when all tasks are done."""
        meta = self.meta
        states = {s["task_id"]: s for s in self.states()}
        ctx, answered, failed, spent = self.ctx(), 0, 0, 0.0
        for name in meta.get("jobs", []):
            run = self.job(name)
            if run.state().get("collected"):
                continue
            for rec in run.download():
                task_id, step = rec["key"].rsplit("|", 1)
                s = states.get(task_id)
                if s is None or f"{s['stage']}{s['b_round']}" != step:
                    continue
                resp = rec.get("response") or {}
                usage = vb.usage_of(resp) if resp else {}
                if usage:
                    spent += J.log_spend(meta["model"], usage, f"batch:{self.run}:{name}",
                                         batch=True)
                try:
                    data = json.loads(vb.response_text(resp))
                except Exception as e:  # noqa: BLE001 — retried next round
                    s["errors"] = s.get("errors", 0) + 1
                    s["last_error"] = (rec.get("status") or str(e))[:300]
                    if s["errors"] >= MAX_ERRORS:
                        s["final"] = {"ranking": [], "no_good_icon": False, "failed": True}
                        s["stage"] = "done"
                    failed += 1
                    continue
                J.advance(ctx, s, data, {"model": meta["model"], "usage": usage,
                                         "usd": round(J.usd(usage, meta["model"], batch=True), 6),
                                         "cached": False, "batch": name})
                answered += 1
            st = run.state()
            st["collected"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            run.save_state(st)
        all_states = list(states.values())
        for s in all_states:
            s["usd"] = round(sum(c.get("usd", 0) for c in s["calls"]), 6)
            s["model"] = meta["model"]
        self.save_states(all_states)
        n_done = sum(s["stage"] == "done" for s in all_states)
        print(f"collected round {meta.get('round')}: {answered} answers, {failed} failed; "
              f"${spent:.2f}; {n_done}/{len(all_states)} tasks done; total logged ${J.spent():.2f}")
        names = self.build_round(all_states)
        if not names:
            meta = self.meta
            meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            meta["jobs"] = []
            self.save_meta(meta)
            return True
        return False


def estimate(b: Batch, remaining_only: bool = False) -> float:
    """Expected batch cost: the online runs' mean cost per task (same model and prompt,
    if any) at batch prices, times the tasks still to do."""
    meta = b.meta
    per_task = DEFAULT_EST
    runs = sorted((J.JUDGE_DIR / "runs").glob(f"{meta['prompt_version']}__{meta['model']}__rep*.jsonl"))
    costs = [s["usd"] for f in runs for s in map(json.loads, f.read_text().splitlines()) if s]
    if costs:
        per_task = sum(costs) / len(costs) / 2      # batch is half the online price
    states = b.states()
    n = sum(s["stage"] != "done" for s in states) if remaining_only else len(states)
    if remaining_only and meta.get("round", 1) > 1:
        n *= 0.4                                    # later rounds are stage B only
    return n * per_task


def cmd_prepare(args) -> None:
    b = Batch(args.run)
    if (b.dir / "meta.json").exists():
        sys.exit(f"run {args.run} exists; pick another name")
    if args.dataset:
        ds = DATASETS[args.dataset]
        source, name, private = ds.path, ds.key, ds.private
    else:
        source, name, private = Path(args.pool).resolve(), args.name, args.private
    recs = read_jsonl(source)
    if args.limit:
        recs = recs[:args.limit]
    b.dir.mkdir(parents=True, exist_ok=True)
    sub = b.dir / "tasks.jsonl"        # the exact task list of this run
    write_jsonl(sub, recs)
    b.save_meta({"run": args.run, "dataset": name, "source": str(sub), "private": private,
                 "model": args.model, "prompt_version": J.PROMPT_VERSION, "gen": J.GEN,
                 "chunk": args.chunk, "n_tasks": len(recs),
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    ctx = b.ctx()
    t0 = time.time()
    states = []
    for i, r in enumerate(recs):
        states.append(J.start(ctx, r))
        if (i + 1) % 500 == 0:
            print(f"  pools: {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)")
    b.save_states(states)
    b.build_round(states)
    print(f"prepared {args.run}: {len(recs)} tasks of {name}; estimate ${estimate(b):.2f} "
          f"(spent so far ${J.spent():.2f})")


def cmd_estimate(args) -> None:
    b = Batch(args.run)
    print(f"{args.run}: total ≈ ${estimate(b):.2f}, remaining ≈ ${estimate(b, True):.2f}; "
          f"spent so far ${J.spent():.2f}")


def cmd_submit(args) -> None:
    Batch(args.run).submit(args.max_usd)


def cmd_status(args) -> None:
    b = Batch(args.run)
    m = b.meta
    print(f"{args.run}: round {m.get('round')}, finished: {m.get('finished', 'no')}")
    b.status()


def cmd_collect(args) -> None:
    b = Batch(args.run)
    if not b.status():
        sys.exit("not every job of this round has ended yet")
    b.collect()


def cmd_run(args) -> None:
    b = Batch(args.run)
    while not b.meta.get("finished"):
        b.submit(args.max_usd)
        while not b.status():
            time.sleep(args.poll)
        if b.collect():
            break
    print(f"{args.run}: finished; export with `python scripts/judge_batch.py export --run {args.run}`")


def cmd_export(args) -> None:
    b = Batch(args.run)
    m = b.meta
    out_dir = (J.JUDGE_DIR / "labels") if m["private"] else (config.RESULTS_DIR / "judge_labels")
    rows = []
    for s in b.states():
        if s["stage"] != "done":
            continue
        f = s["final"]
        rows.append({"task_id": s["task_id"], "dataset": m["dataset"], "model": m["model"],
                     "prompt_version": s["prompt_version"], "analysis": s["analysis"],
                     "ranking": [{"icon": r["icon"], "grade": r["grade"], "why": r["why"]}
                                 for r in f["ranking"]],
                     "no_good_icon": f["no_good_icon"], "failed": bool(f.get("failed")),
                     "searches": [x["query"] for x in s["searches"]]})
    out = out_dir / f"{args.run}.jsonl"
    write_jsonl(out, rows)
    print(f"{len(rows)} labels -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=list(DATASETS))
    g.add_argument("--pool", help="a JSONL of tasks in the shared schema (training pool)")
    p.add_argument("--name", help="dataset name for --pool")
    p.add_argument("--private", action="store_true", help="--pool holds private text")
    p.add_argument("--model", default=J.DEFAULT_MODEL)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0)
    for name in ("estimate", "submit", "status", "collect", "run", "export"):
        q = sub.add_parser(name)
        q.add_argument("--run", required=True)
        if name in ("submit", "run"):
            q.add_argument("--max-usd", type=float, default=None)
        if name == "run":
            q.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    if args.cmd == "prepare" and args.pool and not args.name:
        ap.error("--pool needs --name")
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
