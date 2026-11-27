#!/usr/bin/env python3
"""Round 6 #5: Argon reranks the big model's top 20 as glyph images (tcr/vlm.py).

    nice -n 19 python scripts/vlm_rerank.py topk --sets dev,val,train,cand   # Mac, 1 thread
    python scripts/vlm_rerank.py pilot --arms img+desc,img --max-usd <cap>    # online, 40 tasks
    python scripts/vlm_rerank.py pilot-report                                 # private HTML
    python scripts/vlm_rerank.py prepare --set dev [--arm img+desc] [--limit 3]
    python scripts/vlm_rerank.py lock --project <p> --budget <usd>             # once per batch
    python scripts/vlm_rerank.py run --set dev --expect-project <p> [--max-usd <cap>]
    python scripts/vlm_rerank.py status --set dev
    python scripts/vlm_rerank.py export --set dev
    python scripts/vlm_rerank.py report

topk writes data/private/vlm/topk/<set>.jsonl (resumable). The pilot's answers go to
data/private/vlm/pilot/<arm>.jsonl, its page to data/private/vlm/pilot.html (private:
Personal text, never published). Batch state: data/private/vlm/batch/<set>/ (one Vertex
job per 1,000 requests; a second round retries unusable answers), answers in
answers.jsonl there. export: dev becomes a stored silver ranking (vlm.LABEL, next to the
first stage's stored ranking, so silver_eval.py merges it and grade_unjudged.py can grade
its unjudged picks); cand and train become data/private/vlm/labels/<set>.jsonl.
report: results/vlm/<date>.md and .json (aggregates only, no task text).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config, silver, vlm  # noqa: E402
from tcr import judge as J  # noqa: E402
from tcr import vertex_batch as vb  # noqa: E402
from tcr.datasets import read_jsonl, write_jsonl  # noqa: E402

PILOT_DIR = vlm.WORK / "pilot"
BATCH_DIR = vlm.WORK / "batch"
LABELS_DIR = vlm.WORK / "labels"
OUT_DIR = config.RESULTS_DIR / "vlm"
CHUNK = 1000
MAX_ROUNDS = 2
MAX_MISSING = 3            # more numbers left out than this: the answer is retried
PER_TASK = 0.0060          # USD per request in batch; replaced by the pilot's measurement below
EXTRA = ("g1", "g2", "g3", "g4", "g5", "g6")
# The last VLM batch's lock (round 6c, 2026-11-25): the project it may run on and its cap
# (`lock` writes it once). `run` and `pilot` refuse unless .env's pin, --expect-project and
# this file name the same project, and they never spend past its cap.
LOCK_FILE = config.PRIVATE_DIR / "augment" / "cap-round6c.json"


def label_of(arm: str) -> str:
    return vlm.LABEL if arm == "img+desc" else vlm.LABEL.replace("VLM@", "VLM-img@")


def per_task_usd() -> float:
    """Batch USD per request: measured on the pilot (online, so halved) when it exists."""
    p = PILOT_DIR / "cost.json"
    return json.loads(p.read_text())["batch_per_task"] if p.exists() else PER_TASK


# --- gold ----------------------------------------------------------------------------

def golds_for(rows) -> dict:
    """{task_id: Gold}: eval gold with every existing extra-grade set, pool runs' gold."""
    extra = [n for n in EXTRA if (silver.EXTRA_GRADES / f"{n}.jsonl").exists()]
    out = {}
    dsets = {r["dataset"] for r in rows if r["set"] == "dev"}
    if dsets:
        g = silver.with_extra({d: silver.eval_gold(d) for d in dsets}, extra)
        for d in dsets:
            out.update(g[d])
    for run in {r["run"] for r in rows if r["set"] != "dev" and r["run"]}:
        out.update(silver.load_gold([run]))
    return out


# --- the account and budget lock ----------------------------------------------------

def cmd_lock(args) -> None:
    from tcr import gcp
    if LOCK_FILE.exists():
        sys.exit(f"{LOCK_FILE} exists: {LOCK_FILE.read_text().strip()}")
    pin = gcp.verify()
    if pin.project != args.project:
        sys.exit(f"refusing: .env pins project {pin.project}, not {args.project}")
    from datetime import datetime, timezone
    spent = J.spent()
    lock = {"project": pin.project, "project_number": pin.number, "spent_at_start": round(spent, 2),
            "budget": args.budget, "cap": round(spent + args.budget, 2),
            "set": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(json.dumps(lock, indent=1) + "\n")
    print(f"lock -> {LOCK_FILE}: {lock}")


def check_lock(args) -> float:
    """Stops unless the pin, --expect-project and the lock file agree; returns the cap
    (the smaller of --max-usd and the lock's) on the spend log's running total."""
    from tcr import gcp
    if not LOCK_FILE.exists():
        sys.exit(f"refusing: no {LOCK_FILE.name}; run `vlm_rerank.py lock` first")
    lock = json.loads(LOCK_FILE.read_text())
    pin = gcp.verify()
    if not (pin.project == args.expect_project == lock["project"]) or pin.number != lock["project_number"]:
        sys.exit(f"refusing: .env pins {pin.project} ({pin.number}), --expect-project is "
                 f"{args.expect_project}, the lock names {lock['project']} ({lock['project_number']})")
    if args.max_usd is not None and args.max_usd > lock["cap"]:
        sys.exit(f"refusing: --max-usd {args.max_usd} is above the lock's cap ${lock['cap']:.2f}")
    cap = lock["cap"] if args.max_usd is None else args.max_usd
    print(f"lock ok: project {pin.project}; spent ${J.spent():.2f} of cap ${cap:.2f}", flush=True)
    return cap


# --- pilot (online) ------------------------------------------------------------------

def pilot_rows() -> list:
    ids = vlm.pilot_ids()
    rows = []
    for name in ("dev", "train"):
        top = vlm.load_topk(name)
        missing = [i for i in ids[name] if i not in top]
        if missing:
            sys.exit(f"{len(missing)} pilot tasks of {name} have no top-K yet: run topk --pilot")
        rows += [top[i] for i in ids[name]]
    return rows


def ask_online(ctxs, row, arm: str) -> dict:
    from tcr.llm import complete_json
    system, parts, numbered = vlm.request(ctxs, row, arm)
    res = complete_json(parts[-1], images=parts[:-1], system=system, schema=vlm.SCHEMA,
                        backend="vertex", models=[vlm.MODEL], temperature=J.TEMPERATURE,
                        private=True, tag=f"vlm-pilot|{arm}", gen=J.GEN, retries=3)
    usage = res.get("usage", {})
    cost = 0.0 if res.get("cached") else J.log_spend(vlm.MODEL, usage, f"vlm-pilot:{arm}:{row['task_id']}")
    ans = vlm.parse(res["data"], numbered, row["top"])
    return {**{k: row[k] for k in ("set", "dataset", "run", "task_id")}, "arm": arm,
            "shown": [n for _, n in numbered], **ans, "usage": usage, "usd": round(cost, 6),
            "online_usd": round(J.usd(usage, vlm.MODEL), 6)}


def cmd_pilot(args) -> None:
    args.max_usd = check_lock(args)
    rows = pilot_rows()
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    ctxs = vlm.Contexts()
    for arm in args.arms.split(","):
        out = PILOT_DIR / f"{arm}.jsonl"
        done = {r["task_id"]: r for r in read_jsonl(out)} if out.exists() else {}
        todo = [r for r in rows if r["task_id"] not in done]
        if args.max_usd is not None and J.spent() + 0.02 * len(todo) > args.max_usd:
            sys.exit(f"refusing: spent ${J.spent():.2f} + ~${0.02 * len(todo):.2f} > --max-usd {args.max_usd}")
        print(f"{arm}: {len(todo)} requests", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(ask_online, ctxs, r, arm): r for r in todo}
            for f in as_completed(futs):
                try:
                    a = f.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  {futs[f]['task_id']}: {e}", flush=True)
                    continue
                done[a["task_id"]] = a
                write_jsonl(out, list(done.values()))
        print(f"{arm}: {len(done)} answers, online ${sum(a['online_usd'] for a in done.values()):.2f}",
              flush=True)
    summarize_pilot(args.arms.split(","))


def summarize_pilot(arms) -> dict:
    rows = {r["task_id"]: r for r in pilot_rows()}
    gold = golds_for(list(rows.values()))
    out = {"arms": {}}
    first = {t: silver.task_metrics(gold[t], r["top"][:10]) for t, r in rows.items() if t in gold}
    out["first"] = {s: silver.aggregate({t: m for t, m in first.items() if rows[t]["set"] == s})
                    for s in ("dev", "train")}
    costs = []
    for arm in arms:
        p = PILOT_DIR / f"{arm}.jsonl"
        if not p.exists():
            continue
        ans = {a["task_id"]: a for a in read_jsonl(p)}
        per = {t: silver.task_metrics(gold[t], a["order"][:10]) for t, a in ans.items() if t in gold}
        row = {s: silver.aggregate({t: m for t, m in per.items() if rows[t]["set"] == s}) for s in ("dev", "train")}
        dev = {t: m for t, m in per.items() if rows[t]["set"] == "dev"}
        row["vs_first_dev"] = silver.paired(dev, {t: first[t] for t in dev})
        row["missing"] = sum(a["missing"] for a in ans.values())
        row["top1_from_11_20"] = sum(rows[t]["top"].index(a["order"][0]) >= 10 for t, a in ans.items())
        row["no_good_icon"] = sum(a["no_good_icon"] for a in ans.values())
        u = [a["usage"] for a in ans.values() if a.get("usage")]
        row["tokens"] = {k: round(sum(x.get(k, 0) for x in u) / max(1, len(u))) for k in ("prompt", "output", "thoughts")}
        row["online_per_task"] = round(sum(a["online_usd"] for a in ans.values()) / max(1, len(ans)), 5)
        row["batch_per_task"] = round(sum(J.usd(a["usage"], vlm.MODEL, batch=True) for a in ans.values())
                                      / max(1, len(ans)), 5)
        costs.append(row["batch_per_task"])
        out["arms"][arm] = row
    if costs:
        PILOT_DIR.mkdir(parents=True, exist_ok=True)
        (PILOT_DIR / "cost.json").write_text(json.dumps({"batch_per_task": max(costs)}))
    (PILOT_DIR / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# --- pilot page (private) ---------------------------------------------------------------

def _img(name: str) -> str:
    b = (config.ICON_PNG_DIR / f"{name}.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def _cell(name: str, grade, mark: str = "") -> str:
    g = "" if grade is None else str(grade)
    cls = {3: "g3", 2: "g2", 1: "g1", 0: "g0"}.get(grade, "gn")
    return (f'<div class="c {cls}{mark}"><img src="{_img(name)}" alt=""><span>{html.escape(g)}</span></div>')


def cmd_pilot_report(args) -> None:
    rows = {r["task_id"]: r for r in pilot_rows()}
    gold = golds_for(list(rows.values()))
    arms = [a for a in vlm.ARMS if (PILOT_DIR / f"{a}.jsonl").exists()]
    ans = {a: {x["task_id"]: x for x in read_jsonl(PILOT_DIR / f"{a}.jsonl")} for a in arms}
    tasks = {}
    for run in {r["run"] for r in rows.values()}:
        tasks.update({t["id"]: t for t in silver.run_tasks(run)})
    summ = json.loads((PILOT_DIR / "summary.json").read_text()) if (PILOT_DIR / "summary.json").exists() else {}
    parts = ["<!doctype html><meta charset=utf-8><title>VLM rerank pilot</title><style>",
             "body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
             ".t{border-top:1px solid #ccc;padding:10px 0}.r{display:flex;flex-wrap:wrap;gap:2px;align-items:center}"
             ".lab{width:110px;font-size:12px;color:#555}.c{width:46px;text-align:center;border:2px solid transparent}"
             ".c img{width:40px;height:40px}.c span{display:block;font-size:11px}"
             ".g3{border-color:#1a7f37}.g2{border-color:#8cc68f}.g1{border-color:#e0c060}.g0,.gn{border-color:#eee}"
             ".u{outline:2px dashed #c33}.why{font-size:12px;color:#444;margin:2px 0 0 110px}"
             "pre{font-size:12px;background:#f6f6f6;padding:8px;overflow:auto}</style>",
             "<h1>VLM rerank pilot</h1><p>Each task: the big model's top 20 in its own order with the "
             "judge's grades (border: green 3, light green 2, yellow 1; dashed red = the judge never saw it), "
             "then each arm's order with the VLM's grades.</p>",
             f"<pre>{html.escape(json.dumps(summ, indent=1))}</pre>"]
    for tid, r in rows.items():
        g = gold.get(tid)
        t = tasks.get(tid, {})
        head = html.escape(t.get("title", tid))
        par = " > ".join(t.get("parents") or [])
        parts.append(f'<div class=t><b>{head}</b> <small>{html.escape(par)} · {r["set"]} · {r["dataset"]}</small>')

        def jg(n):
            return None if g is None or n not in g.seen else g.grades.get(silver.canonical(n), 0)

        def strip(label, names, grades=None):
            cells = "".join(_cell(n, (grades or {}).get(n) if grades is not None else jg(n),
                                  "" if g is None or silver.canonical(n) in g.seen else " u") for n in names)
            return f'<div class=r><div class=lab>{label}</div>{cells}</div>'
        parts.append(strip("first stage", r["top"]))
        for a in arms:
            x = ans[a].get(tid)
            if not x:
                continue
            parts.append(strip(f"{a}<br><small>{html.escape(x['subject'])}</small>", x["order"], x["grades"]))
            why = "; ".join(f"{i + 1}. {html.escape(x['why'].get(n, ''))}" for i, n in enumerate(x["order"][:3]))
            parts.append(f'<div class=why>{why}{" · NO GOOD ICON" if x["no_good_icon"] else ""}</div>')
        parts.append("</div>")
    parts += adversarial_pilot_html()
    out = vlm.WORK / "pilot.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"-> {out}")


def adversarial_pilot_html() -> list:
    """#2's pilot on the same page: per seed, the seed task, its confusion, and the tasks
    Flash wrote, with the big model's top-1 and the judge's ranking (augment_pools.py
    adversarial-pilot-judge)."""
    judged_p = config.PRIVATE_DIR / "augment" / "adv-pilot-judged.jsonl"
    if not judged_p.exists():
        return []
    judged = read_jsonl(judged_p)
    seeds = {}
    for p in (config.PRIVATE_DIR / "train_pools" / "personal_synth_pool.adv.jsonl",
              config.DATA_DIR / "train_pools" / "train_mslatte.adv.jsonl"):
        seeds.update({r["task_id"]: r for r in read_jsonl(p)} if p.exists() else {})
    titles = {}
    for run in ("pool-synth-j4", "pool-synth2-j4", "pool-mslatte-j4"):
        titles.update({t["id"]: t["title"] for t in silver.run_tasks(run)})
    n, f = len(judged), sum(r["fooled"] for r in judged)
    out = [f"<h1>Adversarial pilot</h1><p>{n} generated tasks judged online; the big model is fooled "
           f"(top-1 not graded ≥ 2) on {f} ({f / max(1, n):.0%}). Per task: the big model's top-1 "
           "(red border: fooled), then the judge's ranking with grades.</p>"]
    by_seed = {}
    for r in judged:
        by_seed.setdefault(r["meta"]["seed_task_id"], []).append(r)
    for sid, rs in by_seed.items():
        s = seeds.get(sid, {})
        out.append(f'<div class=t><b>seed:</b> {html.escape(titles.get(sid, sid))} '
                   f'<small>({html.escape(s.get("seed_kind", ""))})</small><br>'
                   f'<i>{html.escape(s.get("confusion", ""))}</i>'
                   f'<div class=r><div class=lab>seed: right / wrong</div>'
                   f'{_cell(s["seed_right"], None) if s else ""}{_cell(s["seed_wrong"], None, " u") if s else ""}</div>')
        for r in rs:
            m = r["meta"]
            cells = _cell(r["model_top"][0], None, " u" if r["fooled"] else "")
            cells += "".join(_cell(n, g) for n, g in r["ranking"][:6])
            out.append(f'<div class=r><div class=lab>{html.escape(r["title"])}<br><small>'
                       f'{html.escape(" > ".join(r["parents"]))}</small></div>{cells}</div>'
                       f'<div class=why>right: {html.escape(m.get("right_icon", ""))} · lure: '
                       f'{html.escape(m.get("lure", ""))}{" · NO GOOD ICON" if r["no_good_icon"] else ""}</div>')
        out.append("</div>")
    return out


# --- batch -----------------------------------------------------------------------------

class SetBatch:
    """The batch state of one set: batch/<set>/{meta.json, answers.jsonl, rounds/}."""

    def __init__(self, name: str, max_rounds: int = MAX_ROUNDS, chunk: int = CHUNK):
        if name not in ("dev", "train", "cand", "cand2"):
            sys.exit("sets: dev, train, cand, cand2")
        self.name = name
        # A round asks every task still without an answer; one with max_rounds attempts is
        # given up. Raising it (run --max-rounds) retries those, e.g. after a stalled round.
        self.max_rounds, self.chunk = max_rounds, chunk
        self.dir = BATCH_DIR / name
        self.answers_path = self.dir / "answers.jsonl"

    @property
    def meta(self) -> dict:
        p = self.dir / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save_meta(self, m: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps(m, indent=1))

    def answers(self) -> dict:
        return {r["task_id"]: r for r in read_jsonl(self.answers_path)} if self.answers_path.exists() else {}

    def run_obj(self, rname: str) -> vb.Run:
        return vb.Run(self.dir / "rounds", rname, f"vlm/{self.name}", vlm.MODEL)

    def pending(self, rows: dict, limit: int = 0) -> list:
        ans, att = self.answers(), self.meta.get("attempts", {})
        out = [r for t, r in rows.items() if t not in ans and att.get(t, 0) < self.max_rounds]
        return out[:limit] if limit else out

    def build_round(self, arm: str, limit: int = 0) -> list:
        rows = vlm.load_topk(self.name)
        if not rows:
            sys.exit(f"no top-K for {self.name}: run topk first")
        m = self.meta or {"set": self.name, "arm": arm, "round": 0, "attempts": {}, "runs": {}}
        if m["arm"] != arm:
            sys.exit(f"{self.name} was started with arm {m['arm']}")
        todo = self.pending(rows, limit)
        if not todo:
            return []
        k = m["round"] + 1
        ctxs = vlm.Contexts()
        names = []
        for j in range(0, len(todo), self.chunk):
            chunk = todo[j:j + self.chunk]
            rname = f"r{k}-c{j // self.chunk}"
            run = self.run_obj(rname)
            manifest = {}
            with open(run.input, "w") as f:
                for row in chunk:
                    system, parts, numbered = vlm.request(ctxs, row, arm)
                    req = vb.request_json(system, parts, vlm.SCHEMA, J.TEMPERATURE, J.GEN)
                    f.write(json.dumps({"key": row["task_id"], "request": req}, ensure_ascii=False) + "\n")
                    manifest[row["task_id"]] = {"numbered": numbered}
            run.save_manifest(manifest)
            est = round(len(chunk) * per_task_usd(), 2)
            m["runs"][rname] = {"round": k, "tasks": len(chunk), "estimate": est}
            names.append(rname)
            print(f"{self.name} {rname}: {len(chunk)} requests, estimate ${est:.2f}", flush=True)
        m["round"] = k
        self.save_meta(m)
        return names

    def collect(self, rname: str, job, again: bool = False) -> None:
        """Read one ended job's answers. A FAILED job (e.g. Vertex's 24 h limit) still writes
        the rows it finished, and those are billed, so its output is read too. `again`
        (recollect): re-read a round marked collected, keep only answers not stored yet,
        and leave the attempt counts alone (the first collect counted them)."""
        run = self.run_obj(rname)
        manifest = run.manifest()
        rows = vlm.load_topk(self.name)
        ans = self.answers()
        m = self.meta
        records = run.download(missing_ok=not vb.succeeded(job))
        got, bad, spent, seen = 0, 0, 0.0, set()
        for rec in records:
            tid = rec.get("key")
            if tid not in manifest or (again and tid in ans):
                continue
            seen.add(tid)
            resp = rec.get("response") or {}
            usage = vb.usage_of(resp) if resp else {}
            if usage:
                spent += J.log_spend(vlm.MODEL, usage, f"batch:vlm-{self.name}:{rname}", batch=True)
            try:
                data = json.loads(vb.response_text(resp))
            except Exception:  # noqa: BLE001
                data = None
            numbered = [tuple(x) for x in manifest[tid]["numbered"]]
            a = vlm.parse(data, numbered, rows[tid]["top"]) if data else None
            if a is None or a["missing"] > MAX_MISSING:
                if not again:
                    m["attempts"][tid] = m["attempts"].get(tid, 0) + 1
                bad += 1
                continue
            r = rows[tid]
            ans[tid] = {**{k: r[k] for k in ("set", "dataset", "run", "task_id")}, "arm": m["arm"],
                        "shown": [n for _, n in numbered], **a, "model": vlm.MODEL, "usage": usage}
            got += 1
        for tid in manifest:
            if tid not in seen and not (again and tid in ans):
                if not again:
                    m["attempts"][tid] = m["attempts"].get(tid, 0) + 1
                bad += 1
        write_jsonl(self.answers_path, list(ans.values()))
        m["runs"][rname]["collected"] = True
        self.save_meta(m)
        vb.log(f"{self.name} {rname}: {got} {'new ' if again else ''}answers, {bad} without one; "
               f"${spent:.2f}; total logged ${J.spent():.2f}")


def cmd_prepare(args) -> None:
    sb = SetBatch(args.set)
    if any(not r.get("collected") for r in sb.meta.get("runs", {}).values()):
        sys.exit(f"{args.set} has a round that is not collected yet: use run / status")
    sb.build_round(args.arm, args.limit)


def cmd_run(args) -> None:
    args.max_usd = check_lock(args)
    sb = SetBatch(args.set, args.max_rounds, args.chunk)
    while True:
        open_runs = [n for n, r in sb.meta.get("runs", {}).items() if not r.get("collected")]
        if not open_runs:
            if sb.meta.get("round", 0) >= sb.max_rounds:
                break
            open_runs = sb.build_round(args.arm)
            if not open_runs:
                break
        for n in open_runs:
            run = sb.run_obj(n)
            if not run.state().get("job"):
                est = sb.meta["runs"][n]["estimate"]
                if args.max_usd is not None and J.spent() + est > args.max_usd:
                    sys.exit(f"refusing: spent ${J.spent():.2f} + estimate ${est:.2f} > --max-usd {args.max_usd}")
                run.submit(f"vlm-{args.set}-{n}")
        for n in open_runs:
            sb.collect(n, sb.run_obj(n).wait(args.poll))
    n_ans = len(sb.answers())
    print(f"{args.set}: {n_ans} answers of {len(vlm.load_topk(args.set))} tasks; total logged ${J.spent():.2f}")


def cmd_recollect(args) -> None:
    """Re-read every ended round marked collected (the rounds collected before failed jobs'
    output was read: cand2's round 1). Idempotent: answers already stored are skipped, so
    their spend is not logged twice."""
    sb = SetBatch(args.set)
    for n, r in sb.meta.get("runs", {}).items():
        run = sb.run_obj(n)
        if not r.get("collected") or not run.state().get("job"):
            continue
        job = run.job()
        if vb.ended(job):
            sb.collect(n, job, again=True)
    print(f"{args.set}: {len(sb.answers())} answers of {len(vlm.load_topk(args.set))} tasks")


def cmd_status(args) -> None:
    sb = SetBatch(args.set)
    m = sb.meta
    print(f"{args.set}: arm {m.get('arm')}, round {m.get('round')}, {len(sb.answers())} answers")
    for n, r in m.get("runs", {}).items():
        run = sb.run_obj(n)
        st = vb.describe_job(run.job()) if run.state().get("job") else "not submitted"
        print(f"  {n}: {r['tasks']} requests, estimate ${r['estimate']}, "
              f"{'collected' if r.get('collected') else st}")


def cmd_export(args) -> None:
    sb = SetBatch(args.set)
    ans = sb.answers()
    if not ans:
        sys.exit("no answers yet")
    arm = sb.meta["arm"]
    if args.set == "dev":
        from grade_unjudged import stored
        folder = stored(vlm.FIRST_LABEL).parent
        out = folder / f"{label_of(arm)}.jsonl"
        rows = []
        for a in ans.values():
            sc = vlm.ranked_scores(a)
            rows.append({"task_id": a["task_id"], "dataset": a["dataset"], "top": a["order"][:10],
                         "scores": sc[:10], "signal": float(max(a["grades"].values() or [0])),
                         "no_good_icon": a["no_good_icon"]})
        write_jsonl(out, rows)
    else:
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        out = LABELS_DIR / f"{args.set}.jsonl"
        write_jsonl(out, [{k: a[k] for k in ("set", "dataset", "run", "task_id", "arm", "order", "grades",
                                            "no_good_icon", "subject", "model")} for a in ans.values()])
    print(f"{len(ans)} answers -> {out}")


# --- batch 2's budget -------------------------------------------------------------------

# Measured USD per request in batch (spend log): grade_unjudged g1-g4 $0.0047 per Argon task
# (×1.2: g5 has more new icons per task), Flash graders ~half; the Argon judge $0.0094 per
# request, ~2 per task (pool-synth3-j4); Flash generation, translations and rewrites.
RATE = {"grade-argon": 0.0056, "grade-flash": 0.0025, "judge": 0.0190, "adv-gen": 0.0012, "aug": 0.0025}
G6_TASKS = 0.3 * 3230        # dev tasks whose VLM top 3 holds an icon no grade set covers (guess)


def cmd_budget(args) -> None:
    """Batch 2's remaining spend at measured rates; refuses (exit 1) above --cap."""
    import augment_pools as A
    rows = []
    g5 = silver.EXTRA_GRADES / "g5.jsonl"
    meta = J.JUDGE_DIR / "grade" / "g5" / "meta.json"
    if not g5.exists() and meta.exists():
        m = json.loads(meta.read_text())
        usd = sum(r["tasks"] * RATE["grade-argon" if "argon" in r["model"] else "grade-flash"]
                  for r in m["runs"].values())
        rows.append(("g5: leaders' unjudged top-10 on dev", m["tasks"], usd))
    for name in ("dev", "train", "cand"):
        sb = SetBatch(name)
        top = vlm.load_topk(name)
        n = len(top) - len(sb.answers()) if top else {"dev": 3230, "train": vlm.TRAIN_N, "cand": vlm.CAND_N}[name]
        rows.append((f"#5 VLM rerank, {name}", n, n * per_task_usd()))
    seeds = sum(len(A.adv_seeds(r)[0]) for r in A.ROUND2_POOL_RUNS)
    gen = 0
    for ds, private in (("personal_synth_pool", True), ("train_mslatte", False)):
        p = (config.PRIVATE_DIR if private else config.DATA_DIR) / "train_pools" / f"{ds}.adv.jsonl"
        gen += len(read_jsonl(p)) if p.exists() else 0
    rows.append(("#2 Flash generation", seeds - gen, (seeds - gen) * RATE["adv-gen"]))
    n_adv = round(seeds * A.ADV_PER_SEED * 0.92)
    judged = sum(len(read_jsonl(silver.label_path(r))) for r in A.ADV_RUNS.values() if silver.label_path(r).exists())
    rows.append(("#2 Argon judging", n_adv - judged, max(0, n_adv - judged) * RATE["judge"]))
    rows.append(("#2 translations, rewrites", n_adv, n_adv * RATE["aug"]))
    if not (silver.EXTRA_GRADES / "g6.jsonl").exists():
        rows.append(("g6: the VLM's unjudged top-3 on dev", round(G6_TASKS), G6_TASKS * RATE["grade-argon"]))
    total, spent = sum(r[2] for r in rows), J.spent()
    for what, n, usd in rows:
        print(f"  {what:42s} {n:6d}  ${usd:6.2f}")
    print(f"  {'total':42s} {'':6s}  ${total:6.2f}; spent ${spent:.2f} -> ${spent + total:.2f} "
          f"(cap ${args.cap:.2f}, margin ${args.cap - spent - total:.2f})")
    if spent + total > args.cap:
        sys.exit("refusing: batch 2's estimate crosses the cap (shrink ADV_SEEDS or the VLM sets)")


# --- report -----------------------------------------------------------------------------

def _spearman(x, y) -> float:
    import numpy as np
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_report(args) -> None:
    out = {"date": date.today().isoformat(), "first_stage": vlm.FIRST_LABEL, "sets": {}}
    for name in ("dev", "train", "cand", "cand2"):
        sb = SetBatch(name)
        ans = sb.answers()
        if not ans:
            continue
        rows = vlm.load_topk(name)
        rec = {"n": len(ans), "arm": sb.meta.get("arm"),
               "missing_numbers": sum(a["missing"] for a in ans.values()),
               "no_good_icon": round(sum(a["no_good_icon"] for a in ans.values()) / len(ans), 4),
               "top1_from_11_20": round(sum(rows[t]["top"].index(a["order"][0]) >= 10
                                            for t, a in ans.items()) / len(ans), 4),
               "top1_same_as_first": round(sum(rows[t]["top"][0] == a["order"][0]
                                               for t, a in ans.items()) / len(ans), 4),
               # position bias: Spearman of the VLM's rank with the shown position, mean per task
               "rank_vs_shown_position": round(sum(_spearman([a["order"].index(n) for n in a["shown"]],
                                                             list(range(len(a["shown"]))))
                                                   for a in ans.values()) / len(ans), 4),
               "rank_vs_first_stage": round(sum(_spearman([a["order"].index(n) for n in rows[t]["top"]],
                                                          list(range(len(rows[t]["top"]))))
                                                for t, a in ans.items()) / len(ans), 4)}
        u = [a["usage"] for a in ans.values() if a.get("usage")]
        rec["batch_usd_per_task"] = round(sum(J.usd(x, vlm.MODEL, batch=True) for x in u) / max(1, len(u)), 5)
        if name != "cand":
            gold = golds_for(list(rows.values()))
            first = {t: silver.task_metrics(gold[t], rows[t]["top"][:10]) for t in ans if t in gold}
            per = {t: silver.task_metrics(gold[t], ans[t]["order"][:10]) for t in ans if t in gold}
            by_d = sorted({rows[t]["dataset"] for t in per})
            rec["vlm"] = {d: silver.aggregate({t: m for t, m in per.items() if rows[t]["dataset"] == d}) for d in by_d}
            rec["first"] = {d: silver.aggregate({t: m for t, m in first.items() if rows[t]["dataset"] == d}) for d in by_d}
            rec["vlm"]["pooled"], rec["first"]["pooled"] = silver.aggregate(per), silver.aggregate(first)
            rec["vs_first"] = {d: silver.paired({t: m for t, m in per.items() if rows[t]["dataset"] == d},
                                                {t: m for t, m in first.items() if rows[t]["dataset"] == d})
                               for d in by_d}
            rec["vs_first"]["pooled"] = silver.paired(per, first)
            # the judge's top-1 against the VLM's (agreement on the same lists)
            agree = [silver.canonical(ans[t]["order"][0]) == gold[t].order[0] for t in per if gold[t].order]
            rec["top1_equals_judge_top1"] = round(sum(agree) / max(1, len(agree)), 4)
        out["sets"][name] = rec
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / out["date"]
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1))
    lines = [f"# VLM reranker (round 6 #5), {out['date']}", "",
             f"Argon reranks the top 20 of {vlm.FIRST_LABEL} (glyph images, shuffled order, no names). "
             "Gold: the judge's labels + extra grades " + ", ".join(n for n in EXTRA if (silver.EXTRA_GRADES / f"{n}.jsonl").exists())
             + ". Aggregates only.", ""]
    for name, rec in out["sets"].items():
        lines += [f"## {name} ({rec['n']} tasks, arm {rec['arm']})", ""]
        if "vlm" in rec:
            lines += ["| dataset | first nDCG@5 | VLM nDCG@5 | Δ [95%] | first hit@1 | VLM hit@1 | VLM hit@1 ub | McNemar p |",
                      "|---|---:|---:|---|---:|---:|---:|---:|"]
            for d in rec["vlm"]:
                v, f, c = rec["vlm"][d], rec["first"][d], rec["vs_first"].get(d, {})
                lines.append(f"| {d} | {f['ndcg5']:.3f} | {v['ndcg5']:.3f} | {c.get('d_ndcg5', 0):+.3f} "
                             f"[{c.get('d_ndcg5_ci', [0, 0])[0]:+.3f}, {c.get('d_ndcg5_ci', [0, 0])[1]:+.3f}] | "
                             f"{f['hit1']:.3f} | {v['hit1']:.3f} | {v['hit1_ub']:.3f} | {c.get('mcnemar_p', 1):.3g} |")
            lines.append("")
        for k in ("top1_equals_judge_top1", "top1_same_as_first", "top1_from_11_20", "no_good_icon",
                  "rank_vs_first_stage", "rank_vs_shown_position", "missing_numbers", "batch_usd_per_task"):
            if k in rec:
                lines.append(f"- {k}: {rec[k]}")
        lines.append("")
    stem.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("topk")
    p.add_argument("--sets", default="dev,val,train,cand")
    p.add_argument("--pilot", action="store_true", help="only the pilot's tasks")
    p = sub.add_parser("pilot")
    p.add_argument("--arms", default="img+desc,img")
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--expect-project", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--project", required=True)
    p.add_argument("--budget", type=float, required=True)
    sub.add_parser("pilot-report")
    sub.add_parser("pilot-summary")
    for name in ("prepare", "run", "status", "recollect", "export"):
        q = sub.add_parser(name)
        q.add_argument("--set", required=True)
        if name in ("prepare", "run"):
            q.add_argument("--arm", default="img+desc", choices=vlm.ARMS)
        if name == "prepare":
            q.add_argument("--limit", type=int, default=0)
        if name == "run":
            q.add_argument("--max-usd", type=float, default=None)
            q.add_argument("--poll", type=int, default=120)
            q.add_argument("--expect-project", required=True)
            q.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                           help="rounds before a task is given up (raise it to retry them)")
            q.add_argument("--chunk", type=int, default=CHUNK, help="requests per batch job")
    sub.add_parser("report")
    p = sub.add_parser("budget")
    p.add_argument("--cap", type=float, required=True)
    args = ap.parse_args()
    {"topk": cmd_topk, "pilot": cmd_pilot, "pilot-report": cmd_pilot_report,
     "pilot-summary": lambda a: summarize_pilot(list(vlm.ARMS)),
     "lock": cmd_lock, "prepare": cmd_prepare, "run": cmd_run, "status": cmd_status, "recollect": cmd_recollect, "export": cmd_export,
     "report": cmd_report, "budget": cmd_budget}[args.cmd](args)


def cmd_topk(args) -> None:
    vlm.threads()
    only = None
    if args.pilot:
        ids = vlm.pilot_ids()
        only = set(ids["dev"]) | set(ids["train"])
    for name in args.sets.split(","):
        vlm.score_set(name, only=only, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
