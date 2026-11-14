#!/usr/bin/env python3
"""The last Gemini jobs on the judged training pools (docs/next-label-free-work.md step 0).

- **Translations:** every task of a judge run (scripts/judge_batch.py) into Spanish and
  Russian. The judge's labels carry over, since the icon does not depend on the language.
- **Parent headings** (`--gen-parents`): the Personal-style pool has none, while Personal
  has one on every task. Gemini invents the org outline headings each task would sit
  under, grouping the related tasks of a request under shared ones.

Each round is one Vertex batch job (Gemini 3.8 Flash) over the run's tasks that are not
done yet; a task that fails three rounds is left out. Results are one line per task,
merged across the judge runs of a dataset:

  private datasets  data/private/train_pools/<dataset>.aug.jsonl
  public datasets   data/train_pools/<dataset>.aug.jsonl            (commit)

  {"task_id", "parents": [EN headings, outermost first], "parents_generated": bool,
   "es": {"title", "parents", "body"}, "ru": {...},
   "contaminated": [langs whose title is too close to an eval title], "model", "run"}

Commands:
  run         rounds until every task of --judge-run is done (resumable; refuses a
              round whose estimate would take the spend log past --max-usd)
  status      progress of a judge run's augmentation
  backfill    add the first pool generation's Vertex cost (2026-11-13, not logged then)
              to the spend log; runs once
  cap         the spend-log total not to exceed, from the credit the billing console shows
              (fixed at the first call, in data/private/augment/cap.json)
  plan-extra  how many more Personal-style tasks the budget left under --cap can buy
  extra       the pool tasks not judged yet (after `synth_personal.py pool --n …`),
              near-duplicates of judged ones dropped                   -> --out FILE

The whole sequence, with the pool extension: scripts/gemini_last_jobs.sh.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import judge as J  # noqa: E402
from tcr import vertex_batch as vb  # noqa: E402
from tcr.datasets import read_jsonl, write_jsonl  # noqa: E402

MODEL = "gemini-3.8-flash"
GEN = {"thinking_level": "low"}
TEMPERATURE = 0.4
WORK = config.PRIVATE_DIR / "augment"       # private: requests carry pool text
CAP_FILE = WORK / "cap.json"
JUDGE_RUNS = J.JUDGE_DIR / "batch"
POOL = config.PRIVATE_DIR / "train_pools" / "personal_synth_pool.jsonl"
MAX_TASKS, MAX_CHARS = 15, 4000             # per request: output stays well under 8k tokens
MAX_ATTEMPTS = 3
TRIAL_USD = 300.0
DUP_SIM = 0.95                              # as synth_personal.OS_DUP_SIM
# Per-task costs for plan-extra, measured: Argon judge $0.0190 in batch (pool-synth-j4;
# +10% so a later round is never refused), the first pool generation $10.32 for 4,986 kept
# tasks (×1.5: the top-up round overgenerates 2×).
JUDGE_PER_TASK = 0.0210
GEN_PER_TASK = 0.0031
AUG_PER_TASK_DEFAULT = 0.0010

SYSTEM = """You translate personal to-do items from one person's org-mode task list into
Spanish and Russian, the way a native speaker writes them in their own task list.

- Natural, concise to-do style. Spanish: the infinitive or a noun phrase, as natives write
  lists ("Revisar el informe", "Cita con el dentista"). Russian: the infinitive or a noun
  phrase ("Проверить отчёт", "Запись к стоматологу").
- Translate the meaning, not word by word. Keep the level of detail: add nothing, drop
  nothing.
- Keep unchanged: names of products, tools, libraries, models, datasets, papers, people
  and places; code, identifiers, file names, URLs, numbers and dates; org markup
  (=verbatim=, ~code~, *bold*, list markers, line breaks). In a link [[target][label]]
  translate only the label.
- Widely used English technical terms may stay in English where a native speaker in the
  field would keep them.
- Return one entry per task, with the task's number n, in the same order."""

PARENTS_GIVEN = """The tasks carry their outline headings (PARENTS, outermost first, or
"(none)"). Return them unchanged in "parents" ([] for none) and translate them into the
"parents" of "es" and "ru"."""

PARENTS_GEN = """These tasks have lost their outline headings. For each task, first write
the org-mode headings it sits under in this person's outline (outermost first) in
"parents", in English, then translate them into the "parents" of "es" and "ru".

- The tasks of one request come from one area of life or work. Group related tasks under
  shared headings, as a real outline does: several tasks usually share one.
- Depth: one heading for about half of the tasks, two for most others, rarely three.
  Outer headings name an area or a long-running project, inner ones a specific project or
  topic. One to five words each; no TODO keywords, tags, dates or numbering.
- Generic examples of the form: "Health > Knee rehab", "Home > Kitchen renovation",
  "Research > Survival analysis reading", "Job search", "Travel > Lisbon trip".
- Never use names of private people, employers or addresses."""

_STR_LIST = {"type": "array", "items": {"type": "string"}}
_LANG = {"type": "object", "properties": {"title": {"type": "string"}, "parents": _STR_LIST,
                                          "body": {"type": "string"}},
         "required": ["title", "parents", "body"]}
SCHEMA = {"type": "object", "properties": {"tasks": {"type": "array", "items": {
    "type": "object", "properties": {"n": {"type": "integer"}, "parents": _STR_LIST,
                                     "es": _LANG, "ru": _LANG},
    "required": ["n", "parents", "es", "ru"]}}}, "required": ["tasks"]}

_CYR = re.compile("[а-яА-ЯёЁ]")
_WORD = re.compile("[A-Za-z]{3,}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- the judge run's tasks and the results -------------------------------------------

def judge_meta(run: str) -> dict:
    p = JUDGE_RUNS / run / "meta.json"
    if not p.exists():
        sys.exit(f"no judge run {run!r} ({p})")
    return json.loads(p.read_text())


def out_path(meta: dict) -> Path:
    base = (config.PRIVATE_DIR / "train_pools") if meta["private"] else (config.DATA_DIR / "train_pools")
    return base / f"{meta['dataset']}.aug.jsonl"


def load_results(path: Path) -> dict:
    return {r["task_id"]: r for r in read_jsonl(path)} if path.exists() else {}


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, [results[k] for k in sorted(results)])


class Aug:
    """The augmentation state of one judge run: data/private/augment/<run>/."""

    def __init__(self, run: str, gen_parents: bool):
        self.run, self.gen_parents = run, gen_parents
        self.jmeta = judge_meta(run)
        self.dir = WORK / run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.out = out_path(self.jmeta)

    @property
    def meta(self) -> dict:
        p = self.dir / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {"round": 0, "attempts": {}}

    def save_meta(self, m: dict) -> None:
        (self.dir / "meta.json").write_text(json.dumps(m, indent=1))

    def tasks(self) -> list:
        return read_jsonl(Path(self.jmeta["source"]))

    def job(self, k: int) -> vb.Run:
        return vb.Run(self.dir / "rounds", f"r{k}", f"augment/{self.run}", MODEL)

    def pending(self) -> list:
        done, att = set(load_results(self.out)), self.meta["attempts"]
        return [t for t in self.tasks() if t["id"] not in done and att.get(t["id"], 0) < MAX_ATTEMPTS]

    # --- requests ---
    def chunks(self, tasks: list) -> list:
        """Requests of at most MAX_TASKS tasks / MAX_CHARS characters; with generated
        parents, each request holds one topic cluster, so headings can be shared."""
        if self.gen_parents:
            tasks = sorted(tasks, key=lambda t: ((t.get("meta") or {}).get("cluster", -1), t["id"]))
        out, cur, chars, cl = [], [], 0, None
        for t in tasks:
            c = (t.get("meta") or {}).get("cluster") if self.gen_parents else None
            n = len(t["title"]) + len(t.get("body") or "")
            if cur and (len(cur) >= MAX_TASKS or chars + n > MAX_CHARS or c != cl):
                out.append(cur)
                cur, chars = [], 0
            cur.append(t)
            chars += n
            cl = c
        return out + ([cur] if cur else [])

    def prompt(self, chunk: list) -> str:
        lines = [PARENTS_GEN if self.gen_parents else PARENTS_GIVEN, ""]
        for i, t in enumerate(chunk, 1):
            lines.append(f"TASK {i}")
            lines.append(f"TITLE: {t['title']}")
            if not self.gen_parents:
                lines.append("PARENTS: " + (" > ".join(t.get("parents") or []) or "(none)"))
            body = (t.get("body") or "").strip()
            lines.append("BODY:" + (f"\n{body}" if body else " (empty)"))
            lines.append("")
        return "\n".join(lines)

    def estimate(self, chunks: list) -> float:
        p = J.PRICES[MODEL]
        tin = tout = 0
        for ch in chunks:
            chars = sum(len(t["title"]) + len(t.get("body") or "") + 30 for t in ch)
            tin += 900 + chars / 3.5
            tout += 2 * chars / 2.5 + 40 * len(ch) + 400      # two languages + JSON + thinking
        return (tin * p[2] + tout * p[3]) / 1e6

    # --- one round ---
    def build_round(self, pending: list) -> tuple:
        m = self.meta
        k = m["round"] + 1
        chunks = self.chunks(pending)
        run = self.job(k)
        manifest = {}
        with open(run.input, "w") as f:
            for j, ch in enumerate(chunks):
                key = f"r{k}|{j}"
                manifest[key] = [t["id"] for t in ch]
                req = vb.request_json(SYSTEM, [self.prompt(ch)], SCHEMA, TEMPERATURE, GEN)
                f.write(json.dumps({"key": key, "request": req}, ensure_ascii=False) + "\n")
        run.save_manifest(manifest)
        m.update(round=k, job_round=k)
        self.save_meta(m)
        return k, chunks

    def collect(self, k: int, job) -> None:
        run = self.job(k)
        manifest = run.manifest()
        by_id = {t["id"]: t for t in self.tasks()}
        results = load_results(self.out)
        m = self.meta
        got, bad, spent = 0, 0, 0.0
        seen = set()
        records = run.download() if vb.succeeded(job) or "PARTIALLY" in str(job.state) else []
        for rec in records:
            ids = manifest.get(rec.get("key"))
            if not ids:
                continue
            seen.add(rec["key"])
            resp = rec.get("response") or {}
            usage = vb.usage_of(resp) if resp else {}
            if usage:
                spent += J.log_spend(MODEL, usage, f"batch:augment-{self.run}:r{k}", batch=True)
            try:
                items = {it["n"]: it for it in json.loads(vb.response_text(resp))["tasks"]}
            except Exception:  # noqa: BLE001 — every task of the request is retried
                items = {}
            for i, tid in enumerate(ids, 1):
                rec_out = self.check(by_id[tid], items.get(i))
                if rec_out is None:
                    m["attempts"][tid] = m["attempts"].get(tid, 0) + 1
                    bad += 1
                else:
                    results[tid] = rec_out
                    got += 1
        for key, ids in manifest.items():          # requests without any output record
            if key not in seen:
                for tid in ids:
                    m["attempts"][tid] = m["attempts"].get(tid, 0) + 1
                    bad += 1
        save_results(self.out, results)
        m["job_round"] = None
        self.save_meta(m)
        st = run.state()
        st["collected"] = now()
        run.save_state(st)
        vb.log(f"round {k}: {got} tasks done, {bad} to retry; ${spent:.2f}; "
               f"total logged ${J.spent():.2f}")

    def check(self, task: dict, it) -> dict | None:
        """The result line for a task, or None when the answer is unusable."""
        if not it:
            return None
        given = list(task.get("parents") or [])
        parents = [p.strip() for p in it.get("parents") or [] if p.strip()]
        if self.gen_parents and not given:
            if not 1 <= len(parents) <= 3 or any(len(p.split()) > 8 for p in parents):
                return None
        else:
            parents = given
        out = {"task_id": task["id"], "parents": parents,
               "parents_generated": bool(self.gen_parents and not given)}
        body = (task.get("body") or "").strip()
        for lang in ("es", "ru"):
            tr = it.get(lang) or {}
            title = (tr.get("title") or "").strip()
            tb = (tr.get("body") or "").strip() if body else ""
            tp = [p.strip() for p in tr.get("parents") or []]
            if not title or (body and not tb) or len(tp) != len(parents) or not all(tp):
                return None
            if lang == "ru" and len(_WORD.findall(task["title"])) >= 2 and not _CYR.search(title):
                return None
            out[lang] = {"title": title, "parents": tp, "body": tb}
        out.update(model=MODEL, run=self.run)
        return out

    def finalize(self) -> None:
        """Mark translated titles too close to an eval title (training-text rule)."""
        from tcr.describe_v2 import contamination_hits
        results = load_results(self.out)
        todo = [r for r in results.values() if "contaminated" not in r]
        if not todo:
            return
        flags = {r["task_id"]: [] for r in todo}
        for lang in ("es", "ru"):
            exact, near = contamination_hits([r[lang]["title"] for r in todo])
            for i in exact | near:
                flags[todo[i]["task_id"]].append(lang)
        for r in todo:
            r["contaminated"] = flags[r["task_id"]]
        save_results(self.out, results)
        n = sum(bool(v) for v in flags.values())
        vb.log(f"contamination check: {n} of {len(todo)} tasks have a flagged translation")


# --- commands ------------------------------------------------------------------------

def cmd_run(args) -> None:
    a = Aug(args.judge_run, args.gen_parents)
    if not a.jmeta.get("finished"):
        sys.exit(f"judge run {args.judge_run} is not finished")
    while True:
        k = a.meta.get("job_round")
        if k:
            run = a.job(k)
            if not run.state().get("job"):
                by_id = {t["id"]: t for t in a.tasks()}
                est = a.estimate([[by_id[i] for i in ids] for ids in run.manifest().values()])
                if args.max_usd is not None and J.spent() + est > args.max_usd:
                    sys.exit(f"refusing: spent ${J.spent():.2f} + estimate ${est:.2f} "
                             f"> --max-usd {args.max_usd}")
                run.submit(f"augment-{args.judge_run}-r{k}")
            job = run.wait(args.poll)
            a.collect(k, job)
            continue
        pending = a.pending()
        if not pending:
            a.finalize()
            done = len(load_results(a.out))
            vb.log(f"{args.judge_run}: finished; {a.out} holds {done} tasks")
            return
        k, chunks = a.build_round(pending)
        vb.log(f"round {k}: {len(pending)} tasks in {len(chunks)} requests; "
               f"estimate ${a.estimate(chunks):.2f}")


def cmd_status(args) -> None:
    a = Aug(args.judge_run, False)
    ids = {t["id"] for t in a.tasks()}
    done = ids & set(load_results(a.out))
    m = a.meta
    failed = sum(1 for t, n in m["attempts"].items() if n >= MAX_ATTEMPTS and t not in done)
    print(f"{args.judge_run}: {len(done)}/{len(ids)} done, {failed} given up; round {m['round']}")
    if m.get("job_round") and a.job(m["job_round"]).state().get("job"):
        print("  " + vb.describe_job(a.job(m["job_round"]).job()))


BACKFILL_WHAT = "backfill:pool-gen-2026-11-13"


def cmd_backfill(args) -> None:
    log = J.SPEND_LOG.read_text() if J.SPEND_LOG.exists() else ""
    if BACKFILL_WHAT in log:
        print("backfill: already in the spend log")
        return
    usage, n = {"prompt": 0, "output": 0, "thoughts": 0}, 0
    cutoff = datetime(2026, 11, 14, tzinfo=timezone.utc).timestamp()
    for f in glob.glob(str(config.ROOT / ".cache" / "llm" / "*.json")):
        if Path(f).stat().st_mtime >= cutoff:
            continue
        try:
            d = json.loads(Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        if d.get("model") == MODEL and isinstance(d.get("data"), dict) and "tasks" in d["data"] \
                and d.get("usage") and "n" in (d["data"]["tasks"] or [{}])[0]:
            for k in usage:
                usage[k] += d["usage"].get(k, 0)
            n += 1
    usd = J.log_spend(MODEL, usage, BACKFILL_WHAT, batch=False)
    print(f"backfill: {n} pool-generation calls, ${usd:.2f} added to the spend log")


def cmd_cap(args) -> None:
    if CAP_FILE.exists():
        print(json.loads(CAP_FILE.read_text())["cap"])
        return
    s = J.spent()
    cap = round(min(s + args.credit_left, TRIAL_USD) - args.reserve, 2)
    WORK.mkdir(parents=True, exist_ok=True)
    CAP_FILE.write_text(json.dumps({"cap": cap, "spent_at_start": round(s, 2),
                                    "credit_left": args.credit_left, "reserve": args.reserve,
                                    "set": now()}, indent=1))
    print(f"cap: spend log ${s:.2f}, credit left ${args.credit_left:.2f}, reserve "
          f"${args.reserve:.2f} -> stop at ${cap:.2f} in the log", file=sys.stderr)
    print(cap)


def aug_per_task() -> float:
    """Measured augmentation cost per task (the runs so far), else a default."""
    usd = sum(json.loads(l)["usd"] for l in J.SPEND_LOG.read_text().splitlines()
              if l.strip() and "augment-" in json.loads(l)["what"]) if J.SPEND_LOG.exists() else 0
    n = sum(len(read_jsonl(p)) for p in [config.PRIVATE_DIR / "train_pools" / "personal_synth_pool.aug.jsonl",
                                         config.DATA_DIR / "train_pools" / "train_mslatte.aug.jsonl"]
            if p.exists())
    return usd / n * 1.3 if n >= 500 and usd else AUG_PER_TASK_DEFAULT


def cmd_plan_extra(args) -> None:
    left = args.cap - J.spent()
    per = JUDGE_PER_TASK + aug_per_task() + (0 if args.generated else GEN_PER_TASK)
    k = max(0, math.floor(left / per))
    print(f"plan-extra: ${left:.2f} left under the cap, ${per:.4f} per task -> {k} tasks",
          file=sys.stderr)
    print(k)


def cmd_extra(args) -> None:
    import numpy as np
    from tcr.index import embed_texts
    from tcr.org_tasks import query_text
    judged = {}
    for mp in JUDGE_RUNS.glob("*/meta.json"):
        m = json.loads(mp.read_text())
        if m["dataset"] == "personal_synth_pool":
            judged.update({t["id"]: t for t in read_jsonl(Path(m["source"]))})
    cand = [r for r in read_jsonl(POOL) if r["id"] not in judged]
    if not cand:
        sys.exit("extra: no new pool tasks; run synth_personal.py pool --n … first")

    def emb(rs):
        return embed_texts(config.MULTILINGUAL_MODEL,
                           [query_text(r["title"], r.get("body", ""), with_body=True) for r in rs],
                           is_query=True)
    E_old, E_new = emb(list(judged.values())), emb(cand)
    best = np.concatenate([(E_new[s:s + 2048] @ E_old.T).max(axis=1)
                           for s in range(0, len(cand), 2048)])
    kept = [r for r, b in zip(cand, best) if b <= DUP_SIM]
    write_jsonl(Path(args.out), kept)
    print(f"extra: {len(cand)} new pool tasks, {len(cand) - len(kept)} near-duplicates of "
          f"judged ones dropped -> {len(kept)} in {args.out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--judge-run", required=True)
    p.add_argument("--gen-parents", action="store_true",
                   help="invent parent headings for tasks that have none")
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--poll", type=int, default=120)
    p = sub.add_parser("status")
    p.add_argument("--judge-run", required=True)
    sub.add_parser("backfill")
    p = sub.add_parser("cap")
    p.add_argument("--credit-left", type=float, required=True)
    p.add_argument("--reserve", type=float, default=8.0)
    p = sub.add_parser("plan-extra")
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--generated", action="store_true", help="the new tasks exist already")
    p = sub.add_parser("extra")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
