#!/usr/bin/env python3
"""Gemini jobs on the judged task sets: step 0 (docs/next-label-free-work.md) and round 3.

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
  rewrite     four rewrites (terse, reworded, casual ES, casual RU) of every judged
              training task of --judge-run              -> <dataset>.rw.jsonl
  translate   titles and parents into --langs: of a pool run (training, <dataset>.xl.jsonl)
              or, with --eval, of an eval set's dev tasks (data/datasets/xling/)
  xling-build the translated eval set data/datasets/public_xling.jsonl (eval only)

Round 3 (docs/next-data-and-distillation.md, hand-off docs/vertex-round3-run.md):
scripts/vertex_round3_jobs.sh.

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
XLING_DEFAULT = ("de", "fr", "pt", "it", "zh", "ja", "pl")   # = tcr.silver.XLING_LANGS
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
    """The augmentation state of one judge run: data/private/augment/<run>/ (subclasses:
    <run>.<kind>/). This class translates into ES/RU; Rewrite and Translate below reuse its
    rounds, batch jobs, retries and budget checks with their own request and checks."""
    kind = "aug"
    system, schema = SYSTEM, SCHEMA

    def __init__(self, run: str, gen_parents: bool = False):
        self.run, self.gen_parents = run, gen_parents
        self.jmeta = judge_meta(run)
        self.dir = WORK / (run if self.kind == "aug" else f"{run}.{self.kind}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.out = self.out_file()

    def out_file(self) -> Path:
        return out_path(self.jmeta)

    @property
    def meta(self) -> dict:
        p = self.dir / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {"round": 0, "attempts": {}}

    def save_meta(self, m: dict) -> None:
        (self.dir / "meta.json").write_text(json.dumps(m, indent=1))

    def tasks(self) -> list:
        return read_jsonl(Path(self.jmeta["source"]))

    def job(self, k: int) -> vb.Run:
        folder = f"augment/{self.run}" + ("" if self.kind == "aug" else f".{self.kind}")
        return vb.Run(self.dir / "rounds", f"r{k}", folder, MODEL)

    @property
    def label(self) -> str:
        return self.run if self.kind == "aug" else f"{self.run}-{self.kind}"

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
                req = vb.request_json(self.system, [self.prompt(ch)], self.schema, TEMPERATURE, GEN)
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
                spent += J.log_spend(MODEL, usage, f"batch:augment-{self.label}:r{k}", batch=True)
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


# --- register rewrites (round 3, docs/next-data-and-distillation.md) -----------------

SYSTEM_RW = """You rewrite personal to-do items from one person's org-mode task list the way
the same person might have written them on another day. Each rewrite must point to the
same icon: a glanceable reminder of what the task is about.

Keep, in every rewrite: the central subject of the task (what it is about), its context
or domain, and the action when the action is the point (call, pay, fix, read, sign). Keep
names of products, tools, people and places, numbers and dates as they are. Add nothing
(no new objects, people, places, reasons or details) and drop nothing that changes what
the task is about. The parent headings are context only: do not repeat them in the title.

For each task write four rewrites of the TITLE:
- "terse": how the person jots it in a hurry: fewer words, articles and filler dropped,
  common abbreviations allowed (appt, w/, re:, mtg, info), lowercase allowed.
- "reworded": the same meaning in different words and word order (another verb or
  construction, synonyms), about the same length; natural, never awkward.
- "es": a short, natural Spanish to-do in a casual register (infinitive or noun phrase),
  not a word-for-word translation.
- "ru": a short, natural Russian to-do in a casual register (infinitive or noun phrase),
  not a word-for-word translation.
Return one entry per task, with the task's number n, in the same order."""

SCHEMA_RW = {"type": "object", "properties": {"tasks": {"type": "array", "items": {
    "type": "object", "properties": {"n": {"type": "integer"}, "terse": {"type": "string"},
                                     "reworded": {"type": "string"}, "es": {"type": "string"},
                                     "ru": {"type": "string"}},
    "required": ["n", "terse", "reworded", "es", "ru"]}}}, "required": ["tasks"]}
RW_STYLES = (("terse", "en"), ("reworded", "en"), ("es", "es"), ("ru", "ru"))


def _rw_path(meta: dict) -> Path:
    base = (config.PRIVATE_DIR / "train_pools") if meta["private"] else (config.DATA_DIR / "train_pools")
    return base / f"{meta['dataset']}.rw.jsonl"


class Rewrite(Aug):
    """Four rewrites of every judged training task (not the validation tasks) of a pool run:
    <dataset>.rw.jsonl, {"task_id", "variants": [{"lang", "style", "title"}],
    "contaminated": [variant indices too close to an eval title], "model", "run"}.
    tcr.finetune turns them into para-* rows that keep the original's grades."""
    kind, system, schema = "rw", SYSTEM_RW, SCHEMA_RW

    def out_file(self) -> Path:
        return _rw_path(self.jmeta)

    def tasks(self) -> list:
        from tcr.finetune import is_val_task
        from tcr.silver import load_gold
        gold = load_gold([self.run])
        aug = load_results(out_path(self.jmeta))
        out = []
        for t in super().tasks():
            if t["id"] in gold and not is_val_task(t["id"]):
                parents = list(t.get("parents") or []) or list((aug.get(t["id"]) or {}).get("parents") or [])
                out.append({**t, "parents": parents, "body": ""})
        return out

    def chunks(self, tasks: list) -> list:
        return [tasks[i:i + MAX_TASKS] for i in range(0, len(tasks), MAX_TASKS)]

    def prompt(self, chunk: list) -> str:
        lines = []
        for i, t in enumerate(chunk, 1):
            lines += [f"TASK {i}", f"TITLE: {t['title']}",
                      "PARENTS: " + (" > ".join(t.get("parents") or []) or "(none)"), ""]
        return "\n".join(lines)

    def estimate(self, chunks: list) -> float:
        p = J.PRICES[MODEL]
        tin = tout = 0
        for ch in chunks:
            chars = sum(len(t["title"]) + len(" > ".join(t.get("parents") or [])) + 30 for t in ch)
            tin += 700 + chars / 3.5
            tout += 4 * chars / 3 + 30 * len(ch) + 400
        return (tin * p[2] + tout * p[3]) / 1e6

    def check(self, task: dict, it) -> dict | None:
        if not it:
            return None
        from tcr.describe_v2 import norm
        variants = []
        for style, lang in RW_STYLES:
            t = " ".join(str(it.get(style) or "").split())
            if not t or len(t) > max(200, 3 * len(task["title"])):
                continue
            if lang == "en" and norm(t) == norm(task["title"]):
                continue
            if lang == "ru" and len(_WORD.findall(task["title"])) >= 2 and not _CYR.search(t):
                continue
            variants.append({"lang": lang, "style": style, "title": t})
        if len(variants) < 2:
            return None
        return {"task_id": task["id"], "variants": variants, "model": MODEL, "run": self.run}

    def finalize(self) -> None:
        from tcr.describe_v2 import contamination_hits
        results = load_results(self.out)
        todo = [r for r in results.values() if "contaminated" not in r]
        if not todo:
            return
        flat = [(r["task_id"], k, v["title"]) for r in todo for k, v in enumerate(r["variants"])]
        exact, near = contamination_hits([t for _, _, t in flat])
        flags = {r["task_id"]: [] for r in todo}
        for i in exact | near:
            flags[flat[i][0]].append(flat[i][1])
        for r in todo:
            r["contaminated"] = sorted(flags[r["task_id"]])
        save_results(self.out, results)
        vb.log(f"contamination check: {len(exact | near)} of {len(flat)} rewrites flagged")


# --- translations into more languages ---------------------------------------------------

LANG_STYLE = {
    "es": ("Spanish", 'the infinitive or a noun phrase ("Revisar el informe")'),
    "ru": ("Russian", 'the infinitive or a noun phrase ("Проверить отчёт")'),
    "de": ("German", 'a noun phrase or the infinitive at the end ("Bericht prüfen", "Zahnarzttermin")'),
    "fr": ("French", 'the infinitive or a noun phrase ("Vérifier le rapport", "RDV dentiste")'),
    "pt": ("Brazilian Portuguese", 'the infinitive or a noun phrase ("Revisar o relatório")'),
    "it": ("Italian", 'the infinitive or a noun phrase ("Controllare il report")'),
    "zh": ("Simplified Chinese", 'a short verb–object phrase ("检查报告", "预约牙医")'),
    "ja": ("Japanese", 'a short noun phrase or verb stem, as in a Japanese to-do list ("報告書を確認", "歯医者の予約")'),
    "pl": ("Polish", 'the infinitive or a noun phrase ("Sprawdzić raport", "Wizyta u dentysty")'),
}
SYSTEM_XL = """You translate personal to-do items from one person's org-mode task list into
{names}, the way a native speaker writes them in their own task list.

- Natural, concise to-do style, as natives write lists:
{styles}
- Translate the meaning, not word by word. Keep the level of detail: add nothing, drop
  nothing.
- Keep unchanged: names of products, tools, libraries, models, datasets, papers, people
  and places; code, identifiers, file names, URLs, numbers and dates; org markup.
- Widely used English technical terms may stay in English where a native speaker in the
  field would keep them.
- The task may already be in Spanish or Russian: translate from whatever language it is in.
- Return one entry per task, with the task's number n, in the same order."""
_CJK = re.compile("[぀-ヿ㐀-鿿]")
_LATIN = re.compile("[A-Za-zÀ-ÿĀ-ž]")


def _xl_path(meta: dict, eval_mode: bool) -> Path:
    if eval_mode:          # translations of eval tasks: eval only, never training
        return config.DATASETS_DIR / "xling" / f"{meta['dataset']}.xl.jsonl"
    base = (config.PRIVATE_DIR / "train_pools") if meta["private"] else (config.DATA_DIR / "train_pools")
    return base / f"{meta['dataset']}.xl.jsonl"


class Translate(Aug):
    """Title and parent headings into `langs`. Eval mode (--eval): the dev tasks of an
    eval set's judge run, for the translated eval set (xling-build); otherwise every task
    of a pool run, for training in more languages. One line per task:
    {"task_id", <lang>: {"title", "parents"}, "contaminated": [langs] (pool mode)}."""
    kind = "xl"

    def __init__(self, run: str, langs, eval_mode: bool):
        self.langs, self.eval_mode = list(langs), eval_mode
        self.kind = "xleval" if eval_mode else "xl"
        names = ", ".join(f"{LANG_STYLE[l][0]} ({l})" for l in self.langs)
        styles = "\n".join(f"  - {LANG_STYLE[l][0]}: {LANG_STYLE[l][1]}" for l in self.langs)
        self.system = SYSTEM_XL.format(names=names, styles=styles)
        lang_obj = {"type": "object", "properties": {"title": {"type": "string"}, "parents": _STR_LIST},
                    "required": ["title", "parents"]}
        self.schema = {"type": "object", "properties": {"tasks": {"type": "array", "items": {
            "type": "object", "properties": {"n": {"type": "integer"}, **{l: lang_obj for l in self.langs}},
            "required": ["n", *self.langs]}}}, "required": ["tasks"]}
        super().__init__(run, False)

    def out_file(self) -> Path:
        return _xl_path(self.jmeta, self.eval_mode)

    def tasks(self) -> list:
        ts = super().tasks()
        if self.eval_mode:
            from tcr.datasets import split_of
            from tcr.silver import load_gold
            gold = load_gold([self.run])
            ts = [t for t in ts if split_of(t["id"]) == "dev" and t["id"] in gold]
        else:
            aug = load_results(out_path(self.jmeta))
            ts = [{**t, "parents": list(t.get("parents") or []) or
                   list((aug.get(t["id"]) or {}).get("parents") or [])} for t in ts]
        return [{**t, "body": ""} for t in ts]

    def chunks(self, tasks: list) -> list:
        n = max(4, MAX_TASKS * 2 // max(2, len(self.langs)))
        return [tasks[i:i + n] for i in range(0, len(tasks), n)]

    def prompt(self, chunk: list) -> str:
        lines = ["The tasks carry their outline headings (PARENTS, outermost first, or \"(none)\"). "
                 "Translate the title and every heading into each language; [] for none.", ""]
        for i, t in enumerate(chunk, 1):
            lines += [f"TASK {i}", f"TITLE: {t['title']}",
                      "PARENTS: " + (" > ".join(t.get("parents") or []) or "(none)"), ""]
        return "\n".join(lines)

    def estimate(self, chunks: list) -> float:
        p = J.PRICES[MODEL]
        tin = tout = 0
        for ch in chunks:
            chars = sum(len(t["title"]) + len(" > ".join(t.get("parents") or [])) + 20 for t in ch)
            tin += 800 + chars / 3.5
            tout += len(self.langs) * (chars / 2 + 25 * len(ch)) + 400
        return (tin * p[2] + tout * p[3]) / 1e6

    def check(self, task: dict, it) -> dict | None:
        if not it:
            return None
        parents = list(task.get("parents") or [])
        out = {"task_id": task["id"], "parents": parents}
        latin_src = len(_WORD.findall(task["title"])) >= 2
        for lang in self.langs:
            tr = it.get(lang) or {}
            title = " ".join(str(tr.get("title") or "").split())
            tp = [" ".join(str(x).split()) for x in tr.get("parents") or []]
            if not title or len(tp) != len(parents) or not all(tp):
                return None
            if latin_src and lang in ("zh", "ja") and not _CJK.search(title):
                return None
            if latin_src and lang not in ("zh", "ja", "ru") and not _LATIN.search(title):
                return None
            out[lang] = {"title": title, "parents": tp}
        out.update(model=MODEL, run=self.run, langs=self.langs)
        return out

    def finalize(self) -> None:
        if self.eval_mode:
            return
        from tcr.describe_v2 import contamination_hits
        results = load_results(self.out)
        todo = [r for r in results.values() if "contaminated" not in r]
        flags = {r["task_id"]: [] for r in todo}
        for lang in self.langs:
            rs = [r for r in todo if lang in r]
            exact, near = contamination_hits([r[lang]["title"] for r in rs])
            for i in exact | near:
                flags[rs[i]["task_id"]].append(lang)
        for r in todo:
            r["contaminated"] = flags[r["task_id"]]
        save_results(self.out, results)


def xling_build() -> Path:
    """data/datasets/public_xling.jsonl from the eval-mode translations (shared schema)."""
    from tcr.datasets import DATASETS
    from tcr.silver import XLING_LANGS
    rows = []
    for p in sorted((config.DATASETS_DIR / "xling").glob("*.xl.jsonl")):
        src = p.name.split(".")[0]
        for r in read_jsonl(p):
            for lang in XLING_LANGS:
                if lang in r:
                    rows.append({"id": f"{r['task_id']}~{lang}", "dataset": "public_xling",
                                 "title": r[lang]["title"], "body": "", "parents": r[lang]["parents"],
                                 "lang": lang, "meta": {"orig_id": r["task_id"], "source": src,
                                                        "model": r.get("model")}})
    out = DATASETS["public_xling"].path
    write_jsonl(out, rows)
    vb.log(f"xling: {len(rows)} translated eval tasks -> {out}")
    return out


# --- commands ------------------------------------------------------------------------

def cmd_run(args) -> None:
    drive(Aug(args.judge_run, args.gen_parents), args)


def cmd_rewrite(args) -> None:
    drive(Rewrite(args.judge_run), args)


def cmd_translate(args) -> None:
    drive(Translate(args.judge_run, args.langs.split(","), args.eval), args)


def cmd_xling_build(args) -> None:
    xling_build()


def drive(a: Aug, args) -> None:
    """Rounds until every task is done: build → (budget check) submit → wait → collect."""
    if not a.jmeta.get("finished"):
        sys.exit(f"judge run {a.run} is not finished")
    if getattr(args, "limit", 0):
        a.tasks = (lambda f, n: (lambda: f()[:n]))(a.tasks, args.limit)
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
                run.submit(f"augment-{a.label}-r{k}")
            job = run.wait(args.poll)
            a.collect(k, job)
            continue
        pending = a.pending()
        if not pending:
            a.finalize()
            done = len(load_results(a.out))
            vb.log(f"{a.label}: finished; {a.out} holds {done} tasks")
            return
        k, chunks = a.build_round(pending)
        vb.log(f"round {k}: {len(pending)} tasks in {len(chunks)} requests; "
               f"estimate ${a.estimate(chunks):.2f}")


def cmd_status(args) -> None:
    a = {"aug": lambda: Aug(args.judge_run), "rw": lambda: Rewrite(args.judge_run),
         "xl": lambda: Translate(args.judge_run, args.langs.split(","), False),
         "xleval": lambda: Translate(args.judge_run, args.langs.split(","), True)}[args.kind]()
    ids = {t["id"] for t in a.tasks()}
    done = ids & set(load_results(a.out))
    m = a.meta
    failed = sum(1 for t, n in m["attempts"].items() if n >= MAX_ATTEMPTS and t not in done)
    print(f"{a.label}: {len(done)}/{len(ids)} done, {failed} given up; round {m['round']}")
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
    """The first trial: min(log + credit left, $300) − reserve, in cap.json. With --budget
    (later credits): log + min(credit left − reserve, budget), in cap-<tag>.json."""
    f = CAP_FILE if args.budget is None else WORK / f"cap-{args.tag}.json"
    if f.exists():
        print(json.loads(f.read_text())["cap"])
        return
    s = J.spent()
    if args.budget is None:
        cap = round(min(s + args.credit_left, TRIAL_USD) - args.reserve, 2)
    else:
        cap = round(s + min(args.credit_left - args.reserve, args.budget), 2)
    WORK.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"cap": cap, "spent_at_start": round(s, 2),
                             "credit_left": args.credit_left, "reserve": args.reserve,
                             "budget": args.budget, "set": now()}, indent=1))
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
    p.add_argument("--kind", choices=["aug", "rw", "xl", "xleval"], default="aug")
    p.add_argument("--langs", default=",".join(XLING_DEFAULT))
    for name in ("rewrite", "translate"):
        p = sub.add_parser(name)
        p.add_argument("--judge-run", required=True)
        p.add_argument("--max-usd", type=float, default=None)
        p.add_argument("--poll", type=int, default=120)
        p.add_argument("--limit", type=int, default=0, help="only the first N tasks (a trial)")
        if name == "translate":
            p.add_argument("--langs", default=",".join(XLING_DEFAULT))
            p.add_argument("--eval", action="store_true",
                           help="an eval set's run: its dev tasks, for the translated eval set")
    sub.add_parser("xling-build")
    sub.add_parser("backfill")
    p = sub.add_parser("cap")
    p.add_argument("--credit-left", type=float, required=True)
    p.add_argument("--reserve", type=float, default=8.0)
    p.add_argument("--budget", type=float, default=None, help="this round's budget (later credits)")
    p.add_argument("--tag", default="round3", help="with --budget: cap-<tag>.json")
    p = sub.add_parser("plan-extra")
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--generated", action="store_true", help="the new tasks exist already")
    p = sub.add_parser("extra")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
