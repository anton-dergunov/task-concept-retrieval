"""#5 of round 6 (docs/argon-experiments.md): Argon as a listwise multimodal reranker.

Argon sees a task and the big model's (F1@all-large-all3k) top 20 as numbered glyph
images, in a shuffled order and never with names (CLAUDE.md hard rule 1), and returns
all 20 best first, each graded on the judge's scale. One request per task, no search.
The prompt keeps the judge's principles and few-shot block (tcr/judge.py), so the
grades share the judge's scale and can be read as labels.

Task sets (all under data/private/vlm/, private: Personal text):
  dev    the silver dev tasks of the five eval sets (the measurement)
  train  judged training tasks of the pools, not F1's validation tasks (same texts,
         judge's grades vs the VLM's)
  cand   unjudged candidate tasks (Personal-style + MS-LaTTE): new labels
  cand2  every other unjudged candidate, in the same hash order (the last batch, 2026-11-25)
  val    F1's validation tasks of the round-2 pools: the adversarial seeds (#2), first
         stage only, never reranked

The CLI is scripts/vlm_rerank.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config, silver
from . import judge as J
from .datasets import read_jsonl, split_of

WORK = config.PRIVATE_DIR / "vlm"
TOPK_DIR = WORK / "topk"
FIRST = "F1@all-large-all3k"          # the deployed big model (deployment.md)
FIRST_LABEL = FIRST + ".v2+parent"    # its silver label
LABEL = "VLM@all-large-all3k.v2+parent"
K = 20
MODEL = "gemini-4-argon"
SETS = ("dev", "train", "cand", "val", "cand2")
CAND_N, CAND_MS_SHARE, TRAIN_N = 3000, 0.25, 1000
ROUND2_POOL_RUNS = ("pool-synth-j4", "pool-synth2-j4", "pool-mslatte-j4")   # F1's val split applies


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


# --- task sets -----------------------------------------------------------------------

def _pool_parents(dataset: str) -> Dict[str, list]:
    from .finetune import AUG_FILES
    p = AUG_FILES.get(dataset)
    return {r["task_id"]: r.get("parents") or [] for r in read_jsonl(p)} if p and p.exists() else {}


def _judged_ids() -> set:
    ids = set()
    for d in silver.JUDGE_BATCH.iterdir():
        p = d / "tasks.jsonl"
        if p.exists():
            ids.update(r["id"] for r in read_jsonl(p))
    return ids


def set_tasks(name: str) -> List[dict]:
    """[{set, dataset, run, task_id, title, parents, body}]; `run` is the judge run whose
    snapshot (and gold) the task belongs to (None for cand). `parents` are what F1 reads
    (a pool task's generated headings); the judge's snapshot is what the VLM reads."""
    from .finetune import is_val_task
    out = []
    if name == "dev":
        for d, runs in silver.EVAL_RUNS.items():
            gold = silver.eval_gold(d)
            for t in silver.eval_tasks(d):
                if split_of(t["id"]) == "dev" and t["id"] in gold:
                    out.append({"set": name, "dataset": d, "run": runs[0], "task_id": t["id"],
                                "title": t["title"], "parents": t.get("parents") or [],
                                "body": t.get("body") or ""})
        return out
    if name in ("train", "val"):
        for ds, runs in silver.POOL_RUNS.items():
            aug = _pool_parents(ds)
            for run in runs:
                if name == "val" and run not in ROUND2_POOL_RUNS:
                    continue
                gold = silver.load_gold([run])
                for t in silver.run_tasks(run):
                    g = gold.get(t["id"])
                    if g is None:
                        continue
                    sel = (t.get("meta") or {}).get("selection")
                    val = run in ROUND2_POOL_RUNS and not sel and is_val_task(t["id"])
                    if val != (name == "val"):
                        continue
                    out.append({"set": name, "dataset": ds, "run": run, "task_id": t["id"],
                                "title": t["title"],
                                "parents": list(t.get("parents") or []) or aug.get(t["id"], []),
                                "body": t.get("body") or ""})
        if name == "train":
            out.sort(key=lambda r: _h("vlm-train:" + r["task_id"]))
            out = out[:TRAIN_N]
        return out
    if name in ("cand", "cand2"):
        import sys
        sys.path.insert(0, str(config.ROOT / "scripts" / "datasets"))
        from build_train_pools import mslatte_unused
        from .finetune import CANDIDATES_FILE
        judged = _judged_ids()
        synth = [r for r in read_jsonl(CANDIDATES_FILE) if r["id"] not in judged]
        ms = [r for r in mslatte_unused() if r["id"] not in judged]
        n_ms = round(CAND_N * CAND_MS_SHARE)
        synth = sorted(synth, key=lambda r: _h("vlm-cand:" + r["id"]))
        ms = sorted(ms, key=lambda r: _h("vlm-cand:" + r["id"]))
        if name == "cand":
            pick = synth[:CAND_N - n_ms] + ms[:n_ms]
        else:      # the rest; excludes cand's tasks even if the judged set has changed since
            first = {t["task_id"] for t in load_topk("cand").values()}
            pick = [r for r in synth[CAND_N - n_ms:] + ms[n_ms:] if r["id"] not in first]
        return [{"set": name, "dataset": r.get("dataset") or "cand", "run": None, "task_id": r["id"],
                 "title": r["title"], "parents": r.get("parents") or [], "body": r.get("body") or ""}
                for r in pick]
    raise ValueError(name)


def pilot_ids(n_dev: int = 30, n_train: int = 10) -> Dict[str, List[str]]:
    """Hash-chosen pilot tasks: dev spread over the five eval sets, plus training tasks."""
    dev = sorted(set_tasks("dev"), key=lambda r: _h("vlm-pilot:" + r["task_id"]))
    per = {"personal": 9, "personal_synth": 7, "realistic": 5, "public_short": 5, "public_expanded": 4}
    scale = n_dev / sum(per.values())
    got: Dict[str, List[str]] = {}
    for r in dev:
        if len(got.get(r["dataset"], [])) < round(per[r["dataset"]] * scale):
            got.setdefault(r["dataset"], []).append(r["task_id"])
    train = sorted(set_tasks("train"), key=lambda r: _h("vlm-pilot:" + r["task_id"]))[:n_train]
    return {"dev": [i for v in got.values() for i in v], "train": [r["task_id"] for r in train]}


# --- the first stage's top K ----------------------------------------------------------

def topk_path(name: str) -> Path:
    return TOPK_DIR / f"{name}.jsonl"


def load_topk(name: str) -> Dict[str, dict]:
    p = topk_path(name)
    return {r["task_id"]: r for r in read_jsonl(p)} if p.exists() else {}


def score_set(name: str, batch: int = 16, log=print, only: Optional[set] = None) -> Path:
    """The big model's served top K (rejected glyphs dropped) for every task of the set
    (or only the ids `only`), appended to topk/<set>.jsonl as it goes (resumable).
    Threads: TCR_F1_THREADS."""
    tasks = set_tasks(name)
    done = load_topk(name)
    todo = [t for t in tasks if t["task_id"] not in done and (only is None or t["task_id"] in only)]
    log(f"{name}: {len(tasks)} tasks, {len(todo)} to score")
    if todo:
        score_rows(todo, topk_path(name), batch=batch, log=log, tag=name)
    return topk_path(name)


_FIRST = None


def first_stage():
    global _FIRST
    if _FIRST is None:
        from .data import load_icons
        from .methods import build_method
        _FIRST = build_method(FIRST, load_icons(version="v2"))
    return _FIRST


def score_rows(todo: Sequence[dict], path: Path, batch: int = 16, log=print, tag: str = "") -> None:
    """Append {set, dataset, run, task_id, query, top, scores, signal} per row (rows carry
    task_id, title, parents and optionally set, dataset, run) to `path`."""
    import time
    from .finetune import MAX_QUERY_LEN, QUERY_PREFIX
    from .org_tasks import query_text
    m = first_stage()
    rej = silver.rejected_canonical()
    keep = np.array([n not in rej for n in m.names])
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            qs = [query_text(t["title"], parents=t.get("parents") or []) for t in chunk]
            V = m.enc.encode([QUERY_PREFIX + q for q in qs], MAX_QUERY_LEN, batch=batch)
            S = V @ m.matrix.T
            for t, q, sc in zip(chunk, qs, S):
                served = np.where(keep, sc, -np.inf)
                order = np.argsort(-served)[:K]
                std = float(sc.std()) or 1.0
                f.write(json.dumps({"set": t.get("set"), "dataset": t.get("dataset"), "run": t.get("run"),
                                    "task_id": t["task_id"], "query": q,
                                    "top": [m.names[i] for i in order],
                                    "scores": [round(float(sc[i]), 5) for i in order],
                                    "signal": round((float(sc.max()) - float(sc.mean())) / std, 4)},
                                   ensure_ascii=False) + "\n")
            f.flush()
            if (s // batch) % 20 == 0:
                rate = (time.time() - t0) / (s + len(chunk))
                log(f"  {tag}: {s + len(chunk)}/{len(todo)} ({rate:.2f} s/task)")


# --- the request ----------------------------------------------------------------------

ARMS = ("img+desc", "img")
_DESC = ", each with a one-line description of what it draws"
assert _DESC in J.SYSTEM
SYSTEM = {"img+desc": J.SYSTEM,
          "img": J.SYSTEM.replace(_DESC, "").replace(" Look at the image itself, not only its description.",
                                                     " Read each image closely.")}

RERANK = """\
Rank the {n} CANDIDATES for the TASK above: do steps 1, 2 and 4 of your procedure (there \
is no search: these candidates are all you have).
- subject: the central subject of the task, in a few words (the head of the title in the \
sense the task means it; the parent's topic when the title is generic or figurative).
- ranking: all {n} candidate numbers, each exactly once, best reminder first, each with a \
grade: 3 = shows the central subject; 2 = shows its context, domain or activity; 1 = a \
partial or loose association; 0 = not a useful reminder for this task. Order the icons \
graded 0 too, by how close they come.
  Order: when the subject is a thing that can be drawn, its plain icon comes first; then \
its context (the parent's domain); then icons of a specific aspect, operation or \
metaphor. A clever icon of an aspect does not beat the plain icon of the object. Only \
when the subject is an idea with no drawable object does its best metaphor come first.
- why: a short reason for each of the first 3 entries only; leave it out for the rest.
- no_good_icon: true only when nothing here is better than a generic or property-level \
icon."""

SCHEMA = {"type": "object", "properties": {
    "subject": {"type": "string"},
    "ranking": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "integer"}, "grade": {"type": "integer"}, "why": {"type": "string"}},
        "required": ["id", "grade"]}},
    "no_good_icon": {"type": "boolean"}},
    "required": ["subject", "ranking", "no_good_icon"]}


@lru_cache(maxsize=None)
def _cand_tasks(name: str = "cand") -> tuple:
    return tuple(set_tasks(name))


def shuffled(task_id: str, top: Sequence[str]) -> List[str]:
    """The candidates in a per-task hash order: the VLM must not read the first stage's."""
    return sorted(top, key=lambda n: _h(f"vlm-order:{task_id}:{n}"))


class Contexts:
    """J.Context per judge run (its snapshot, as the judge saw the tasks) or per set."""

    def __init__(self):
        self._c: Dict[str, J.Context] = {}

    def get(self, row: dict, rec: Optional[dict] = None) -> J.Context:
        key = row["run"] or f"{row['set']}:{row['dataset']}"     # cand / cand2: no judge run
        if key not in self._c:
            if row["run"]:
                self._c[key] = J.Context(row["dataset"], records=silver.run_tasks(row["run"]))
            else:
                recs = [{"id": t["task_id"], "title": t["title"], "parents": t["parents"], "body": t["body"]}
                        for t in _cand_tasks(row["set"]) if t["dataset"] == row["dataset"]]
                self._c[key] = J.Context(row["dataset"], records=recs)
        return self._c[key]


def request(ctxs: Contexts, row: dict, arm: str):
    """(system, parts, numbered) for one task: parts end with the instruction;
    numbered = [(number, icon)] in the order shown."""
    ctx = ctxs.get(row)
    rec = ctx.by_id[row["task_id"]]
    numbered = [(i + 1, n) for i, n in enumerate(shuffled(row["task_id"], row["top"]))]
    parts = J._header(ctx, rec) + [f"CANDIDATES: {len(numbered)} icons."]
    if arm == "img+desc":
        parts += J.sheets_parts(numbered, "Candidates")
    else:
        parts += [f"Candidates, sheet 1 (numbers 1–{len(numbered)}):", (J.sheet_png(numbered), "image/png")]
    return SYSTEM[arm], parts + [RERANK.format(n=len(numbered))], numbered


def parse(data: dict, numbered: Sequence[tuple], first: Sequence[str]) -> dict:
    """The VLM's answer as {order (icons, best first), grades, why, missing, subject,
    no_good_icon}. Numbers it left out are appended in the first stage's order, grade 0."""
    by_num = dict(numbered)
    order, grades, why = [], {}, {}
    for e in (data or {}).get("ranking") or []:
        try:
            n = by_num.get(int(e.get("id")))
        except (TypeError, ValueError):
            continue
        if n is None or n in grades:
            continue
        order.append(n)
        grades[n] = max(0, min(3, int(e.get("grade") or 0)))
        if e.get("why"):
            why[n] = str(e["why"])
    missing = [n for n in first if n not in grades]
    for n in missing:
        order.append(n)
        grades[n] = 0
    return {"order": order, "grades": grades, "why": why, "missing": len(missing),
            "subject": str((data or {}).get("subject") or ""),
            "no_good_icon": bool((data or {}).get("no_good_icon"))}


def ranked_scores(ans: dict) -> List[float]:
    """A score per icon of the VLM order for stored rankings: grade, then rank."""
    n = len(ans["order"])
    return [round(ans["grades"][x] + (n - i) / (n + 1), 5) for i, x in enumerate(ans["order"])]


def threads() -> None:
    """One thread unless told otherwise (the author's Mac is shared)."""
    os.environ.setdefault("TCR_F1_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", os.environ["TCR_F1_THREADS"])
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["TCR_F1_THREADS"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
