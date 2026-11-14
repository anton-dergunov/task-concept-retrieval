"""Silver labels: the LLM judge's graded rankings as gold (docs/next-label-free-work.md
step 1) and as training targets (step 2).

Gold for a task is the judge's ranking with its grades (3 / 2 / 1) as gains, per glyph:
every icon name is mapped to the name `load_icons` keeps for its glyph, so aliases count
once. Icons whose glyph the author rejected in the manual review are removed from gold
(a method that shows one gets no credit), but methods are *not* filtered: `rej@1`
measures how often a method's top-1 is a rejected glyph, and the "served" metrics apply
the production filter (`tcr.data.rejected_names`) to the ranking first.

The judge saw more icons than it graded (its candidate pool plus its searches); an icon
it saw and left ungraded was judged not relevant, an icon it never saw is *unjudged*.
`seen` holds the former, from the judge run's states (private, local only).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence

from . import config
from .data import glyph_hashes, load_icons, rejected_names
from .datasets import read_jsonl

JUDGE_BATCH = config.PRIVATE_DIR / "judge" / "batch"
PUBLIC_LABELS = config.RESULTS_DIR / "judge_labels"
PRIVATE_LABELS = config.PRIVATE_DIR / "judge" / "labels"

# dataset key -> judge runs (label file stem = run name). Eval sets first, then the
# training pools; the pools' task text comes from each run's own snapshot
# (JUDGE_BATCH/<run>/tasks.jsonl), since the pool file was rewritten after judging.
EVAL_RUNS: Dict[str, List[str]] = {
    "personal": ["personal-j4"],
    "personal_synth": ["synth-j4"],
    "realistic": ["realistic-j4"],
    "public_short": ["public-short-j4"],
    "public_expanded": ["public-expanded-j4"],
}
POOL_RUNS: Dict[str, List[str]] = {
    "personal_synth_pool": ["pool-synth-j4", "pool-synth2-j4"],
    "train_mslatte": ["pool-mslatte-j4"],
}


@dataclass
class Gold:
    """The judge's verdict on one task, per canonical glyph name."""
    task_id: str
    grades: Dict[str, int]                  # icon -> 3 / 2 / 1, rejected glyphs removed
    seen: FrozenSet[str] = frozenset()      # every icon the judge looked at (incl. graded)
    order: List[str] = field(default_factory=list)   # the judge's ranking, as canonical names
    no_good_icon: bool = False


# --- glyph identity --------------------------------------------------------------------

@lru_cache(maxsize=1)
def _canon_map() -> Dict[str, str]:
    """glyph hash -> the representative name the methods use (v1's choice, then v2)."""
    hashes = glyph_hashes()
    out: Dict[str, str] = {}
    for ver in ("v1", "v2"):
        for ic in load_icons(version=ver):
            h = hashes.get(ic.name)
            if h is not None:
                out.setdefault(h, ic.name)
    return out


def canonical(name: str) -> str:
    """The name the methods use for `name`'s glyph (itself when no method has it)."""
    return _canon_map().get(glyph_hashes().get(name), name)


@lru_cache(maxsize=1)
def rejected_canonical() -> FrozenSet[str]:
    return frozenset(canonical(n) for n in rejected_names())


# --- loading ---------------------------------------------------------------------------

def label_path(run: str) -> Path:
    pub = PUBLIC_LABELS / f"{run}.jsonl"
    return pub if pub.exists() else PRIVATE_LABELS / f"{run}.jsonl"


def _seen_by_run(run: str) -> Dict[str, FrozenSet[str]]:
    p = JUDGE_BATCH / run / "states.jsonl"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                out[s["task_id"]] = frozenset(canonical(n) for n in (s.get("ids") or {}).values())
    return out


def load_gold(runs: Sequence[str]) -> Dict[str, Gold]:
    """{task_id: Gold} over the label files of `runs`; failed tasks are left out."""
    rej = rejected_canonical()
    out: Dict[str, Gold] = {}
    for run in runs:
        seen = _seen_by_run(run)
        for lab in read_jsonl(label_path(run)):
            if lab.get("failed"):
                continue
            grades: Dict[str, int] = {}
            order: List[str] = []
            for e in lab["ranking"]:
                n = canonical(e["icon"])
                if n in rej or n in grades:
                    continue
                grades[n] = int(e["grade"])
                order.append(n)
            s = seen.get(lab["task_id"], frozenset())
            out[lab["task_id"]] = Gold(lab["task_id"], grades, s | frozenset(grades), order,
                                       bool(lab.get("no_good_icon")))
    return out


def run_tasks(run: str) -> List[dict]:
    """The tasks of a judge run exactly as the judge saw them (its snapshot)."""
    return read_jsonl(JUDGE_BATCH / run / "tasks.jsonl")


# --- metrics ---------------------------------------------------------------------------

def ndcg(grades: Dict[str, int], ranked: Sequence[str], k: int = 5) -> float:
    """Graded nDCG@k: gain = the judge's grade, ideal = gold sorted by grade."""
    dcg = sum(grades.get(n, 0) / math.log2(i + 2) for i, n in enumerate(ranked[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def served(ranked: Sequence[str]) -> List[str]:
    """The ranking a production export shows: rejected glyphs removed."""
    rej = rejected_canonical()
    return [n for n in ranked if canonical(n) not in rej]


def task_metrics(g: Gold, ranked: Sequence[str], k: int = 5) -> dict:
    """Per-task silver metrics of one method's ranking (names as the method returns)."""
    r = [canonical(n) for n in ranked]
    s = served(r)
    top, stop = (r[0] if r else None), (s[0] if s else None)
    m = {
        "ndcg5": ndcg(g.grades, r, k),
        "hit1": int(top is not None and g.grades.get(top, 0) >= 2),
        "judged1": int(top is not None and top in g.seen),
        "rej1": int(top is not None and top in rejected_canonical()),
        "ndcg5_served": ndcg(g.grades, s, k),
        "hit1_served": int(stop is not None and g.grades.get(stop, 0) >= 2),
    }
    m["hit1_judged"] = m["hit1"] if m["judged1"] else None
    return m


METRICS = ("ndcg5", "hit1", "judged1", "hit1_judged", "rej1", "ndcg5_served", "hit1_served")


def aggregate(per_task: Dict[str, dict]) -> dict:
    from .judging import wilson
    out = {"n": len(per_task)}
    for k in METRICS:
        vals = [m[k] for m in per_task.values() if m.get(k) is not None]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    hits = sum(m["hit1"] for m in per_task.values())
    out["hit1_ci"] = [round(x, 3) for x in wilson(hits, len(per_task))]
    return out


def paired(a: Dict[str, dict], b: Dict[str, dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Method a vs b on the same tasks: mean ΔnDCG@5 with a bootstrap 95% interval, and
    McNemar's exact test on hit@1 (only the discordant tasks count)."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {}
    import numpy as np
    d = np.array([a[t]["ndcg5"] - b[t]["ndcg5"] for t in ids])
    n = len(d)
    boots = d[np.random.default_rng(seed).integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    only_a = sum(a[t]["hit1"] and not b[t]["hit1"] for t in ids)
    only_b = sum(b[t]["hit1"] and not a[t]["hit1"] for t in ids)
    return {"n": n, "d_ndcg5": round(float(d.mean()), 4),
            "d_ndcg5_ci": [round(float(lo), 4), round(float(hi), 4)],
            "hit1_only_a": only_a, "hit1_only_b": only_b,
            "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4)}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: Binomial(b + c, 1/2) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
