"""Bootstrap evaluation harness + leaderboard.

Real silver/gold labels arrive next round; this harness wires every metric so
methods are comparable now. It is intentionally optimistic/biased (see below)
and is for relative comparison + plumbing, not absolute numbers.

Design (see design/experimentation-strategy.md §8 and the plan):
  Positives (weak): sample icons, hold out ONE example_task each as a query with
    gold = that icon. Document-based methods (B0/B1/M1) are built WITHOUT
    example_tasks; M2 is built excluding exactly the held-out queries — so no
    method can trivially self-match.
  Negatives / abstention: poor_matches queries (named icon should NOT be top-1)
    and gibberish queries (should abstain). Used for the precision-coverage
    curve, gate AUROC, and poor-match suppression.

Run: python -m tcr.eval [--sample N] [--methods B0,B1,M1,M2,M3]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config
from .data import IconDoc, load_icons
from .methods.base import Matcher
from .methods.bm25 import BM25Matcher
from .methods.biencoder import BiEncoderMatcher
from .methods.nearest_example import NearestExampleMatcher
from .methods.hybrid import HybridMatcher

GIBBERISH = [
    "asdfqwer zxcv", "blorp glarn fizzbuzzle", "qwertyuiop", "zzxxcc vvbb nnmm",
    "lorem ipsum dolor sit", "xyzzy plugh frobnicate",
]

MULTILINGUAL_SMOKE = [
    "Caminar 8000 pasos al día",          # es: walk 8000 steps a day
    "Reservar cita con el dentista",       # es: book dentist appointment
    "Купить продукты в магазине",          # ru: buy groceries
    "Прочитать главу книги",               # ru: read a book chapter
]


@dataclass
class EvalSet:
    queries: List[str]            # held-out example queries
    gold: List[str]               # gold icon name per query
    holdout: frozenset            # the exact query strings (for M2 exclusion)
    poor_queries: List[str]       # poor_match strings
    poor_gold_bad: List[str]      # the icon each poor query should NOT match


def build_eval_set(sample: int, min_usefulness: float, seed: int = 0) -> EvalSet:
    rng = random.Random(seed)
    icons = [ic for ic in load_icons()
             if ic.usefulness >= min_usefulness and len(ic.example_tasks) >= 2]
    rng.shuffle(icons)
    chosen = icons[:sample]
    queries, gold = [], []
    for ic in chosen:
        ex = rng.choice(ic.example_tasks)
        queries.append(ex)
        gold.append(ic.name)
    holdout = frozenset(queries)

    poor_q, poor_bad = [], []
    for ic in chosen:
        if ic.poor_matches:
            poor_q.append(rng.choice(ic.poor_matches))
            poor_bad.append(ic.name)
    return EvalSet(queries, gold, holdout, poor_q, poor_bad)


def build_methods(keys: List[str], holdout: frozenset) -> Dict[str, Matcher]:
    """Construct matchers for the eval with the example holdout applied."""
    icons = load_icons()
    out: Dict[str, Matcher] = {}
    for k in keys:
        if k == "B0":
            out[k] = BiEncoderMatcher(icons, model_name=config.ENGLISH_MODEL, view="document",
                                      field_weights=None, quality_weight=0.0,
                                      include_examples=False, name="B0")
        elif k == "B1":
            out[k] = BM25Matcher(icons, include_examples=False, name="B1")
        elif k == "M1":
            out[k] = BiEncoderMatcher(icons, model_name=config.MULTILINGUAL_MODEL, view="field",
                                      field_weights=config.FIELD_WEIGHTS,
                                      quality_weight=config.QUALITY_PRIOR_WEIGHT,
                                      include_examples=False, name="M1")
        elif k == "M2":
            out[k] = NearestExampleMatcher(icons, model_name=config.MULTILINGUAL_MODEL,
                                           quality_weight=config.QUALITY_PRIOR_WEIGHT,
                                           exclude_examples=holdout, name="M2")
        elif k == "M3":
            b1 = BM25Matcher(icons, include_examples=False, name="B1")
            m1 = BiEncoderMatcher(icons, model_name=config.MULTILINGUAL_MODEL, view="field",
                                  field_weights=config.FIELD_WEIGHTS,
                                  quality_weight=config.QUALITY_PRIOR_WEIGHT,
                                  include_examples=False, name="M1")
            out[k] = HybridMatcher([b1, m1], name="M3")
    return out


def _dcg(rel: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel))


def evaluate_method(matcher: Matcher, es: EvalSet, k: int = 10) -> dict:
    n = len(es.queries)
    p_at_1 = 0
    recall_at_k = 0
    rr_sum = 0.0
    ndcg_sum = 0.0
    top1_conf: List[float] = []     # standardized gate confidence per positive query
    top1_correct: List[int] = []    # 1 if top-1 == gold
    latencies: List[float] = []

    for q, g in zip(es.queries, es.gold):
        t0 = time.perf_counter()
        ranked, conf = matcher.rank_and_signal(q, top_k=k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        names = [nm for nm, _ in ranked]
        top1_conf.append(conf)
        correct = 1 if names and names[0] == g else 0
        top1_correct.append(correct)
        p_at_1 += correct
        if g in names:
            rank = names.index(g)
            recall_at_k += 1
            rr_sum += 1.0 / (rank + 1)
            rel = [1 if nm == g else 0 for nm in names]
            ndcg_sum += _dcg(rel) / _dcg(sorted(rel, reverse=True)) if any(rel) else 0.0

    # gibberish: should abstain -> collect gate confidence as negatives
    gib_conf: List[float] = []
    for q in GIBBERISH:
        _, conf = matcher.rank_and_signal(q, top_k=k)
        gib_conf.append(conf)

    # poor-match suppression: named icon should NOT be top-1
    suppressed = 0
    for q, bad in zip(es.poor_queries, es.poor_gold_bad):
        ranked = matcher.rank_query(q, top_k=k)
        if not ranked or ranked[0][0] != bad:
            suppressed += 1
    poor_supp = suppressed / len(es.poor_queries) if es.poor_queries else float("nan")

    # gate AUROC: separate "should show" (correct top-1) from "should not"
    # (wrong top-1 on positives + gibberish), using the standardized confidence.
    gate_scores = top1_conf + gib_conf
    gate_labels = top1_correct + [0] * len(gib_conf)
    auroc = _safe_auroc(gate_scores, gate_labels)

    # precision-coverage: sort positives by gate confidence desc; precision among
    # the top X% shown (correct == top1==gold).
    pc = _precision_coverage(top1_conf, top1_correct, coverages=(0.5, 0.7, 0.9))

    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)] if latencies else 0.0

    return {
        "precision_at_1": p_at_1 / n,
        f"recall_at_{k}": recall_at_k / n,
        "mrr": rr_sum / n,
        f"ndcg_at_{k}": ndcg_sum / n,
        "gate_auroc": auroc,
        "poor_match_suppression": poor_supp,
        "precision_at_coverage": pc,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "n_queries": n,
    }


def _safe_auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    if len(set(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return round(float(roc_auc_score(labels, scores)), 4)
    except Exception:
        return None


def _precision_coverage(scores: List[float], correct: List[int],
                        coverages=(0.5, 0.7, 0.9)) -> Dict[str, float]:
    order = np.argsort(-np.asarray(scores))
    correct_sorted = np.asarray(correct)[order]
    n = len(scores)
    out = {}
    for cov in coverages:
        m = max(1, int(round(cov * n)))
        out[f"p@1_cov{int(cov*100)}"] = round(float(correct_sorted[:m].mean()), 4)
    return out


def multilingual_smoke(matcher: Matcher, k: int = 3) -> List[dict]:
    rows = []
    for q in MULTILINGUAL_SMOKE:
        ranked = matcher.rank_query(q, top_k=k)
        rows.append({"query": q, "top": [{"name": n, "score": round(s, 3)} for n, s in ranked]})
    return rows


def leaderboard_md(results: Dict[str, dict], k: int) -> str:
    cols = ["precision_at_1", f"recall_at_{k}", "mrr", f"ndcg_at_{k}",
            "gate_auroc", "poor_match_suppression", "latency_p50_ms", "latency_p95_ms"]
    header = "| method | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for m, r in results.items():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="tcr.eval")
    parser.add_argument("--sample", type=int, default=150, help="number of held-out positive queries")
    parser.add_argument("--min-usefulness", type=float, default=0.6)
    parser.add_argument("--methods", default="B0,B1,M1,M2,M3")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    keys = [m.strip() for m in args.methods.split(",") if m.strip()]
    es = build_eval_set(args.sample, args.min_usefulness, seed=args.seed)
    print(f"Eval set: {len(es.queries)} positive queries, "
          f"{len(es.poor_queries)} poor-match queries, {len(GIBBERISH)} gibberish.\n")

    methods = build_methods(keys, es.holdout)
    results: Dict[str, dict] = {}
    multiling: Dict[str, list] = {}
    for key, matcher in methods.items():
        print(f"Evaluating {key} ...", flush=True)
        results[key] = evaluate_method(matcher, es, k=args.k)
        if key in ("M1", "M3"):
            multiling[key] = multilingual_smoke(matcher)

    md = leaderboard_md(results, args.k)
    print("\n## Leaderboard\n")
    print(md)

    print("\n## Precision @ coverage\n")
    for key, r in results.items():
        print(f"  {key}: {r['precision_at_coverage']}")

    if multiling:
        print("\n## Multilingual smoke (qualitative; ES/RU)\n")
        for key, rows in multiling.items():
            print(f"  [{key}]")
            for row in rows:
                top = ", ".join(f"{t['name']}({t['score']})" for t in row["top"])
                print(f"    {row['query']!r} -> {top}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = config.RESULTS_DIR / f"eval-{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"sample": args.sample, "min_usefulness": args.min_usefulness,
                       "methods": keys, "k": args.k, "seed": args.seed,
                       "multilingual_model": config.MULTILINGUAL_MODEL,
                       "english_model": config.ENGLISH_MODEL,
                       "abstain_threshold": config.ABSTAIN_THRESHOLD},
            "results": results,
            "multilingual_smoke": multiling,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
