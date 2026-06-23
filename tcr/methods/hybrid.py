"""M3 — hybrid retrieval: rank by Reciprocal Rank Fusion, gate on member signal.

RRF (score = sum_m 1/(k + rank_m)) is robust for *ordering* across methods with
incomparable score scales. But pure RRF encodes only ranks, so it cannot drive
abstention. We therefore:

  * order candidates by RRF, reporting each candidate's best member score for
    display, and
  * derive the abstention confidence as the max of the members' standardized
    gate signals — a real, scale-robust relevance signal (a strong signal from
    the lexical OR the dense member counts).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple

from .. import config
from .base import Matcher, Ranked


class HybridMatcher(Matcher):
    def __init__(self, matchers: Sequence[Matcher], rrf_k: int = config.RRF_K,
                 name: str = "M3", **kw):
        super().__init__(name=name, **kw)
        self.matchers = list(matchers)
        self.rrf_k = rrf_k

    def rank_and_signal(self, query_text: str, top_k: int = 10) -> Tuple[Ranked, float]:
        pool = max(top_k * 5, 50)
        rrf = defaultdict(float)     # name -> fused rank score (ordering)
        conf = defaultdict(float)    # name -> best member display score
        signals: List[float] = []    # per-member standardized gate signals
        for m in self.matchers:
            ranked, signal = m.rank_and_signal(query_text, top_k=pool)
            signals.append(signal)
            for rank, (name, score) in enumerate(ranked):
                rrf[name] += 1.0 / (self.rrf_k + rank + 1)
                conf[name] = max(conf[name], float(score))
        if not rrf:
            return [], 0.0
        order = sorted(rrf.keys(), key=lambda n: -rrf[n])[:top_k]
        ranked_out = [(name, conf[name]) for name in order]
        gate_signal = max(signals) if signals else 0.0
        return ranked_out, gate_signal
