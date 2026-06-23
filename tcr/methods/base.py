"""Common Matcher interface.

A Matcher scores all icons for a query (`full_scores`). From that we derive both
the ranked top-k and a scale-robust *gate signal*: how many standard deviations
the best icon stands above how the query matches the corpus overall. Absolute
cosine is a poor confidence signal (e.g. multilingual-e5 compresses everything
into ~0.8), but this standardized signal separates good matches from
"nothing fits" across encoders. See design/matching-methods.md (abstention gate).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..abstention import AbstentionGate, Decision
from ..org_query import Query

Ranked = List[Tuple[str, float]]


def standardized_confidence(scores: np.ndarray) -> float:
    """(max - mean) / std over the full score vector. 0 when undefined."""
    if scores is None or scores.size == 0:
        return 0.0
    std = float(scores.std())
    if std <= 1e-9:
        return 0.0
    return (float(scores.max()) - float(scores.mean())) / std


class Matcher:
    name: str = "base"

    def __init__(self, name: Optional[str] = None, gate: Optional[AbstentionGate] = None):
        if name:
            self.name = name
        self.gate = gate or AbstentionGate()

    # --- subclasses implement this --------------------------------------
    def full_scores(self, query_text: str) -> Tuple[Sequence[str], np.ndarray]:
        """Return (names, scores) over ALL icons, aligned."""
        raise NotImplementedError

    # --- generic, derived from full_scores ------------------------------
    def rank_and_signal(self, query_text: str, top_k: int = 10) -> Tuple[Ranked, float]:
        names, scores = self.full_scores(query_text)
        if scores is None or len(scores) == 0:
            return [], 0.0
        order = np.argsort(-scores)[:top_k]
        ranked = [(names[i], float(scores[i])) for i in order]
        return ranked, standardized_confidence(scores)

    def rank(self, query_text: str, top_k: int = 10) -> Ranked:
        return self.rank_and_signal(query_text, top_k=top_k)[0]

    # --- shared helpers --------------------------------------------------
    def _as_text(self, query: Union[str, Query]) -> str:
        return query.text if isinstance(query, Query) else str(query)

    def rank_query(self, query: Union[str, Query], top_k: int = 10) -> Ranked:
        return self.rank(self._as_text(query), top_k=top_k)

    def match_one(self, query: Union[str, Query], top_k: int = 10) -> Decision:
        ranked, conf = self.rank_and_signal(self._as_text(query), top_k=top_k)
        return self.gate.decide(ranked, conf)
