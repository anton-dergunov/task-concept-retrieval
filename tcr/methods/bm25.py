"""B1 — BM25 lexical baseline over icon documents.

Fast, interpretable, no model download. Also the engine behind the keyboard
search box in the web UI. Scores are min-max normalized to [0,1] per query so
they are comparable with the dense methods and usable by the abstention gate.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from rank_bm25 import BM25Okapi

from ..data import IconDoc
from .base import Matcher, Ranked

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Matcher(Matcher):
    def __init__(self, icons: Sequence[IconDoc], include_examples: bool = True,
                 name: str = "B1", **kw):
        super().__init__(name=name, **kw)
        self.icons = list(icons)
        self.names = [ic.name for ic in self.icons]
        corpus = [_tokenize(ic.document(include_examples=include_examples)) for ic in self.icons]
        self.bm25 = BM25Okapi(corpus)

    # Saturation constant for the bounded confidence transform raw/(raw+K).
    SAT_K = 8.0

    def full_scores(self, query_text: str):
        import numpy as np
        toks = _tokenize(query_text)
        if not toks:
            return self.names, np.zeros(len(self.names), dtype=np.float32)
        raw = np.asarray(self.bm25.get_scores(toks), dtype=np.float32)
        # Bounded confidence that stays LOW when there is little overlap (so the
        # abstention gate can fire), instead of min-max which forces top -> 1.0.
        scores = raw / (raw + self.SAT_K)
        scores[raw <= 0] = 0.0
        return self.names, scores
