"""Dense bi-encoder matchers.

B0: single concatenated document, English bge-small-en, plain cosine (replica of
    the current emoji matcher).
M1: multilingual encoder, per-field cosine combined with config field weights,
    plus an icon_usefulness quality prior.

Scores are kept in a cosine-like [0,1]-ish range so the abstention threshold is
meaningful; the quality prior is a small additive nudge.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import config
from ..data import IconDoc, TEXT_FIELDS
from ..index import IconIndex, embed_texts
from .base import Matcher, Ranked


class BiEncoderMatcher(Matcher):
    def __init__(
        self,
        icons: Sequence[IconDoc],
        model_name: str,
        view: str = "document",
        field_weights: Optional[Dict[str, float]] = None,
        quality_weight: float = 0.0,
        include_examples: bool = True,
        name: str = "biencoder",
        **kw,
    ):
        super().__init__(name=name, **kw)
        self.icons = list(icons)
        self.names = [ic.name for ic in self.icons]
        self.model_name = model_name
        self.quality_weight = quality_weight
        self.usefulness = np.array([ic.usefulness for ic in self.icons], dtype=np.float32)

        if view == "field":
            # one index per field, combined with weights
            fw = field_weights or {f: 1.0 for f in TEXT_FIELDS}
            if not include_examples:
                fw = {k: v for k, v in fw.items() if k != "example_tasks"}
            self.field_weights = fw
            self.field_indices = {
                f: IconIndex.build(self.icons, model_name, view=f, include_examples=include_examples)
                for f in fw
            }
            self.doc_index = None
        else:
            self.field_weights = None
            self.field_indices = None
            self.doc_index = IconIndex.build(
                self.icons, model_name, view="document", include_examples=include_examples
            )

    def _scores(self, query_text: str) -> np.ndarray:
        qvec = embed_texts(self.model_name, [query_text], is_query=True)[0]
        if self.doc_index is not None:
            sims = self.doc_index.cosine(qvec)
        else:
            total_w = sum(self.field_weights.values()) or 1.0
            sims = np.zeros(len(self.icons), dtype=np.float32)
            for f, w in self.field_weights.items():
                sims += w * self.field_indices[f].cosine(qvec)
            sims /= total_w
        if self.quality_weight:
            sims = sims + self.quality_weight * self.usefulness
        return sims

    def full_scores(self, query_text: str):
        return self.names, self._scores(query_text)
