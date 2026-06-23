"""M2 — nearest-example voting (asymmetric).

Each icon ships example_tasks (which are *literally tasks*). We embed every
example, find the ones most similar to the query, then aggregate per icon (max
similarity), plus a quality prior. Exploits task<->task similarity, the same
distribution as the query.

The flattened example matrix (~tens of thousands of short texts) is cached to
.npz keyed on (model, icon-name hash) so repeat runs are instant.
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np

from .. import config
from ..data import IconDoc
from ..index import embed_texts, get_model
from .base import Matcher, Ranked


def _cache_path(model_name: str, names: Sequence[str]) -> "object":
    h = hashlib.sha1()
    h.update(model_name.encode())
    h.update("\n".join(names).encode())
    safe = model_name.replace("/", "_")
    return config.CACHE_DIR / f"{safe}__examples__{h.hexdigest()[:16]}.npz"


class NearestExampleMatcher(Matcher):
    def __init__(self, icons: Sequence[IconDoc], model_name: str,
                 quality_weight: float = 0.0, exclude_examples: frozenset = frozenset(),
                 name: str = "M2", **kw):
        super().__init__(name=name, **kw)
        self.icons = list(icons)
        self.names = [ic.name for ic in self.icons]
        self.model_name = model_name
        self.quality_weight = quality_weight
        self.exclude_examples = frozenset(exclude_examples)
        self.usefulness = np.array([ic.usefulness for ic in self.icons], dtype=np.float32)
        self._build_example_matrix()

    def _build_example_matrix(self) -> None:
        # owner[k] = index of the icon that example k belongs to.
        # Held-out examples (eval queries) are excluded so M2 can't self-match.
        owners: List[int] = []
        texts: List[str] = []
        for i, ic in enumerate(self.icons):
            for ex in ic.example_tasks:
                if ex in self.exclude_examples:
                    continue
                owners.append(i)
                texts.append(ex)
        self.owners = np.array(owners, dtype=np.int32)

        # Skip the on-disk cache when a holdout is active (the matrix is run-specific).
        path = _cache_path(self.model_name, self.names)
        matrix = None
        if not self.exclude_examples and path.exists():
            try:
                data = np.load(path)
                if str(data["model"]) == self.model_name and int(data["n_examples"]) == len(texts):
                    matrix = data["matrix"]
            except Exception:
                matrix = None
        if matrix is None:
            model = get_model(self.model_name)
            prefix = config.doc_prefix_for(self.model_name)
            payload = [prefix + t for t in texts] if prefix else texts
            emb = model.encode(payload, convert_to_numpy=True, show_progress_bar=True)
            emb = np.asarray(emb, dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = emb / norms
            if not self.exclude_examples:
                config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, model=self.model_name, n_examples=len(texts), matrix=matrix)
        self.example_matrix = matrix  # (n_examples, dim) normalized

    def full_scores(self, query_text: str):
        qvec = embed_texts(self.model_name, [query_text], is_query=True)[0]
        sims = self.example_matrix @ qvec  # (n_examples,)
        # per-icon max over its examples (np.maximum.at for grouped max)
        per_icon = np.full(len(self.icons), -1.0, dtype=np.float32)
        np.maximum.at(per_icon, self.owners, sims)
        per_icon[per_icon < 0] = 0.0  # icons with no examples
        if self.quality_weight:
            per_icon = per_icon + self.quality_weight * self.usefulness
        return self.names, per_icon
