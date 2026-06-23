"""Embedding index over icon documents with on-disk caching.

Reuses the cache pattern from productivity-system/scripts/org_emoji_matcher.py:
an .npz keyed on (icon names, model, view) so a stale cache is never silently
mixed with a different vector space. Encoding 4k short docs takes a few seconds;
caching makes repeat runs instant.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np

from . import config
from .data import IconDoc

# A single SentenceTransformer per model name (loading is expensive).
_MODELS: dict = {}


def get_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODELS:
        _MODELS[model_name] = SentenceTransformer(model_name)
    return _MODELS[model_name]


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _cache_path(model_name: str, view: str, names: Sequence[str]) -> "object":
    h = hashlib.sha1()
    h.update(model_name.encode())
    h.update(view.encode())
    h.update("\n".join(names).encode())
    digest = h.hexdigest()[:16]
    safe_model = model_name.replace("/", "_")
    return config.CACHE_DIR / f"{safe_model}__{view}__{digest}.npz"


def embed_texts(model_name: str, texts: List[str], is_query: bool = False) -> np.ndarray:
    """Encode arbitrary texts (no caching) — used for queries."""
    model = get_model(model_name)
    prefix = config.query_prefix_for(model_name) if is_query else config.doc_prefix_for(model_name)
    payload = [prefix + t for t in texts] if prefix else texts
    emb = model.encode(payload, convert_to_numpy=True, show_progress_bar=False)
    return _normalize(np.asarray(emb, dtype=np.float32))


class IconIndex:
    """Holds the normalized embedding matrix for a set of icon documents."""

    def __init__(self, icons: Sequence[IconDoc], model_name: str, view: str, matrix: np.ndarray):
        self.icons = list(icons)
        self.names = [ic.name for ic in self.icons]
        self.model_name = model_name
        self.view = view
        self.matrix = matrix  # (n_icons, dim), L2-normalized

    @classmethod
    def build(
        cls,
        icons: Sequence[IconDoc],
        model_name: str,
        view: str = "document",
        include_examples: bool = True,
        use_cache: bool = True,
    ) -> "IconIndex":
        icons = list(icons)
        names = [ic.name for ic in icons]
        view_key = f"{view}{'' if include_examples else '_noex'}"
        path = _cache_path(model_name, view_key, names)
        if use_cache and path.exists():
            try:
                data = np.load(path)
                if list(data["names"]) == names and str(data["model"]) == model_name:
                    return cls(icons, model_name, view, data["matrix"])
            except Exception:
                pass

        texts = [_doc_text(ic, view, include_examples) for ic in icons]
        model = get_model(model_name)
        prefix = config.doc_prefix_for(model_name)
        payload = [prefix + t for t in texts] if prefix else texts
        emb = model.encode(payload, convert_to_numpy=True, show_progress_bar=True)
        matrix = _normalize(np.asarray(emb, dtype=np.float32))

        if use_cache:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, names=np.array(names), model=model_name, matrix=matrix)
        return cls(icons, model_name, view, matrix)

    def cosine(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine of one normalized query vector against all icons → (n_icons,)."""
        return self.matrix @ query_vec


def _doc_text(icon: IconDoc, view: str, include_examples: bool) -> str:
    if view == "document":
        return icon.document(include_examples=include_examples)
    # a single field view
    return icon.field_text(view)
