"""Matcher implementations and a registry/factory.

Methods (see design/matching-methods.md):
  B0 - bge-small-en concatenated bi-encoder (current-approach replica)
  B1 - BM25 lexical baseline
  M1 - multilingual field-weighted bi-encoder + quality prior
  M2 - nearest-example voting (task <-> example_tasks)
  M3 - hybrid: RRF of B1 + M1
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from ..data import IconDoc, load_icons
from .base import Matcher
from .bm25 import BM25Matcher
from .biencoder import BiEncoderMatcher
from .nearest_example import NearestExampleMatcher
from .hybrid import HybridMatcher
from .. import config

# Factories take (icons, include_examples) and return a built Matcher.
_BUILDERS: Dict[str, Callable] = {
    "B0": lambda icons, inc: BiEncoderMatcher(
        icons, model_name=config.ENGLISH_MODEL, view="document",
        field_weights=None, quality_weight=0.0, include_examples=inc, name="B0"),
    "B1": lambda icons, inc: BM25Matcher(icons, include_examples=inc, name="B1"),
    "M1": lambda icons, inc: BiEncoderMatcher(
        icons, model_name=config.MULTILINGUAL_MODEL, view="field",
        field_weights=config.FIELD_WEIGHTS, quality_weight=config.QUALITY_PRIOR_WEIGHT,
        include_examples=inc, name="M1"),
    "M2": lambda icons, inc: NearestExampleMatcher(
        icons, model_name=config.MULTILINGUAL_MODEL,
        quality_weight=config.QUALITY_PRIOR_WEIGHT, name="M2"),
    "M3": lambda icons, inc: HybridMatcher(
        [BM25Matcher(icons, include_examples=inc, name="B1"),
         BiEncoderMatcher(icons, model_name=config.MULTILINGUAL_MODEL, view="field",
                          field_weights=config.FIELD_WEIGHTS,
                          quality_weight=config.QUALITY_PRIOR_WEIGHT,
                          include_examples=inc, name="M1")],
        name="M3"),
}

ALL_METHODS = ["B0", "B1", "M1", "M2", "M3"]


def build_method(key: str, icons: Sequence[IconDoc] = None, include_examples: bool = True) -> Matcher:
    if key not in _BUILDERS:
        raise KeyError(f"unknown method {key!r}; choices: {ALL_METHODS}")
    if icons is None:
        icons = load_icons()
    return _BUILDERS[key](icons, include_examples)


__all__ = [
    "Matcher", "BM25Matcher", "BiEncoderMatcher", "NearestExampleMatcher",
    "HybridMatcher", "build_method", "ALL_METHODS",
]
