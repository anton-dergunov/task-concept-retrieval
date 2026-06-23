"""Shared search engine: "query -> top-k icons" with timing and the gate.

Used by the CLI and the web server now, and the iPad labelling tool later. A
small cache keeps built matchers warm across calls within a process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from . import data
from .data import IconDoc
from .methods import ALL_METHODS, build_method
from .methods.base import Matcher

_MATCHER_CACHE: Dict[str, Matcher] = {}


@dataclass
class Hit:
    name: str
    score: float
    char: str
    shown: bool  # whether this would pass the abstention gate as top-1


@dataclass
class SearchResult:
    query: str
    method: str
    latency_ms: float
    decision_icon: Optional[str]   # top-1 if shown, else None (abstained)
    shown: bool
    confidence: float              # standardized gate signal
    reason: str
    hits: List[Hit]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def get_matcher(method: str) -> Matcher:
    if method not in _MATCHER_CACHE:
        _MATCHER_CACHE[method] = build_method(method)
    return _MATCHER_CACHE[method]


def search(query: str, method: str = "M3", k: int = 10) -> SearchResult:
    matcher = get_matcher(method)
    by_name = data.icons_by_name()
    t0 = time.perf_counter()
    ranked, confidence = matcher.rank_and_signal(query, top_k=k)
    decision = matcher.gate.decide(ranked, confidence)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    hits: List[Hit] = []
    for i, (name, score) in enumerate(ranked):
        ic = by_name.get(name)
        hits.append(Hit(
            name=name,
            score=round(float(score), 4),
            char=ic.char if ic else "",
            shown=(i == 0 and decision.shown),
        ))
    return SearchResult(
        query=query,
        method=method,
        latency_ms=round(latency_ms, 2),
        decision_icon=(decision.icon if decision.shown else None),
        shown=decision.shown,
        confidence=round(float(confidence), 3),
        reason=decision.reason,
        hits=hits,
    )


def available_methods() -> List[str]:
    return list(ALL_METHODS)
