"""Abstention gate: turn a ranked candidate list into "top-1 or None".

Because the agenda auto-shows the top-1, a wrong icon is worse than no icon.
The gate decides on a *standardized confidence* (how far the best icon stands
above how the query matches the corpus overall — see methods/base.py), which is
robust across encoders where absolute cosine is not. A `calibrate()` hook maps
that signal to P(acceptable) via logistic regression once labels exist.

See design/experimentation-strategy.md §7 and design/matching-methods.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import config

Ranked = List[Tuple[str, float]]  # [(icon_name, display_score), ...] sorted desc


@dataclass
class Decision:
    icon: Optional[str]      # chosen icon name, or None if abstaining
    score: float             # top-1 display score (e.g. cosine)
    confidence: float        # standardized gate signal
    shown: bool              # whether we show it
    reason: str              # why (for inspection/debugging)


class AbstentionGate:
    def __init__(self, threshold: float = config.ABSTAIN_THRESHOLD):
        # `threshold` is a minimum standardized confidence (z-score), unless a
        # calibrator is fitted, in which case it is a minimum probability.
        self.threshold = threshold
        self._calibrator = None

    def to_prob(self, confidence: float) -> Optional[float]:
        if self._calibrator is not None:
            return float(self._calibrator(confidence))
        return None

    def decide(self, ranked: Ranked, confidence: float) -> Decision:
        if not ranked:
            return Decision(None, 0.0, confidence, False, "no candidates")
        top_name, top_score = ranked[0]
        prob = self.to_prob(confidence)
        gauge = prob if prob is not None else confidence
        kind = "prob" if prob is not None else "confidence"
        shown = gauge >= self.threshold
        reason = ("shown" if shown
                  else f"{kind} {gauge:.3f} < threshold {self.threshold:.3f}")
        return Decision(top_name, top_score, confidence, shown, reason)

    def calibrate(self, confidences: List[float], labels: List[int]) -> None:
        """Fit confidence -> P(acceptable) (Platt scaling). After calibration,
        `threshold` is interpreted as a probability (set it in [0,1])."""
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression

            X = np.asarray(confidences, dtype=float).reshape(-1, 1)
            y = np.asarray(labels, dtype=int)
            if len(set(y.tolist())) < 2:
                return
            lr = LogisticRegression()
            lr.fit(X, y)
            self._calibrator = lambda c: float(lr.predict_proba([[c]])[0, 1])
        except Exception:
            self._calibrator = None
