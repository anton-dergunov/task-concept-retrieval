"""Leakage gates and audit metrics for synthetic data derived from private tasks.

Per-item gates (a synthetic item is rejected if any fires):
  exact      normalized title equals a private title
  ngram      shares a word 5-gram with a private task that is absent from public text
             and contains >= 2 uncommon words (outside the top-5k English words) —
             generic phrasing ("I should be able to") is not a leak
  url        contains a URL that appears in the private tasks
  rare       contains a token that is rare in the private tasks (df <= RARE_DF) and
             absent from general English (wordfreq, ~320k words) and every public
             reference corpus: names, codenames, places
  nn         embedding cosine to the nearest private task above τ, where τ is the
             given quantile of the private-holdout → private-train nearest-neighbour
             similarity (a DCR-style test: closer than real tasks are to each other)
  pii        Presidio / regex hit on contact or account identifiers
Presidio needs:  python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

URL_RE = re.compile(r"https?://\S+")
WORD_RE = re.compile(r"[^\W_]+")
RARE_DF = 3
COMMON_N = 5000            # words this common never make a shared n-gram "specific"
HARD_PII = {"EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS",
            "US_SSN", "UK_NHS", "CRYPTO", "MEDICAL_LICENSE"}
SOFT_PII = {"PERSON", "LOCATION", "NRP"}   # reported, not rejected: fiction invents these


def words(text: str) -> List[str]:
    return WORD_RE.findall(URL_RE.sub(" ", text).casefold())


def urls(text: str) -> set:
    return {u.rstrip(".,;)]") for u in URL_RE.findall(text)}


def ngrams(text: str, n: int = 5) -> set:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def english_vocab(n: int) -> set:
    from wordfreq import top_n_list
    return set(top_n_list("en", n, wordlist="large"))


def norm_title(t: str) -> str:
    return " ".join(words(t))


def _text(r: dict) -> str:
    return r["title"] + "\n" + r.get("body", "")


class LeakageGate:
    def __init__(self, private: Sequence[dict], public_texts: Iterable[str],
                 private_emb: Optional[np.ndarray] = None, nn_quantile: float = 0.95,
                 seed: int = 0):
        self.titles = {norm_title(r["title"]) for r in private}
        self.grams = set()
        self.urls = set()
        df = Counter()
        for r in private:
            t = _text(r)
            self.grams |= ngrams(t)
            self.urls |= urls(t)
            df.update(set(words(t)))
        public_vocab = english_vocab(500000)
        self.common = english_vocab(COMMON_N)
        public_grams = set()
        for t in public_texts:
            public_vocab.update(words(t))
            public_grams |= ngrams(t)
        self.rare = {w for w, c in df.items() if c <= RARE_DF and w not in public_vocab
                     and len(w) > 2 and not w.isdigit()}
        # Only distinctive private phrasing counts: not seen in public text, and carrying
        # at least two uncommon words.
        self.grams = {g for g in self.grams - public_grams
                      if sum(w not in self.common for w in g) >= 2}
        self.emb = private_emb
        self.tau = None
        if private_emb is not None:
            self.tau = holdout_nn_quantile(private_emb, nn_quantile, seed)
        self._presidio = None

    # -- PII -------------------------------------------------------------------
    def presidio(self):
        if self._presidio is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            nlp = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}).create_engine()
            self._presidio = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        return self._presidio

    def pii(self, text: str) -> Dict[str, List[str]]:
        hits: Dict[str, List[str]] = {}
        for r in self.presidio().analyze(URL_RE.sub(" ", text), language="en", score_threshold=0.5):
            hits.setdefault(r.entity_type, []).append(text[r.start:r.end])
        return hits

    # -- gates -----------------------------------------------------------------
    def check(self, rec: dict, emb: Optional[np.ndarray] = None) -> Dict[str, object]:
        """Return {gate: evidence} for every gate that fires; {} means the item passes."""
        t = _text(rec)
        fired: Dict[str, object] = {}
        if norm_title(rec["title"]) in self.titles:
            fired["exact"] = True
        shared = ngrams(t) & self.grams
        if shared:
            fired["ngram"] = len(shared)
        if urls(t) & self.urls:
            fired["url"] = sorted(urls(t) & self.urls)
        rare = sorted(set(words(t)) & self.rare)
        if rare:
            fired["rare"] = len(rare)          # count only: the tokens are private
        if emb is not None and self.emb is not None:
            sim = float((self.emb @ emb).max())
            if sim > self.tau:
                fired["nn"] = round(sim, 3)
        hard = {k: v for k, v in self.pii(t).items() if k in HARD_PII}
        if hard:
            fired["pii"] = sorted(hard)
        return fired


def nn_sims(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Max cosine of each query row against the corpus (rows L2-normalized)."""
    out = np.empty(len(queries), dtype=np.float32)
    for i in range(0, len(queries), 512):
        out[i:i + 512] = (queries[i:i + 512] @ corpus.T).max(axis=1)
    return out


def holdout_nn_quantile(emb: np.ndarray, q: float, seed: int = 0) -> float:
    """τ: quantile of holdout→train nearest-neighbour similarity within private data."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(emb))
    half = len(emb) // 2
    return float(np.quantile(nn_sims(emb[idx[:half]], emb[idx[half:]]), q))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
