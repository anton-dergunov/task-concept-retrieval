"""The exported bundle reproduces the in-repo methods (tcr/bundle.py).

No model is downloaded: the ONNX encoder is replaced by a stub.
"""

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from tcr import bundle
from tcr.data import IconDoc
from tcr.methods.base import Matcher
from tcr.methods.bm25 import _tokenize
from tcr.methods.hybrid import HybridMatcher
from tcr.org_tasks import query_text


def _icon(name, concepts, intents, examples, reasoning, usefulness=0.8):
    return IconDoc(name=name, char="", popularity=0, usefulness=usefulness,
                   visual_concepts=concepts, task_intents=intents, example_tasks=examples,
                   poor_matches=[], reasoning=reasoning)


ICONS = [
    _icon("savings", ["piggy bank", "coins"], ["save money"], ["Save for a car", "Budget"],
          "A piggy bank for saving money."),
    _icon("dentistry", ["tooth"], ["dental care"], ["Book the dentist", "Floss"],
          "A tooth, for dentist visits."),
    _icon("grocery", ["shopping basket"], ["buy food"], ["Buy milk", "Shop for dinner"],
          "A basket of groceries to buy."),
]


class StubEncoder:
    """Deterministic pseudo-embeddings, so the fused-matrix algebra can be checked."""
    spec = {**bundle.ENCODER}

    def encode(self, texts, prefix=""):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(prefix + t)) % (2 ** 32))
            v = rng.normal(size=8).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def test_bm25_matrix_equals_rank_bm25():
    docs = [_tokenize(ic.document()) for ic in ICONS]
    vocab, indptr, indices, data = bundle.bm25_weights(docs)
    row = {t: i for i, t in enumerate(vocab)}
    ref = BM25Okapi(docs)
    for query in ["save money money", "buy food for dinner", "tooth", "a"]:
        got = np.zeros(len(ICONS))
        for tok in _tokenize(query):
            r = row.get(tok)
            if r is not None:
                got[indices[indptr[r]:indptr[r + 1]]] += data[indptr[r]:indptr[r + 1]]
        np.testing.assert_allclose(got, ref.get_scores(_tokenize(query)), rtol=1e-5, atol=1e-6)


def test_fused_dense_equals_weighted_cosines():
    enc = StubEncoder()
    weights = {"visual_concepts": 0.5, "task_intents": 1.0, "reasoning": 0.3}
    dense = bundle.fused_dense(ICONS, enc, weights)
    q = enc.encode(["query"])[0]
    want = sum(w * (enc.encode([ic.field_text(f) for ic in ICONS], prefix="passage: ") @ q)
               for f, w in weights.items()) / sum(weights.values())
    np.testing.assert_allclose(dense @ q, want, rtol=1e-5)


@pytest.mark.parametrize("task, inputs", [
    ({"title": "Read the *paper* [1/3]", "parents": ["Research"], "body": "Long *body*"},
     ["title"]),
    ({"title": "Read the paper", "parents": ["Big", "Research"], "body": ""},
     ["title", "parents"]),
])
def test_compose_query_matches_query_text(task, inputs):
    spec = {"inputs": inputs, "normalize": "org-plain", "body_chars": 400}
    want = query_text(bundle.normalize(task["title"], "org-plain"),
                      parents=task["parents"] if "parents" in inputs else ())
    assert bundle.compose_query(task, spec) == want


class _Fixed(Matcher):
    def __init__(self, names, scores, name):
        super().__init__(name=name)
        self._names, self._scores = names, np.array(scores, dtype=np.float32)

    def full_scores(self, query_text):
        return self._names, self._scores


def test_hybrid_ignores_a_member_that_matched_nothing():
    """All-zero BM25 scores must not vote (icon 0 would otherwise tie and win)."""
    names = ["10k", "grocery", "savings"]
    bm25 = _Fixed(names, [0.0, 0.0, 0.0], "B1")
    dense = _Fixed(names, [0.80, 0.86, 0.81], "M1")
    ranked, _ = HybridMatcher([bm25, dense]).rank_and_signal("Купить продукты")
    assert ranked[0][0] == "grocery"
