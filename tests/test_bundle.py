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


# --- F1 bundles -----------------------------------------------------------------

class _VecEncoder:
    """Returns a fixed unit vector per text (as OnnxEncoder.encode does)."""
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def encode(self, texts, prefix=""):
        return np.array([self.vec for _ in texts], dtype=np.float32)


SPEC = {"name": "stub", "query_prefix": "query: ", "max_length": 64,
        "model": {"url": "https://example.org/a/model_fp16.onnx", "sha256": "ab" * 32, "size": 1},
        "tokenizer": {"url": "https://example.org/a/tokenizer.json", "sha256": "cd" * 32, "size": 1}}


def test_f1_bundle_is_dense_only_and_reads_parents(tmp_path):
    names = ["savings", "dentistry", "grocery"]
    matrix = np.eye(3, 4, dtype=np.float32)
    (tmp_path / "bm25.npz").write_bytes(b"stale")
    m = bundle.export_f1(tmp_path, names, matrix, SPEC, threshold=0.5, gate={"min_prob": 0.2})
    assert not (tmp_path / "bm25.npz").exists()
    assert m["bm25"] is None and m["query"]["inputs"] == ["title", "parents"]
    assert m["tag"].startswith("F1-")
    assert not np.load(tmp_path / "prior.npy").any()
    got = bundle.BundleMatcher(tmp_path, encoder=_VecEncoder([0, 1, 0, 0]))
    assert got.gate.threshold == 0.5
    assert got.match_one("x").icon == "dentistry"
    assert got.query_for({"title": "Book *it*", "parents": ["Health"]}) == "Health: Book it"


def test_f1_tag_ignores_encoder_urls_and_gate_record(tmp_path):
    names, matrix = ["a", "b"], np.eye(2, dtype=np.float32)
    t1 = bundle.export_f1(tmp_path, names, matrix, SPEC, 0.5, {"min_prob": 0.2})["tag"]
    moved = {**SPEC, "model": {**SPEC["model"], "url": "https://example.org/b/model_fp16.onnx"}}
    t2 = bundle.export_f1(tmp_path, names, matrix, moved, 0.5, {"min_prob": 0.3})["tag"]
    other = {**SPEC, "model": {**SPEC["model"], "sha256": "ef" * 32}}
    t3 = bundle.export_f1(tmp_path, names, matrix, other, 0.5, {"min_prob": 0.2})["tag"]
    t4 = bundle.export_f1(tmp_path, names, matrix, SPEC, 0.6, {"min_prob": 0.2})["tag"]
    assert t1 == t2 and t1 != t3 and t1 != t4


def test_f1_tokenizer_ships_without_padding(tmp_path):
    """Training's padding in tokenizer.json would be mean-pooled into the query vector."""
    import json
    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    ckpt.mkdir()
    out.mkdir()
    (ckpt / "f1_meta.json").write_text(json.dumps({
        "run": "F1@t", "base_model": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ", "max_query_len": 64}))
    tok = {"version": "1.0", "padding": {"pad_to_multiple_of": 8}, "truncation": {"max_length": 512},
           "model": {"type": "Unigram"}}
    (ckpt / "tokenizer.json").write_text(json.dumps(tok))
    (out / "model_fp16.onnx").write_bytes(b"onnx")         # reused, not rebuilt
    spec = bundle.export_f1_onnx(ckpt, out, "fp16", hf_repo="u/r")
    shipped = json.loads((out / "tokenizer.json").read_text())
    assert shipped["padding"] is None and shipped["truncation"] is None
    assert shipped["model"] == tok["model"]
    assert spec["model"]["url"] == "https://huggingface.co/u/r/resolve/main/model_fp16.onnx"
    assert spec["max_length"] == 64
    again = bundle.export_f1_onnx(ckpt, out, "fp16", hf_repo="u/r")
    assert again["tokenizer"]["sha256"] == spec["tokenizer"]["sha256"]   # byte-stable
