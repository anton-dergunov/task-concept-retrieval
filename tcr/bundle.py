"""Deployable matcher bundles: export a method as standalone files, and score with one.

The agenda (agentic-org-planner) runs a matcher without this repository, without
torch and without sentence-transformers: only onnxruntime + tokenizers + numpy.
A *bundle* is everything that runner needs, precomputed here:

  manifest.json  what the files are, how to build the query text, gate/fusion
                 settings, and where to download the ONNX encoder (URL + sha256)
  names.txt      icon names, one per line; row i of every matrix is icon i
  dense.npy      fused dense matrix D (fp16). M1's score
                   sum_f w_f * cos(q, e_f) / sum_f w_f
                 is linear in the normalized query q, so it collapses to D @ q
  prior.npy      additive quality prior (QUALITY_PRIOR_WEIGHT * usefulness)
  bm25.npz       B1 as a sparse vocab x icons weight matrix (CSR). Okapi BM25's
                 per-(term, icon) weight does not depend on the query, so a
                 query's raw score is the sum of the rows of its tokens

The icon side is encoded with the *same* ONNX encoder the runner uses, so both
sides share one vector space. `BundleMatcher` scores through the same
`HybridMatcher` code as M3, so the deployed artifact can be compared with M3
directly (`scripts/export_bundle.py --check`).

The runner in agentic-org-planner (`scripts/org_task_icon_matcher.py`) mirrors
`OnnxEncoder`, `compose_query` and the scoring here; keep them in step.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .data import IconDoc
from .methods.base import Matcher
from .methods.bm25 import BM25Matcher, _tokenize
from .methods.hybrid import HybridMatcher
from .org_query import normalize_text

BUNDLE_FORMAT = 1

# The encoder the bundle is built for: multilingual-e5-small (MIT), exported to
# ONNX by Xenova. Pinned to a revision; files verified by sha256.
#
# fp16, not the int8 export: on the Realistic tasks the bundle agrees with M3 on
# 99.8% of top-1 picks with the fp16 (and fp32) encoder, but on only 80% with
# int8. e5 packs icon scores tightly (median top-1/top-2 margin 0.003), and int8
# noise (query cosine 0.996 to fp32) is enough to reorder those near-ties. fp16
# costs 235 MB against int8's 118 MB and fp32's 470 MB.
_HF = "https://huggingface.co/Xenova/multilingual-e5-small/resolve/761b726dd34fb83930e26aab4e9ac3899aa1fa78"
ENCODER = {
    "name": "intfloat/multilingual-e5-small (Xenova ONNX, fp16)",
    "model": {
        "url": f"{_HF}/onnx/model_fp16.onnx",
        "sha256": "0e0fe349c99ea21c6f3aa273af21f7fb753c1e1174ef1647032029c2be3251c3",
        "size": 235336732,
    },
    "tokenizer": {
        "url": f"{_HF}/tokenizer.json",
        "sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        "size": 17082730,
    },
    "query_prefix": "query: ",
    "doc_prefix": "passage: ",
    "max_length": 512,
}

# BM25 parameters: rank_bm25.BM25Okapi defaults, and B1's saturation constant.
BM25_K1, BM25_B, BM25_EPSILON = 1.5, 0.75, 0.25
TOKEN_PATTERN = r"[^\W_]+"   # must equal methods/bm25.py's _TOKEN_RE


# --- Encoder ------------------------------------------------------------------

def fetch(spec: dict, cache_dir: Path) -> Path:
    """Return the local copy of SPEC ({url, sha256}), downloading it once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{spec['sha256'][:16]}-{spec['url'].rsplit('/', 1)[-1]}"
    if path.exists() and _sha256(path) == spec["sha256"]:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {spec['url']} ...", flush=True)
    urllib.request.urlretrieve(spec["url"], tmp)
    got = _sha256(tmp)
    if got != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {spec['url']}: {got}")
    tmp.replace(path)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class OnnxEncoder:
    """Sentence encoder: tokenizer -> ONNX transformer -> mean pool -> L2."""

    def __init__(self, spec: dict = ENCODER, cache_dir: Optional[Path] = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        cache_dir = cache_dir or (config.CACHE_DIR / "onnx")
        self.spec = spec
        self.tokenizer = Tokenizer.from_file(str(fetch(spec["tokenizer"], cache_dir)))
        self.tokenizer.enable_truncation(spec["max_length"])
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # Basic graph optimizations only: onnxruntime 1.27's layer-norm fusion
        # fails to load the fp16 encoder (fixed by 1.30), and the exporter and
        # the runner must run the same graph to give the same answers.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(str(fetch(spec["model"], cache_dir)), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: Sequence[str], prefix: str = "") -> np.ndarray:
        """One L2-normalized vector per text.

        Texts are run one at a time, never batched, so a text's vector cannot
        depend on its batch mates. With an int8 export it would: activations are
        quantized with a scale taken over the whole input tensor (measured: cosine
        0.996 to itself, as large as the int8-vs-fp32 difference)."""
        out = []
        for t in texts:
            enc = self.tokenizer.encode(prefix + t)
            ids = np.array([enc.ids], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            pooled = self.session.run(None, feeds)[0][0].mean(axis=0)   # (seq, dim) -> (dim,)
            out.append(pooled / max(float(np.linalg.norm(pooled)), 1e-12))
        return np.array(out, dtype=np.float32)


# --- Query text -----------------------------------------------------------------

_STATS_COOKIE_RE = re.compile(r"\s*\[\d*(?:/\d*|%)\]")


def normalize(text: str, mode: str) -> str:
    """The named normalizations a manifest may ask for.

    "org-plain": drop statistics cookies ([1/3], [50%]), then `normalize_text`
    (links -> readable text, emphasis markers, timestamps, planning keywords).
    "none": the text as is, org markup included.
    """
    if mode == "none":
        return text.strip()
    if mode == "org-plain":
        return normalize_text(_STATS_COOKIE_RE.sub("", text))
    raise ValueError(f"unknown normalization {mode!r}")


def compose_query(task: dict, query_spec: dict) -> str:
    """Build the matcher's query text from a task record, per the manifest.

    Same shape as `org_tasks.query_text`: the title, prefixed by the nearest parent
    heading when "parents" is an input, followed by a body prefix when "body" is.
    """
    inputs, mode = query_spec["inputs"], query_spec["normalize"]
    text = normalize(task.get("title") or "", mode)
    parents = task.get("parents") or []
    if "parents" in inputs and parents:
        parent = normalize(parents[-1], mode)
        text = f"{parent}: {text}" if parent else text
    body = task.get("body") or ""
    if "body" in inputs and body:
        body = normalize(body.replace("\n", " "), mode)[:query_spec["body_chars"]]
        text = f"{text}. {body}" if text else body
    return text


# --- Export ---------------------------------------------------------------------

def bm25_weights(docs: Sequence[List[str]]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """rank_bm25.BM25Okapi as a CSR matrix: (vocab, indptr, indices, data).

    Row t holds weight(t, d) = idf(t) * tf(k1+1) / (tf + k1(1 - b + b*dl/avgdl))
    for every icon d containing t, so get_scores(query) == sum of the query's rows
    (a token repeated in the query counts once per occurrence, as in rank_bm25).
    """
    n = len(docs)
    freqs = [Counter(d) for d in docs]
    doc_len = np.array([len(d) for d in docs], dtype=np.float64)
    avgdl = doc_len.sum() / n
    df = Counter(t for f in freqs for t in f)
    idf = {t: np.log(n - c + 0.5) - np.log(c + 0.5) for t, c in df.items()}
    eps = BM25_EPSILON * (sum(idf.values()) / len(idf))
    idf = {t: (eps if v < 0 else v) for t, v in idf.items()}

    vocab = sorted(df)
    rows: Dict[str, List[Tuple[int, float]]] = {t: [] for t in vocab}
    norm = BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avgdl)
    for d, f in enumerate(freqs):
        for t, tf in f.items():
            rows[t].append((d, idf[t] * tf * (BM25_K1 + 1) / (tf + norm[d])))
    indptr, indices, data = [0], [], []
    for t in vocab:
        for d, w in rows[t]:
            indices.append(d)
            data.append(w)
        indptr.append(len(indices))
    return (vocab, np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32),
            np.array(data, dtype=np.float32))


def fused_dense(icons: Sequence[IconDoc], encoder: OnnxEncoder,
                field_weights: Dict[str, float]) -> np.ndarray:
    """D = sum_f (w_f / sum w) * E_f, each E_f the row-normalized field embeddings."""
    total = sum(field_weights.values())
    dense = None
    for f, w in field_weights.items():
        emb = encoder.encode([ic.field_text(f) for ic in icons], prefix=encoder.spec["doc_prefix"])
        dense = emb * (w / total) if dense is None else dense + emb * (w / total)
    return dense


def export(out_dir: Path, icons: Sequence[IconDoc], encoder: OnnxEncoder,
           method: str = "M3", source: str = "") -> dict:
    """Write a bundle for METHOD (M3: B1 + M1 fused by RRF; M1: dense only)."""
    if method not in ("M1", "M3"):
        raise ValueError("only M1 and M3 can be bundled")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [ic.name for ic in icons]
    dense = fused_dense(icons, encoder, config.FIELD_WEIGHTS).astype(np.float16)
    prior = (config.QUALITY_PRIOR_WEIGHT *
             np.array([ic.usefulness for ic in icons], dtype=np.float32))
    (out_dir / "names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    np.save(out_dir / "dense.npy", dense)
    np.save(out_dir / "prior.npy", prior)
    files = ["names.txt", "dense.npy", "prior.npy"]
    if method == "M3":
        vocab, indptr, indices, data = bm25_weights(
            [_tokenize(ic.document(include_examples=True)) for ic in icons])
        np.savez_compressed(out_dir / "bm25.npz", vocab=np.array(vocab), indptr=indptr,
                            indices=indices, data=data)
        files.append("bm25.npz")
    elif (out_dir / "bm25.npz").exists():
        (out_dir / "bm25.npz").unlink()

    manifest = {
        "format": BUNDLE_FORMAT,
        "method": method,
        "source": source,
        "icons": "names.txt",
        "encoder": {k: v for k, v in encoder.spec.items() if k != "doc_prefix"},
        "query": {"inputs": ["title"], "normalize": "org-plain", "body_chars": 400},
        "dense": {"matrix": "dense.npy", "prior": "prior.npy"},
        "bm25": ({"matrix": "bm25.npz", "token_pattern": TOKEN_PATTERN,
                  "saturation": BM25Matcher.SAT_K} if method == "M3" else None),
        "fusion": {"rrf_k": config.RRF_K, "pool": 50, "threshold": config.ABSTAIN_THRESHOLD},
    }
    # Cache identity: a consumer that caches answers drops them when this changes.
    # It covers everything that decides an answer: the matrices and every setting
    # (encoder, query, fusion), but not `source`, which changes with every commit.
    digest = hashlib.sha256()
    for f in files:
        digest.update((out_dir / f).read_bytes())
    digest.update(json.dumps({k: v for k, v in manifest.items() if k != "source"},
                             sort_keys=True).encode())
    manifest = {"tag": f"{method}-{digest.hexdigest()[:12]}", **manifest}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# --- Scoring with a bundle (the runner's logic, as Matchers) ----------------------

class _DenseMember(Matcher):
    def __init__(self, names, dense, prior, encoder, prefix, name="M1"):
        super().__init__(name=name)
        self.names, self.dense, self.prior = names, dense.astype(np.float32), prior
        self.encoder, self.prefix = encoder, prefix

    def full_scores(self, query_text: str):
        q = self.encoder.encode([query_text], prefix=self.prefix)[0]
        return self.names, self.dense @ q + self.prior


class _BM25Member(Matcher):
    def __init__(self, names, bm25: dict, saturation: float, name="B1"):
        super().__init__(name=name)
        self.names = names
        self.row = {t: i for i, t in enumerate(bm25["vocab"].tolist())}
        self.indptr, self.indices, self.data = bm25["indptr"], bm25["indices"], bm25["data"]
        self.saturation = saturation

    def full_scores(self, query_text: str):
        raw = np.zeros(len(self.names), dtype=np.float32)
        for tok in _tokenize(query_text):
            r = self.row.get(tok)
            if r is not None:
                lo, hi = self.indptr[r], self.indptr[r + 1]
                raw[self.indices[lo:hi]] += self.data[lo:hi]
        scores = raw / (raw + self.saturation)
        scores[raw <= 0] = 0.0
        return self.names, scores


class BundleMatcher(Matcher):
    """Score with an exported bundle, exactly as the agenda's runner does."""

    def __init__(self, bundle_dir: Path, encoder: Optional[OnnxEncoder] = None, name: str = "M3x"):
        super().__init__(name=name)
        bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        names = (bundle_dir / self.manifest["icons"]).read_text(encoding="utf-8").split()
        enc = encoder or OnnxEncoder({**ENCODER, **self.manifest["encoder"]})
        dense = _DenseMember(names, np.load(bundle_dir / self.manifest["dense"]["matrix"]),
                             np.load(bundle_dir / self.manifest["dense"]["prior"]),
                             enc, self.manifest["encoder"]["query_prefix"])
        members: List[Matcher] = []
        if self.manifest.get("bm25"):
            with np.load(bundle_dir / self.manifest["bm25"]["matrix"]) as z:
                members.append(_BM25Member(names, dict(z), self.manifest["bm25"]["saturation"]))
        members.append(dense)
        fusion = self.manifest["fusion"]
        self.gate.threshold = fusion["threshold"]
        self._inner = (HybridMatcher(members, rrf_k=fusion["rrf_k"], name=name)
                       if len(members) > 1 else dense)

    def rank_and_signal(self, query_text: str, top_k: int = 10):
        return self._inner.rank_and_signal(query_text, top_k=top_k)

    def query_for(self, task: dict) -> str:
        return compose_query(task, self.manifest["query"])
