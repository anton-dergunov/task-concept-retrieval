"""I1: zero-shot text -> glyph-image retrieval with a CLIP-style model's two towers
(docs/next-label-free-work.md step 3), and its fusion with a text method.

The glyph vectors are computed once, on a GPU, by scripts/i1_embed.py (Kaggle) and
imported to .cache/i1/<model>.npz; only the task text is encoded here, with the model's
text tower, and cached per text in .cache/i1/<model>.queries.npz so reruns and weight
sweeps cost nothing. The icon's name is never used: the image is rendered from its
codepoint, and the name is only the id that joins the two sides (hard rule 1).

Method keys (tcr.methods.build_method):
  I1-sb, I1-so, I1-jc     SigLIP 2 base / SigLIP 2 so400m / jina-clip-v2, image only
  <text>+<I1 key>~<w>     fusion, e.g. F1@all+I1-sb~0.2: per query, each method's scores
                          are standardized over the icons, then (1 - w)·text + w·image
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from .base import Matcher

I1_DIR = config.CACHE_DIR / "i1"
MODELS = {   # key -> (short name of scripts/i1_embed.py, HF id)
    "I1-sb": ("siglip2-base", "google/siglip2-base-patch16-256"),
    "I1-so": ("siglip2-so400m", "google/siglip2-so400m-patch16-256"),
    "I1-jc": ("jina-clip-v2", "jinaai/jina-clip-v2"),
}


def _features(out):
    import torch
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class _TextTower:
    """The text side of a CLIP-style model, on CPU (few threads: the laptop stays usable)."""

    def __init__(self, short: str, hf_id: str):
        import torch
        torch.set_num_threads(int(os.environ.get("TCR_I1_THREADS", "2")))
        self.torch, self.jina = torch, short.startswith("jina")
        if self.jina:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).eval()
        else:
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(hf_id)
            self.model = AutoModel.from_pretrained(hf_id).eval()

    def encode(self, texts: Sequence[str], batch: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), batch):
                chunk = list(texts[s:s + batch])
                if self.jina:
                    v = self.model.encode_text(chunk, task="retrieval.query")
                    v = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))
                else:   # SigLIP is trained on text padded to 64 tokens: keep that
                    x = self.tok(chunk, padding="max_length", max_length=64, truncation=True,
                                 return_tensors="pt")
                    v = _features(self.model.get_text_features(**x))
                out.append(torch.nn.functional.normalize(v.float(), dim=-1).numpy())
        return np.concatenate(out)


class ImageTowerMatcher(Matcher):
    def __init__(self, icons, key: str, **kw):
        super().__init__(name=key, **kw)
        self.short, self.hf_id = MODELS[key]
        path = I1_DIR / f"{self.short}.npz"
        if not path.exists():
            raise FileNotFoundError(f"no glyph vectors at {path}: run scripts/i1_embed.py on "
                                    "Kaggle and import them (docs/f1-kaggle-run.md)")
        d = np.load(path)
        mat, row = d["matrix"], {str(n): i for i, n in enumerate(d["names"])}
        want = [ic.name for ic in icons]
        missing = [n for n in want if n not in row]
        if missing:
            raise ValueError(f"{len(missing)} icons have no glyph vector, e.g. {missing[:3]}")
        self.names = want
        self.matrix = mat[[row[n] for n in want]]
        self._tower: Optional[_TextTower] = None
        self._qpath = I1_DIR / f"{self.short}.queries.npz"
        self._q: Dict[str, np.ndarray] = {}
        if self._qpath.exists():
            q = np.load(self._qpath)
            self._q = {str(h): v for h, v in zip(q["sha1"], q["matrix"])}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def prepare(self, texts: Sequence[str]) -> None:
        """Encode every text not cached yet, in batches, and save the cache."""
        todo = sorted({t for t in texts if self._key(t) not in self._q})
        if not todo:
            return
        if self._tower is None:
            self._tower = _TextTower(self.short, self.hf_id)
        for t, v in zip(todo, self._tower.encode(todo)):
            self._q[self._key(t)] = v
        keys = sorted(self._q)
        np.savez_compressed(self._qpath, sha1=np.array(keys), matrix=np.stack([self._q[k] for k in keys]))

    def full_scores(self, query_text: str):
        k = self._key(query_text)
        if k not in self._q:
            self.prepare([query_text])
        return self.names, self.matrix @ self._q[k]


def _z(x: np.ndarray) -> np.ndarray:
    s = float(x.std())
    return (x - float(x.mean())) / s if s > 1e-9 else x * 0.0


class FusionMatcher(Matcher):
    """(1 - w) · z(text scores) + w · z(image scores), over the text method's icons."""

    def __init__(self, text: Matcher, image: ImageTowerMatcher, w: float, name: str, **kw):
        super().__init__(name=name, **kw)
        self.text, self.image, self.w = text, image, w
        self._map: Optional[Tuple[List[str], np.ndarray]] = None

    def prepare(self, texts: Sequence[str]) -> None:
        for m in (self.text, self.image):
            if hasattr(m, "prepare"):
                m.prepare(texts)

    def full_scores(self, query_text: str):
        names, ts = self.text.full_scores(query_text)
        inames, iscores = self.image.full_scores(query_text)
        if self._map is None or self._map[0] is not names:
            pos = {n: i for i, n in enumerate(inames)}
            self._map = (names, np.array([pos[n] for n in names]))
        return names, (1 - self.w) * _z(np.asarray(ts)) + self.w * _z(iscores[self._map[1]])
