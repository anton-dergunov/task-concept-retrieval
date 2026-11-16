#!/usr/bin/env python3
"""Encode every glyph image with the image towers of method I1 (docs/next-label-free-work.md
step 3): zero-shot text -> glyph-image retrieval. Standalone (torch, transformers, pillow),
shipped in the F1 tarball so it runs on Kaggle's GPU; the text side (the tasks) is
encoded where the eval sets live (tcr/methods/image.py).

    python i1_embed.py --data f1_data --out i1 [--models siglip2-base,siglip2-so400m,jina-clip-v2]

Reads f1_data/glyphs/<name>.png (256 px, black on white) for every icon in icons.jsonl;
writes <out>/<model>.npz: names, matrix (L2-normalized float32), model id. A model that
fails (e.g. remote code that does not load) is reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODELS = {   # short name -> HF id (kept in step with tcr/methods/image.py)
    "siglip2-base": "google/siglip2-base-patch16-256",
    "siglip2-so400m": "google/siglip2-so400m-patch16-256",
    "jina-clip-v2": "jinaai/jina-clip-v2",
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def features(out) -> torch.Tensor:
    """get_*_features returns a tensor (transformers 4) or an output object (5)."""
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def encode_siglip(hf_id: str, images: list, device: str, batch: int) -> np.ndarray:
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(images), batch):
            x = proc(images=images[s:s + batch], return_tensors="pt").to(device)
            out.append(F.normalize(features(model.get_image_features(**x)).float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def encode_jina(hf_id: str, images: list, device: str, batch: int) -> np.ndarray:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(device).eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(images), batch):
            v = model.encode_image(images[s:s + batch])       # numpy by default
            v = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))
            out.append(F.normalize(v.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="the unpacked f1_data directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    from PIL import Image

    root = Path(a.data)
    names = [json.loads(l)["name"] for l in (root / "icons.jsonl").read_text().splitlines() if l.strip()]
    images = [Image.open(root / "glyphs" / f"{n}.png").convert("RGB") for n in names]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"{len(images)} glyphs, device {device}")
    for short in a.models.split(","):
        dst = out / f"{short}.npz"
        if dst.exists():
            log(f"{short}: done already")
            continue
        hf_id = MODELS[short]
        t0 = time.time()
        try:
            fn = encode_jina if short.startswith("jina") else encode_siglip
            mat = fn(hf_id, images, device, a.batch)
        except Exception as e:          # one tower failing must not stop the others
            log(f"{short}: FAILED ({type(e).__name__}: {e})")
            continue
        np.savez_compressed(dst, names=np.array(names), matrix=mat.astype(np.float32), model=hf_id)
        log(f"{short}: {mat.shape} in {time.time() - t0:.0f} s -> {dst}")
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
