"""Image embeddings of the rendered glyphs, and a look-alike ordering of them.

Purely visual: nothing here reads an icon's name, tags or description — the name
only locates the PNG. Used to lay out the icon review grid so that near-identical
glyphs sit together (scripts/build_curation_bundle.py), and meant for the
"contrast" pass of the v2 descriptions (docs/icon-descriptions-v2.md §5).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from . import config
from .index import _cache_path, _normalize

# Already a common local download; any CLIP/SigLIP checkpoint on the Hub works.
GLYPH_MODEL = "openai/clip-vit-base-patch32"


def glyph_embeddings(names: Sequence[str], model_name: str = GLYPH_MODEL,
                     use_cache: bool = True, batch_size: int = 64) -> np.ndarray:
    """L2-normalized image embedding of each icon's PNG, cached like the text indexes."""
    names = list(names)
    path = _cache_path(model_name, "glyph", names)
    if use_cache and path.exists():
        try:
            data = np.load(path)
            if list(data["names"]) == names and str(data["model"]) == model_name:
                return data["matrix"]
        except Exception:
            pass

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(names), batch_size):
            images = [Image.open(config.ICON_PNG_DIR / f"{n}.png").convert("RGB")
                      for n in names[i:i + batch_size]]
            out = model.get_image_features(**processor(images=images, return_tensors="pt"))
            if not torch.is_tensor(out):   # some versions return a model output
                out = out.pooler_output
            chunks.append(out.cpu().numpy())
    matrix = _normalize(np.concatenate(chunks).astype(np.float32))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array(names), model=model_name, matrix=matrix)
    return matrix


def lookalike_order(matrix: np.ndarray, n_groups: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Order the rows so that similar glyphs are neighbours, and cut the order into
    `n_groups` contiguous groups.

    Ward clustering (on unit vectors, squared Euclidean distance is 2·(1 − cosine))
    gives groups of comparable size; the dendrogram's leaf order, optimized so that
    adjacent leaves are as close as possible, is the display order. Returns
    (row indices in display order, [(start, size)] of each group in that order).
    """
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    dist = pdist(matrix, "euclidean")
    tree = optimal_leaf_ordering(linkage(dist, "ward"), dist)
    order = [int(i) for i in leaves_list(tree)]
    labels = fcluster(tree, n_groups, "maxclust")
    groups, start = [], 0
    for pos in range(1, len(order) + 1):
        # A cluster is a subtree, so its leaves are contiguous in the leaf order.
        if pos == len(order) or labels[order[pos]] != labels[order[start]]:
            groups.append((start, pos - start))
            start = pos
    return order, groups
