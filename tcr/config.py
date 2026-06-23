"""Central configuration: paths, model names, and method defaults.

Everything tunable lives here so experiments change one place. Environment
variables override the common knobs for quick sweeps.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ICON_DESC_DIR = DATA_DIR / "icon_descriptions"
ICON_PNG_DIR = DATA_DIR / "icons"
CATALOG_PATH = DATA_DIR / "material_symbols_catalog.json"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache" / "tcr"  # embedding caches (gitignored)

# --- Models ------------------------------------------------------------------
# Default encoder is multilingual: tasks may be English, Spanish, or Russian.
MULTILINGUAL_MODEL = os.environ.get("TCR_MODEL", "intfloat/multilingual-e5-small")
# English-only model used by the B0 baseline (mirrors the current emoji matcher).
ENGLISH_MODEL = os.environ.get("TCR_EN_MODEL", "BAAI/bge-small-en-v1.5")

# --- Quality gating ----------------------------------------------------------
# Icons with discard=true are dropped entirely. icon_usefulness (0-10) becomes a
# prior in [0,1] that down-weights weak icons in scoring.
QUALITY_PRIOR_WEIGHT = float(os.environ.get("TCR_QUALITY_WEIGHT", "0.05"))

# --- Field weights for the field-weighted bi-encoder (M1) --------------------
# Keys must match IconDoc field-view names.
FIELD_WEIGHTS = {
    "visual_concepts": 0.5,
    "task_intents": 1.0,
    "example_tasks": 1.0,
    "reasoning": 0.3,
}

# --- Abstention --------------------------------------------------------------
# Minimum *standardized confidence* (z-score: how many std-devs the top icon
# stands above the query's mean similarity to the corpus) required to show an
# icon. This is encoder-robust where absolute cosine is not. Heuristic default
# pending proper calibration (see design docs); the eval reports threshold-
# independent metrics (AUROC, precision-coverage) that don't depend on it.
ABSTAIN_THRESHOLD = float(os.environ.get("TCR_ABSTAIN_THRESHOLD", "3.6"))

# --- Retrieval defaults ------------------------------------------------------
DEFAULT_TOP_K = 10
RRF_K = 60  # reciprocal-rank-fusion constant

# --- Query-instruction prefixes (model-family specific) ----------------------
# E5 models want "query: " / "passage: "; BGE wants a retrieval instruction.
QUERY_PREFIXES = {
    "e5": "query: ",
    "bge": "Represent this sentence for searching relevant passages: ",
}
DOC_PREFIXES = {
    "e5": "passage: ",
}


def query_prefix_for(model_name: str) -> str:
    name = model_name.lower()
    for key, prefix in QUERY_PREFIXES.items():
        if key in name:
            return prefix
    return ""


def doc_prefix_for(model_name: str) -> str:
    name = model_name.lower()
    for key, prefix in DOC_PREFIXES.items():
        if key in name:
            return prefix
    return ""


# Server
SERVER_HOST = os.environ.get("TCR_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("TCR_PORT", "8765"))
