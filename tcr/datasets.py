"""Registry and I/O for the task datasets (shared JSONL schema).

Record: {"id", "dataset", "title", "body", "lang", "meta": {...}}. Title and
body keep org markup; `meta` holds everything that must never be shown or
matched on (tags, provenance, translation group, original concept).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from . import config


@dataclass(frozen=True)
class Dataset:
    key: str
    display: str       # the only thing the labelling UI shows about provenance
    path: Path
    id_prefix: str
    private: bool = False


DATASETS: Dict[str, Dataset] = {d.key: d for d in [
    Dataset("personal", "Personal", config.PRIVATE_DIR / "personal.jsonl", "per", private=True),
    Dataset("realistic", "Realistic", config.DATASETS_DIR / "realistic.jsonl", "rea"),
    Dataset("personal_synth", "Personal-synth", config.DATASETS_DIR / "personal_synth.jsonl", "syn"),
    Dataset("public_short", "Public-short", config.DATASETS_DIR / "public_short.jsonl", "pub"),
    Dataset("public_expanded", "Public-expanded", config.DATASETS_DIR / "public_expanded.jsonl", "exp"),
]}


def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load(key: str) -> List[dict]:
    return read_jsonl(DATASETS[key].path)


def available() -> List[Dataset]:
    """Datasets whose file exists locally, in registry order."""
    return [d for d in DATASETS.values() if d.path.exists()]
