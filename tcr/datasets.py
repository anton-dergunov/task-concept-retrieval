"""Registry and I/O for the task datasets (shared JSONL schema).

Record: {"id", "dataset", "title", "body", "lang", "meta": {...}}. Title and
body keep org markup; `meta` holds everything that must never be shown or
matched on (tags, provenance, translation group, original concept).
"""

from __future__ import annotations

import hashlib
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
    # Released by `synth_personal.py publish` after its privacy audit and review;
    # the unreviewed candidates live in data/private/personal_synth.jsonl.
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


SPLIT_SEED = "split-v1"
# Tasks used to develop the LLM judge (its prompt examples and error analysis): neither
# dev nor test for matcher evaluation. Private ids, so the list lives in data/private/.
JUDGE_DEV_IDS = config.PRIVATE_DIR / "judge" / "judge_dev_ids.txt"
_judge_dev = None


def judge_dev_ids() -> frozenset:
    global _judge_dev
    if _judge_dev is None:
        _judge_dev = frozenset(
            line.strip() for line in (JUDGE_DEV_IDS.read_text().splitlines()
                                      if JUDGE_DEV_IDS.exists() else [])
            if line.strip() and not line.startswith("#"))
    return _judge_dev


def split_of(task_id: str) -> str:
    """"dev" / "test" / "judge-dev" (docs/fine-tuning-plan.md §2).

    "judge-dev": the tasks the LLM judge was developed on (JUDGE_DEV_IDS); they never
    count for matcher evaluation. Every other task is split 50/50 by id hash,
    independent of the labelling order (which uses a different hash), so any labelled
    prefix is split evenly. "test" is touched only for final numbers; "dev" is for
    tuning, gate calibration and judge calibration.
    """
    if task_id in judge_dev_ids():
        return "judge-dev"
    h = int(hashlib.sha1((SPLIT_SEED + task_id).encode("utf-8")).hexdigest()[:8], 16)
    return "test" if h % 2 == 0 else "dev"
