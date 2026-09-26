#!/usr/bin/env python3
"""Download the public corpora the Public-* datasets are sampled from.

  data/raw/MS-LaTTE.json             10,101 real to-do titles (Wunderlist; MIT repo,
                                     paper: permissive CDLA) — github.com/microsoft/MS-LaTTE
  data/raw/massive/<locale>.jsonl    MASSIVE 1.1, en-US / es-ES / ru-RU only (CC BY 4.0)

Idempotent: skips files that are already present.
Run: python scripts/datasets/fetch_public.py
"""

from __future__ import annotations

import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402

MSLATTE_URL = "https://raw.githubusercontent.com/microsoft/MS-LaTTE/main/MS-LaTTE.json"
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
LOCALES = ["en-US", "es-ES", "ru-RU"]


def download(url: str, dest: Path) -> None:
    print(f"GET {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    mslatte = config.RAW_DIR / "MS-LaTTE.json"
    if not mslatte.exists():
        download(MSLATTE_URL, mslatte)
    print(f"ok {mslatte}")

    massive_dir = config.RAW_DIR / "massive"
    wanted = {f"{loc}.jsonl" for loc in LOCALES}
    if not all((massive_dir / w).exists() for w in wanted):
        massive_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tgz = Path(tmp) / "massive.tar.gz"
            download(MASSIVE_URL, tgz)
            with tarfile.open(tgz) as tar:
                for m in tar.getmembers():
                    name = Path(m.name).name
                    if m.isfile() and name in wanted:
                        with tar.extractfile(m) as src:
                            (massive_dir / name).write_bytes(src.read())
    for w in sorted(wanted):
        print(f"ok {massive_dir / w}")


if __name__ == "__main__":
    main()
