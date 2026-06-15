#!/usr/bin/env python3
"""Build a Material Symbols catalog: name, codepoint, char, tags, categories, popularity."""
import json
import urllib.request
from pathlib import Path

META_URL = "https://fonts.google.com/metadata/icons?key=material_symbols&incomplete=1"
CODEPOINTS_URL = (
    "https://raw.githubusercontent.com/google/material-design-icons/master/"
    "variablefont/MaterialSymbolsOutlined%5BFILL%2CGRAD%2Copsz%2Cwght%5D.codepoints"
)
OUT = Path("data/material_symbols_catalog.json")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def load_metadata() -> dict:
    raw = fetch(META_URL)
    raw = raw[raw.index("{"):]          # strip the )]}' XSSI prefix
    return {ic["name"]: ic for ic in json.loads(raw)["icons"]}


def load_codepoints() -> dict:        # ground truth for the installable font
    pairs = (ln.split() for ln in fetch(CODEPOINTS_URL).splitlines() if ln.strip())
    return {name: int(cp, 16) for name, cp in pairs}


def main():
    meta, cps = load_metadata(), load_codepoints()
    rows, untagged = [], 0
    for name, cp in sorted(cps.items()):
        m = meta.get(name, {})
        if not m.get("tags"):
            untagged += 1
        rows.append({
            "name": name,
            "codepoint": f"{cp:04x}",
            "char": chr(cp),
            "tags": m.get("tags", []),
            "categories": m.get("categories", []),
            "popularity": m.get("popularity", 0),
        })
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} symbols, {untagged} without tags -> {OUT}")
    print(json.dumps(next(r for r in rows if r["name"] == "vaccines"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
