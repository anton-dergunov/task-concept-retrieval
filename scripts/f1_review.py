#!/usr/bin/env python3
"""A private HTML page for reviewing F1's picks by eye: held-out pool tasks (never trained
on, not in the author's labelling queue) with F1's top 5, the judge's graded icons and
M3.v2's top 5. Every icon carries what the judge thought of it, so F1's *unjudged* picks
(the gap between silver hit@1 and its upper bound) can be judged by eye.

    python scripts/f1_review.py [--model F1@all] [--n 50] [--open]

Output: data/private/f1/review.html (private: Personal-style pool text). Glyphs are linked
from data/icons/ by relative path.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tcr import silver  # noqa: E402
from tcr.data import load_icons  # noqa: E402
from tcr.methods import build_method  # noqa: E402
from judge_report import CSS  # noqa: E402

DATA = ROOT / "data" / "private" / "f1" / "f1_data"
OUT = ROOT / "data" / "private" / "f1" / "review.html"
ICON_REL = "../../icons"          # from data/private/f1/ to data/icons/
SHARE = {"pool-synth": 0.7, "pool-mslatte": 0.3}

EXTRA_CSS = """
.ic.g3{border-color:var(--hit);border-width:3px}.ic.g2{border-color:var(--hit)}
.ic.g1{border-color:#8fbf9f}.ic.amb{border-color:var(--warn);border-style:dashed}
.ic.neg{border-color:#c0392b}.ic.unj{border-color:var(--top);border-style:dotted;border-width:3px}
.ic.held::after{content:"H";position:absolute;bottom:-7px;left:-7px;font-size:9px;
background:var(--warn);color:#fff;border-radius:6px;padding:0 3px;line-height:13px}
.sum{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 0}
.k{display:inline-flex;align-items:center;gap:6px;margin-right:14px}
.k .ic{width:22px;height:22px}
.src{font-size:11px;color:var(--mut);margin-left:6px}
"""

KINDS = {  # css class, legend text
    "g3": "judge grade 3", "g2": "grade 2", "g1": "grade 1",
    "amb": "judge: relevant, left ungraded", "neg": "judge saw it, not relevant",
    "unj": "unjudged (the judge never saw it)",
}


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sample(n: int) -> list:
    rows = [q for q in read_jsonl(DATA / "queries.jsonl")
            if q["split"] == "val" and q["lang"] == "en" and q["source"] in SHARE]
    out = []
    for src, share in SHARE.items():
        mine = sorted((q for q in rows if q["source"] == src), key=lambda q: _h("f1-review:" + q["task_id"]))
        out += mine[:round(n * share)]
    return sorted(out, key=lambda q: _h("f1-review-order:" + q["task_id"]))


def kind(i: int, q: dict) -> str:
    g = q["grades"].get(str(i))
    if g:
        return f"g{g}"
    if i in q["ambiguous"]:
        return "amb"
    if i in q["hard_negs"]:
        return "neg"
    return "unj"


def icon(name: str, cls: str, badge: str, tip: str) -> str:
    return (f'<span class="ic {cls}" title="{html.escape(tip)}"><img src="{ICON_REL}/{name}.png" '
            f'alt="">{f"<b>{badge}</b>" if badge else ""}</span>')


def row(label: str, icons: str, note: str = "") -> str:
    return (f'<div class="row"><div class="lab">{label}</div><div class="icons">{icons}</div></div>'
            + (f'<div class="note">{note}</div>' if note else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="F1@all", help="F1 or F1@<tag> (.cache/models/F1-<tag>)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()

    names = [r["name"] for r in read_jsonl(DATA / "icons.jsonl")]
    held = {r["name"] for r in read_jsonl(DATA / "icons.jsonl") if r["holdout"]}
    idx = {n: i for i, n in enumerate(names)}
    icons = load_icons(version="v2")
    f1 = build_method(a.model, icons)
    m3 = build_method("M3", icons)
    tasks = sample(a.n)
    held_out_run = a.model != "F1"

    cards, f1_kinds, m3_kinds = [], {}, {}
    for q in tasks:
        text = q.get("text_parent") or q["text"]

        def ranked(m):
            out = []
            for n, _ in m.rank(text, top_k=25):
                c = silver.canonical(n)
                if c in idx and c not in out:
                    out.append(c)
                if len(out) == 5:
                    break
            return out

        def show(top, tally):
            parts = []
            for r, n in enumerate(top, 1):
                k = kind(idx[n], q)
                if r == 1:
                    tally[k] = tally.get(k, 0) + 1
                cls = k + (" held" if held_out_run and n in held else "")
                parts.append(icon(n, cls, str(r), f"{n}: {KINDS[k]}"))
            return "".join(parts)

        f1_top, m3_top = ranked(f1), ranked(m3)
        judge = sorted(q["grades"].items(), key=lambda kv: -kv[1])
        judge_html = "".join(icon(names[int(i)], f"g{g}", "", f"{names[int(i)]}: grade {g}")
                             for i, g in judge)
        parent = ""
        if q.get("text_parent"):
            parent = f'<div class="parent">{html.escape(q["text_parent"][:-len(q["text"])].rstrip(": "))}</div>'
        k1 = kind(idx[f1_top[0]], q) if f1_top else "unj"
        cards.append(
            f'<div class="card"><div class="title">{html.escape(q["text"])}'
            f'<span class="src">{q["source"]}</span></div>{parent}'
            + row(f"{a.model} top 5", show(f1_top, f1_kinds), "" if k1 != "unj" else
                  "F1's top-1 was never shown to the judge: is it good?")
            + row("judge (graded)", judge_html)
            + row("M3.v2 top 5", show(m3_top, m3_kinds)) + "</div>")

    def tally_line(label, t):
        n = sum(t.values())
        hit = t.get("g3", 0) + t.get("g2", 0)
        return (f"<tr><td>{label}</td><td>{hit}/{n}</td><td>{t.get('g1', 0)}</td><td>{t.get('amb', 0)}</td>"
                f"<td>{t.get('neg', 0)}</td><td>{t.get('unj', 0)}</td></tr>")

    legend = "".join(f'<span class="k"><span class="ic {k}"></span>{v}</span>' for k, v in KINDS.items())
    if held_out_run:
        legend += '<span class="k"><span class="ic held"></span>icon held out of training</span>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>F1 review</title>
<style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>F1 review: {len(tasks)} held-out pool tasks</h1>
<p class="lead">{a.model} (with the parent heading) against the LLM judge and M3.v2. The tasks are the
10% of the training pools held out for validation: never trained on, but used to pick the best
checkpoint, so slightly optimistic. ~70% Personal-style (synthetic), ~30% MS-LaTTE.
Hover an icon for its name and verdict. The judge graded only icons in its candidate pool;
a dotted blue border marks one it never saw, which silver counts as a miss.</p>
<p class="legend">{legend}</p>
<h2>Top-1 verdicts</h2>
<table><tr><th>method</th><th>grade ≥ 2</th><th>grade 1</th><th>relevant, ungraded</th>
<th>seen, not relevant</th><th>unjudged</th></tr>
{tally_line(a.model, f1_kinds)}{tally_line("M3.v2", m3_kinds)}</table>
<h2>Tasks</h2>
{''.join(cards)}
</main></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"-> {OUT}", file=sys.stderr)
    if a.open:
        subprocess.run(["open", str(OUT)])


if __name__ == "__main__":
    main()
