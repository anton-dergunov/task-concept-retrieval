#!/usr/bin/env python3
"""A local HTML report of the LLM judge's development: the author's picks next to the
judge's for every validation task, per prompt version and model, and how the metrics
converged. Private (it shows Personal tasks): data/private/judge/report.html.

Reads the runs (data/private/judge/runs/) and the scores of `judge_dev.py score`.
Glyphs are linked from data/icons/ by relative path, so the file stays small.

Run: python scripts/judge_report.py [--open]
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tcr import judge as J  # noqa: E402
from tcr.datasets import load  # noqa: E402
import judge_dev as D  # noqa: E402

OUT = J.JUDGE_DIR / "report.html"
ICON_REL = "../../icons"          # from data/private/judge/ to data/icons/

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e0;--card:#fff;--hit:#1f8f4e;
--top:#0b62d6;--warn:#b45309}
@media (prefers-color-scheme: dark){:root{--bg:#161618;--fg:#ececee;--mut:#9a9aa0;
--line:#2c2c30;--card:#1e1e21}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:32px 0 8px}
p.lead{color:var(--mut);margin:0 0 16px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left} th{color:var(--mut);font-weight:600}
tr.best td{font-weight:650}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.title{font-weight:650}.parent{color:var(--mut);font-size:13px}
.row{display:flex;align-items:flex-start;gap:10px;margin-top:8px}
.lab{width:130px;flex:none;font-size:12px;color:var(--mut);padding-top:10px}
.icons{display:flex;flex-wrap:wrap;gap:6px}
.ic{position:relative;width:44px;height:44px;border:2px solid var(--line);border-radius:8px;
background:#fff;display:flex;align-items:center;justify-content:center}
.ic img{width:34px;height:34px}
.ic.in{border-color:var(--hit)} .ic.u1{border-color:var(--top);border-width:3px}
.ic b{position:absolute;top:-7px;left:-7px;background:var(--fg);color:var(--bg);font-size:10px;
border-radius:8px;padding:0 4px;line-height:15px}
.ic i{position:absolute;bottom:1px;right:3px;font-style:normal;font-size:9px;letter-spacing:-1px;
color:#8a8a90}
.note{font-size:12.5px;color:var(--mut);margin:6px 0 0 140px}
.tag{font-size:11px;padding:1px 6px;border-radius:6px;border:1px solid var(--line);color:var(--mut)}
.tag.hit{color:var(--hit);border-color:var(--hit)}
.legend{font-size:13px;color:var(--mut)}
"""


def icon(name: str, cls: str = "", badge: str = "", tip: str = "", grade: int = 0) -> str:
    dots = f"<i>{'●' * grade}</i>" if grade else ""
    return (f'<span class="ic {cls}" title="{html.escape(tip)}"><img src="{ICON_REL}/{name}.png" '
            f'alt="">{f"<b>{badge}</b>" if badge else ""}{dots}</span>')


def runs_by_name() -> dict:
    out = {}
    for f in sorted((J.JUDGE_DIR / "runs").glob("*.jsonl")):
        if "smoke" in f.stem:
            continue
        out[f.stem] = {s["task_id"]: s for s in map(json.loads, f.read_text().splitlines()) if s}
    return out


def config_of(run: str) -> str:
    return run.replace("__rep1", "").replace("__rep2", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    summary = json.loads(D.SUMMARY.read_text())
    scores = json.loads((J.JUDGE_DIR / "scores.json").read_text())
    runs = runs_by_name()
    labels = D.human_labels()
    ids = D.validation_ids()
    recs = {r["id"]: r for r in load("personal")}

    # Convergence: the reps of a config averaged.
    cfgs = {}
    for name in runs:
        cfgs.setdefault(config_of(name), []).append(name)
    rows = []
    for cfg, names in cfgs.items():
        aggs = [summary[n] for n in names if n in summary]
        if not aggs:
            continue
        avg = lambda k: sum(a.get(k, 0) or 0 for a in aggs) / len(aggs)  # noqa: E731
        flips = ""
        if len(names) == 2 and all(n in scores for n in names):
            a, b = (scores[n] for n in names)
            common = [t for t in a if t in b and not a[t]["none_task"]]
            flips = f"{sum(a[t]['hit1'] != b[t]['hit1'] for t in common)}/{len(common)}"
        version, model = cfg.split("__")[0], cfg.split("__")[1]
        variant = cfg.split("__")[2] if cfg.count("__") >= 2 else ""
        rows.append((cfg, version, model, variant, len(names), avg("hit1"), avg("hit1_judged"),
                     avg("exact1"), avg("ndcg5"), avg("u1_collected"), flips, avg("usd")))
    rows.sort(key=lambda r: (r[2] != J.DEFAULT_MODEL, r[1], r[3]))
    base = [(n, summary[n]) for n in ("M3.v2", "M1.v2", "M1p.v2") if n in summary]

    h = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Judge development</title>"
         f"<style>{CSS}</style></head><body><main>",
         "<h1>LLM judge: development on your labels</h1>",
         f"<p class='lead'>{len(ids)} validation tasks (your first 51 labels minus the 5 used "
         "as examples in the prompt). hit@1: the judge's first icon is one of your picks; "
         "judged: the same, only where you saw that icon; exact@1: it is your first pick; "
         "nDCG@5: the judge's top 5 against your ranking; collected: your first pick made "
         "the judge's shortlist. Two runs per configuration (averaged); flips = tasks whose "
         "hit@1 differs between the two runs (the noise).</p>",
         "<h2>How it converged</h2><table><tr><th>configuration</th><th>runs</th><th>hit@1</th>"
         "<th>judged</th><th>exact@1</th><th>nDCG@5</th><th>collected</th><th>flips</th>"
         "<th>$ / task (online)</th></tr>"]
    best = max(rows, key=lambda r: r[8])[0] if rows else None
    for r in rows:
        cfg, version, model, variant = r[:4]
        label = f"{version} · {model.replace('gemini-', '')}" + (f" · {variant}" if variant else "")
        h.append(f"<tr class='{'best' if cfg == best else ''}'><td>{label}</td><td>{r[4]}</td>"
                 f"<td>{r[5]:.2f}</td><td>{r[6]:.2f}</td><td>{r[7]:.2f}</td><td>{r[8]:.2f}</td>"
                 f"<td>{r[9]:.2f}</td><td>{r[10]}</td><td>{r[11]:.4f}</td></tr>")
    for n, a in base:
        h.append(f"<tr><td>matcher {n} (no LLM)</td><td>1</td><td>{a['hit1']:.2f}</td><td></td>"
                 f"<td>{a['exact1']:.2f}</td><td>{a['ndcg5']:.2f}</td><td></td><td></td><td>0</td></tr>")
    h.append("</table><h2>Prompt versions</h2><ul>")
    for v, text in J.PROMPT_HISTORY.items():
        h.append(f"<li><b>{v}</b>: {html.escape(text)}</li>")
    h.append("</ul>")

    # Per task: you, then each version (rep 1) of the default model, then other models.
    shown = [c for c in sorted({config_of(n) for n in runs},
                               key=lambda c: (c.split("__")[1] != J.DEFAULT_MODEL, c))]
    rep1 = {c: next((n for n in cfgs[c] if n.endswith("rep1") or "__rep1__" in n), cfgs[c][0])
            for c in shown}
    last = rep1.get(best) if best else None
    h.append("<h2>Every task</h2><p class='legend'>Badges are positions in each ranking (yours, "
             "then each judge's). Blue border: your first pick; green: one of your other "
             "picks. Dots: the judge's grade (●●● shows the subject, ●● its context, ● a "
             "partial fit). Hover an icon for its description and the judge's reason. "
             "Tasks where the best configuration misses come first.</p>")

    def key(t):
        s = runs.get(last, {}).get(t)
        if not s:
            return (1, t)
        ranked = [r["icon"] for r in s["final"]["ranking"]]
        return (int(bool(ranked) and ranked[0] in labels[t]["ranking"]), t)

    for t in sorted(ids, key=key):
        r, lab = recs[t], labels[t]
        u = lab["ranking"]
        parents = " › ".join(r.get("parents") or [])
        h.append(f"<div class='card'><div class='title'>{html.escape(r['title'])}</div>"
                 f"<div class='parent'>{html.escape(parents)}</div>")
        you = "".join(icon(n, "u1" if i == 0 else "in", str(i + 1), J.icon_line(n))
                      for i, n in enumerate(u)) or "<i>no good icon</i>"
        h.append(f"<div class='row'><div class='lab'>You</div><div class='icons'>{you}</div></div>")
        for c in shown:
            s = runs[rep1[c]].get(t)
            if not s:
                continue
            ranked = s["final"]["ranking"]
            cells = "".join(icon(x["icon"], "u1" if u and x["icon"] == u[0] else
                                 "in" if x["icon"] in u else "", str(i + 1),
                                 f"grade {x['grade']}: {J.icon_line(x['icon'])} — {x['why']}",
                                 grade=x["grade"]) for i, x in enumerate(ranked[:8]))
            hit = bool(ranked) and bool(u) and ranked[0]["icon"] in u
            tag = f"<span class='tag {'hit' if hit else ''}'>{'hit' if hit else 'miss'}</span>"
            if s["final"]["no_good_icon"]:
                cells = cells + " <i>(no good icon)</i>"
            name = c.replace("gemini-", "").replace("__", " · ")
            h.append(f"<div class='row'><div class='lab'>{html.escape(name)} {tag}</div>"
                     f"<div class='icons'>{cells or '<i>nothing</i>'}</div></div>")
        s = runs.get(last, {}).get(t)
        if s and s.get("analysis"):
            a = s["analysis"]
            q = "; ".join(x["query"] for x in s["searches"]) or "none"
            h.append(f"<div class='note'><b>{html.escape(best.replace('gemini-', '').replace('__', ' · '))}</b>"
                     f" read it as: {html.escape(a.get('main_topic', ''))} — "
                     f"{html.escape(a.get('distinguishing_topic', ''))}. Associations: "
                     f"{html.escape(', '.join(a.get('associations', [])))}. Searched: "
                     f"{html.escape(q)}.</div>")
        h.append("</div>")
    h.append("</main></body></html>")
    OUT.write_text("\n".join(h), encoding="utf-8")
    print(f"-> {OUT}")
    if args.open:
        subprocess.run(["open", str(OUT)])


if __name__ == "__main__":
    main()
