#!/usr/bin/env python3
"""Model pilot for the v2 icon descriptions (docs/icon-descriptions-v2.md §5, step 1).

Which single Gemini model should describe all ~3,600 glyphs? The same v2 request
(tcr/describe_v2.py) goes to every model config for a stratified sample of glyphs;
the outputs are judged blind against the glyph, and the report compares misreadings,
output quality and cost.

  sample   pick the glyphs (stratified by the manual icon review)  -> sample.json
  run      call every config for every glyph                       -> raw/<config>/<icon>.json
  sheets   blind judging packets (glyph ids, shuffled slots A-E)   -> judging/, blind_key.json
  report   unblind the verdicts, compute the metrics               -> report.md, report.html

Everything lands in results/v2_pilot/ (committed: public glyphs and generated text
only; the contamination check reports counts, never eval text).

Run: python scripts/v2_pilot.py sample | run [--configs …] [--limit N] | sheets | report
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import describe_v2 as dv2  # noqa: E402
from tcr.data import glyph_hashes, icon_labels, load_icons  # noqa: E402
from tcr.llm import LLMError, complete_json  # noqa: E402

OUT = config.RESULTS_DIR / "v2_pilot"
RAW = OUT / "raw"
JUDGING = OUT / "judging"
SEED = "v2-pilot-1"

# name -> (model id, generation options). Same prompt, references and image for all.
CONFIGS = {
    "3-flash": ("gemini-3-flash-preview", {}),
    "3.8-flash": ("gemini-3.8-flash", {}),
    "3.8-flash-low": ("gemini-3.8-flash", {"thinking_level": "low"}),
    "3.1-pro": ("gemini-3.1-pro-preview", {}),
    "4-argon": ("gemini-4-argon", {}),
}

# USD per 1M tokens: (standard in, standard out, batch in, batch out); output includes
# thinking. From the Gemini API pricing page, read 2026-11-05 (Vertex normally matches).
# Promotions: gemini-3.8-flash until 2026-12-31, gemini-4-argon until 2026-11-30
# (both double afterwards).
PRICES = {
    "gemini-3-flash-preview": (0.50, 3.00, 0.25, 1.50),
    "gemini-3.8-flash": (0.75, 3.75, 0.375, 1.875),
    "gemini-3.1-pro-preview": (2.00, 12.00, 1.00, 6.00),
    "gemini-4-argon": (2.00, 10.00, 1.00, 5.00),
}

# The glyphs pass 1 will describe: every distinct glyph except the human rejects.
PASS1_GLYPHS = 3621

# v1 misreadings found in the spot check (docs/icon-descriptions-v2.md §1).
KNOWN = ("monitor_weight", "tune", "electric_meter", "skeleton")
STRATA = {"useful": 16, "unsure": 24, "reject": 6}   # with KNOWN: 50 glyphs


# --- sample ------------------------------------------------------------------------

def levels_by_glyph():
    """{glyph hash: level}; 'somewhat' and unlabelled glyphs are 'unsure'."""
    hashes = glyph_hashes()
    out = {}
    for name, lab in icon_labels().items():
        if hashes.get(name):
            out[hashes[name]] = "unsure" if lab == "somewhat" else lab
    return out


def cmd_sample(args):
    hashes = glyph_hashes()
    levels = levels_by_glyph()
    reps = [ic.name for ic in load_icons(include_discarded=True)]   # one name per glyph
    by_level = defaultdict(list)
    for n in sorted(reps):
        by_level[levels.get(hashes[n], "unsure")].append(n)
    rng = random.Random(SEED)
    picked = []
    known_glyphs = {hashes[k] for k in KNOWN}
    for level, n in STRATA.items():
        pool = [x for x in by_level[level] if hashes[x] not in known_glyphs]
        picked += rng.sample(pool, n)
    sample = [{"icon": n, "glyph": hashes[n], "level": levels.get(hashes[n], "unsure"),
               "known_v1_misread": n in KNOWN} for n in list(KNOWN) + sorted(picked)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
    print(f"{len(sample)} glyphs:", dict(Counter(s["level"] for s in sample)))


def load_sample():
    return json.loads((OUT / "sample.json").read_text(encoding="utf-8"))


# --- run ---------------------------------------------------------------------------

def cost(model, usage, batch=False):
    p = PRICES[model]
    pin, pout = (p[2], p[3]) if batch else (p[0], p[1])
    return (usage["prompt"] * pin + (usage["output"] + usage["thoughts"]) * pout) / 1e6


def run_one(cfg, item):
    model, gen = CONFIGS[cfg]
    system, prompt, images = dv2.build_request(item["icon"], key=item["glyph"])
    out = {"icon": item["icon"], "config": cfg, "model": model, "gen": gen,
           "prompt_version": dv2.PROMPT_VERSION}
    try:
        res = complete_json(prompt, system=system, schema=dv2.RESPONSE_SCHEMA, backend="vertex",
                            models=[model], temperature=1.0, images=images, gen=gen, tag=cfg,
                            validate=lambda d: not dv2.hard_problems(d), retries=3)
        out.update(data=res["data"], usage=res.get("usage"),
                   soft_problems=dv2.soft_problems(res["data"]))
    except LLMError as e:
        out.update(data=None, usage=None, error=str(e)[:500])
    path = RAW / cfg / f"{item['icon']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def cmd_run(args):
    sample = load_sample()[: args.limit] if args.limit else load_sample()
    cfgs = args.configs or list(CONFIGS)
    jobs = [(c, it) for c in cfgs for it in sample
            if args.force or not (RAW / c / f"{it['icon']}.json").exists()
            or json.loads((RAW / c / f"{it['icon']}.json").read_text())["data"] is None]
    print(f"{len(jobs)} calls ({len(cfgs)} configs × {len(sample)} glyphs, minus done)")
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, c, it): (c, it["icon"]) for c, it in jobs}
        for f in as_completed(futs):
            c, icon = futs[f]
            r = f.result()
            done += 1
            u = r.get("usage") or {}
            status = "ERROR " + r["error"][:80] if r.get("error") else (
                f"{u.get('prompt', 0)}+{u.get('output', 0)}+{u.get('thoughts', 0)} tok, "
                f"{u.get('seconds', 0)}s" + (f", {len(r['soft_problems'])} soft" if r["soft_problems"] else ""))
            print(f"[{done}/{len(jobs)}] {c:14} {icon:28} {status}", flush=True)
    cost_table(cfgs, sample)


def load_raw(cfg, icon):
    p = RAW / cfg / f"{icon}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cost_table(cfgs, sample):
    print(f"\n{'config':14} {'n':>3} {'in':>6} {'out':>6} {'think':>6} {'s':>5} "
          f"{'$/glyph':>8} {'pass1 $':>8} {'batch $':>8}")
    for c in cfgs:
        rs = [r for it in sample if (r := load_raw(c, it["icon"])) and r.get("usage")]
        if not rs:
            continue
        m = CONFIGS[c][0]
        mean = {k: statistics.mean(r["usage"][k] for r in rs) for k in ("prompt", "output", "thoughts", "seconds")}
        per = statistics.mean(cost(m, r["usage"]) for r in rs)
        per_b = statistics.mean(cost(m, r["usage"], batch=True) for r in rs)
        print(f"{c:14} {len(rs):3d} {mean['prompt']:6.0f} {mean['output']:6.0f} {mean['thoughts']:6.0f} "
              f"{mean['seconds']:5.0f} {per:8.4f} {per * PASS1_GLYPHS:8.0f} {per_b * PASS1_GLYPHS:8.0f}")


# --- sheets (blind) ----------------------------------------------------------------

SLOTS = "ABCDE"


def _label_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def contact_sheet(items, path, px=256, cols=5):
    from PIL import Image, ImageDraw
    rows = math.ceil(len(items) / cols)
    pad = 40
    sheet = Image.new("RGB", (cols * px, rows * (px + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _label_font(28)
    for i, (gid, icon) in enumerate(items):
        x, y = (i % cols) * px, (i // cols) * (px + pad)
        draw.text((x + 10, y + 6), gid, fill=(200, 0, 0), font=font)
        sheet.paste(Image.open(io.BytesIO(dv2.glyph_png(icon))).convert("RGB"), (x, y + pad))
        draw.rectangle([x, y, x + px - 1, y + px + pad - 1], outline=(200, 200, 200))
    sheet.save(path)


def describe_slot(d):
    dep = d["depiction"]
    mods = "; ".join(dep.get("modifiers") or []) or "none"
    reads = "; ".join(f"{r['meaning']} ({r['strength']})" for r in d["readings"])
    sc = d["scores"]
    return (f"- drawn: {dep['primary_object']} | modifiers: {mods} | {dep.get('composition', '')}\n"
            f"  - readings: {reads}\n"
            f"  - ambiguity: {d.get('ambiguity') or '—'}\n"
            f"  - discard: {d['discard']} | recognizability {sc['recognizability']}, "
            f"task_fit {sc['task_fit']}, distinctiveness {sc['distinctiveness']}")


def text_slot(d):
    ex = d["example_tasks"]
    pick = ex[::2][:8]
    lines = ["  - examples: " + " / ".join(
        (f"[{e['parent']}] " if e.get("parent") else "") + e["task"] for e in pick)]
    lines.append("  - ES/RU: " + " / ".join(d["example_tasks_es"][:1] + d["example_tasks_ru"][:1]))
    lines.append("  - near-misses: " + " / ".join(
        f"{n['task']} → better: {n['better']} ({n['trap']})" for n in d["near_misses"][:4]))
    return "\n".join(lines)


JUDGE_README = """# Blind judging: v2 icon-description pilot

Each glyph (g01…g50) was described by 5 anonymous model configs, shown as slots A–E in
an order shuffled per glyph. Look at the glyph on the contact sheet, then judge each
slot. Judge what is DRAWN and how a person would read it; you are not told the icon's
official name, and it does not matter: a reading most people would share is correct
even if the font names the glyph differently.

## Task 1: glyph reading (all 50 glyphs; packets `batch_*.md` with `sheet_*.png`)

For each glyph and slot, a verdict on `drawn` + `readings` together:

- `correct`: the depiction is right and the primary reading is what most people would
  take it to mean (or a fair reading of a genuinely ambiguous glyph).
- `partly`: the main object is right but a meaningful modifier is missed or misread, or
  the primary reading is weak while a secondary one is right.
- `wrong`: the main object is misidentified (e.g. a bathroom scale read as a chat
  bubble), or the readings would send the icon to the wrong tasks.
- `none`: the slot has no output.

Ambiguity, discard and scores are context; do not judge them, but mention in the note
if they are clearly off (e.g. discard=true for a clear everyday object).

Write one JSON line per (glyph, slot) to `results/v2_pilot/judgments.jsonl`:
`{"glyph": "g07", "slot": "C", "verdict": "partly", "note": "misses the slash (disabled)"}`.
Notes are short; required for `partly` and `wrong`.

## Task 2: text quality (10 glyphs; `text_packet.md`)

For each glyph and slot, rate 1–5 (5 = best), using the selection principles: the icon
should show the task's central subject; properties and generic verbs don't count.

- `realism`: do the examples read like real people's task titles (varied, concrete,
  natural lengths), not templates?
- `fit`: is this icon really the best icon for the examples (central subject), given
  what the glyph shows?
- `near_miss`: are the near-misses genuinely hard (they touch the concept) and is the
  `better` icon right?

Write to `results/v2_pilot/text_judgments.jsonl`:
`{"glyph": "g07", "slot": "C", "realism": 4, "fit": 3, "near_miss": 4, "note": "…"}`.

Do NOT open `results/v2_pilot/blind_key.json` or `results/v2_pilot/raw/`.
"""


def cmd_sheets(args):
    sample = load_sample()
    rng = random.Random(SEED + "-blind")
    order = sample[:]
    rng.shuffle(order)
    key = {}
    JUDGING.mkdir(parents=True, exist_ok=True)
    for old in JUDGING.glob("*"):
        old.unlink()
    gids = []
    for i, it in enumerate(order):
        gid = f"g{i + 1:02d}"
        cfgs = list(CONFIGS)
        rng.shuffle(cfgs)
        key[gid] = {"icon": it["icon"], "slots": dict(zip(SLOTS, cfgs))}
        gids.append(gid)
    for b in range(0, len(gids), 10):
        chunk = gids[b:b + 10]
        n = b // 10 + 1
        contact_sheet([(g, key[g]["icon"]) for g in chunk], JUDGING / f"sheet_{n}.png")
        lines = [f"# Batch {n} (glyphs on sheet_{n}.png)\n"]
        for g in chunk:
            lines.append(f"## {g}\n")
            for s, c in key[g]["slots"].items():
                r = load_raw(c, key[g]["icon"])
                body = describe_slot(r["data"]) if r and r.get("data") else "- (no output)"
                lines.append(f"**{s}**\n{body}\n")
        (JUDGING / f"batch_{n}.md").write_text("\n".join(lines), encoding="utf-8")
    # Text packet: 10 glyphs that are not human rejects.
    text_gids = [g for g in gids if levels_of(sample)[key[g]["icon"]] != "reject"][:10]
    lines = ["# Text quality (glyphs are on the sheets of their batch)\n"]
    for g in text_gids:
        lines.append(f"## {g} (sheet_{gids.index(g) // 10 + 1}.png)\n")
        for s, c in key[g]["slots"].items():
            r = load_raw(c, key[g]["icon"])
            body = ("  - primary reading: " + r["data"]["readings"][0]["meaning"] + "\n"
                    + text_slot(r["data"])) if r and r.get("data") else "  - (no output)"
            lines.append(f"**{s}**\n{body}\n")
    (JUDGING / "text_packet.md").write_text("\n".join(lines), encoding="utf-8")
    (JUDGING / "README.md").write_text(JUDGE_README, encoding="utf-8")
    (OUT / "blind_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(gids) // 10 + (len(gids) % 10 > 0)} batches + text packet ({len(text_gids)} glyphs) in {JUDGING}")


def levels_of(sample):
    return {s["icon"]: s["level"] for s in sample}


# --- report ------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def length_bucket(t):
    w = dv2.words(t)
    return "2-5" if 2 <= w <= 5 else "6-12" if 6 <= w <= 12 else "13-16" if 13 <= w <= 16 else "other"


def contamination(examples_by_cfg):
    """Per config: (exact normalized matches, near-duplicates at e5 cosine ≥ 0.95) of
    generated example tasks against every eval title. Counts only."""
    evals = dv2.eval_titles()
    banned = {dv2.norm(t) for t in evals}
    out = {c: [sum(dv2.norm(t) in banned for t in ex), 0] for c, ex in examples_by_cfg.items()}
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.MULTILINGUAL_MODEL)
        pre = config.query_prefix_for(config.MULTILINGUAL_MODEL)
        E = model.encode([pre + t for t in evals], batch_size=128, normalize_embeddings=True)
        for c, ex in examples_by_cfg.items():
            if ex:
                X = model.encode([pre + t for t in ex], batch_size=128, normalize_embeddings=True)
                out[c][1] = int(((X @ E.T).max(axis=1) >= 0.95).sum())
    except Exception as e:  # noqa: BLE001 — the exact check still stands
        print(f"  (embedding check skipped: {e})")
    return out


def cmd_report(args):
    sample = load_sample()
    lev = levels_of(sample)
    key = json.loads((OUT / "blind_key.json").read_text(encoding="utf-8"))
    verdicts = {}
    for src in ("judgments.jsonl", "judgments_user.jsonl"):   # the user's overrides win
        for j in read_jsonl(OUT / src):
            cfg = key[j["glyph"]]["slots"][j["slot"]]
            verdicts[(key[j["glyph"]]["icon"], cfg)] = {**j, "source": src}
    textq = {}
    for j in read_jsonl(OUT / "text_judgments.jsonl"):
        textq[(key[j["glyph"]]["icon"], key[j["glyph"]]["slots"][j["slot"]])] = j
    icons = [s["icon"] for s in sample]
    rows = {}
    examples_by_cfg = {}
    for c, (model, _) in CONFIGS.items():
        rs = {i: load_raw(c, i) for i in icons}
        ok = {i: r for i, r in rs.items() if r and r.get("data")}
        v = Counter(verdicts.get((i, c), {}).get("verdict", "—") for i in icons)
        n_j = sum(v[k] for k in ("correct", "partly", "wrong"))
        ex = [e["task"] for r in ok.values() for e in r["data"]["example_tasks"]]
        examples_by_cfg[c] = ex + [t for r in ok.values()
                                   for t in r["data"]["example_tasks_es"] + r["data"]["example_tasks_ru"]]
        per_icon = defaultdict(set)
        for i, r in ok.items():
            for e in r["data"]["example_tasks"]:
                per_icon[dv2.norm(e["task"])].add(i)
        mix = Counter(length_bucket(t) for t in ex)
        usage = [r["usage"] for r in ok.values() if r.get("usage")]
        tq = [textq[(i, c)] for i in icons if (i, c) in textq]
        disc = lambda level, want: [r["data"]["discard"] == want for i, r in ok.items() if lev[i] == level]
        fit = lambda level: [r["data"]["scores"]["task_fit"] for i, r in ok.items() if lev[i] == level]
        rows[c] = {
            "model": model, "ok": len(ok), "failed": len(icons) - len(ok),
            "retries": sum(1 for r in ok.values() if (r.get("usage") or {}).get("attempts", 1) > 1),
            "judged": n_j, "correct": v["correct"], "partly": v["partly"], "wrong": v["wrong"],
            "wrong_ci": wilson(v["wrong"], n_j), "not_correct_ci": wilson(v["wrong"] + v["partly"], n_j),
            "soft": statistics.mean(len(r["soft_problems"]) for r in ok.values()) if ok else 0,
            "n_examples": statistics.mean(len(r["data"]["example_tasks"]) for r in ok.values()) if ok else 0,
            "mix": {b: mix[b] / max(1, len(ex)) for b in ("2-5", "6-12", "13-16", "other")},
            "reused": sum(1 for s in per_icon.values() if len(s) > 1),
            "domains": statistics.median(len({e["domain"] for e in r["data"]["example_tasks"]})
                                         for r in ok.values()) if ok else 0,
            "traps": statistics.mean(len({n["trap"] for n in r["data"]["near_misses"]})
                                     for r in ok.values()) if ok else 0,
            "reject_discarded": (sum(disc("reject", True)), len(disc("reject", True))),
            "useful_kept": (sum(disc("useful", False)), len(disc("useful", False))),
            "fit_by_level": {l: round(statistics.mean(fit(l)), 2) if fit(l) else None
                             for l in ("useful", "unsure", "reject")},
            "tokens": {k: statistics.mean(u[k] for u in usage) for k in ("prompt", "output", "thoughts", "seconds")} if usage else {},
            "usd_glyph": statistics.mean(cost(model, u) for u in usage) if usage else 0,
            "usd_glyph_batch": statistics.mean(cost(model, u, True) for u in usage) if usage else 0,
            "textq": {k: round(statistics.mean(t[k] for t in tq), 2) for k in ("realism", "fit", "near_miss")} if tq else {},
        }
    contam = contamination(examples_by_cfg)
    for c in rows:
        rows[c]["contam_exact"], rows[c]["contam_near"] = contam[c]
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = report_md(rows, len(icons))
    (OUT / "report.md").write_text(md, encoding="utf-8")
    (OUT / "report.html").write_text(report_html(rows, sample, verdicts, textq), encoding="utf-8")
    print(md)
    print(f"\nwrote {OUT / 'report.md'} and {OUT / 'report.html'}")


def pct(x):
    return f"{100 * x:.0f}%"


def report_md(rows, n):
    L = [f"# v2 pilot: {n} glyphs × {len(rows)} configs\n",
         "Glyph reading (blind verdicts; 95% Wilson CI):\n",
         "| config | model | judged | correct | partly | wrong | wrong CI | partly+wrong CI |",
         "|---|---|---|---|---|---|---|---|"]
    for c, r in rows.items():
        L.append(f"| {c} | {r['model']} | {r['judged']} | {r['correct']} | {r['partly']} | {r['wrong']} | "
                 f"{pct(r['wrong_ci'][0])}–{pct(r['wrong_ci'][1])} | "
                 f"{pct(r['not_correct_ci'][0])}–{pct(r['not_correct_ci'][1])} |")
    L += ["", "Output quality:\n",
          "| config | ok/failed | retries | soft problems/glyph | examples | 2-5 / 6-12 / 13-16 / other "
          "(target 31/50/19/0) | reused strings | domains (median) | traps/glyph | text realism / fit / near-miss |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in rows.items():
        m = r["mix"]
        tq = r["textq"]
        L.append(f"| {c} | {r['ok']}/{r['failed']} | {r['retries']} | {r['soft']:.1f} | {r['n_examples']:.1f} | "
                 f"{pct(m['2-5'])} / {pct(m['6-12'])} / {pct(m['13-16'])} / {pct(m['other'])} | {r['reused']} | "
                 f"{r['domains']} | {r['traps']:.1f} | "
                 + (f"{tq['realism']} / {tq['fit']} / {tq['near_miss']}" if tq else "—") + " |")
    L += ["", "Agreement with the manual icon review, and contamination:\n",
          "| config | rejects discarded | useful kept | task_fit useful / unsure / reject | "
          "eval matches: exact / cos≥0.95 |", "|---|---|---|---|---|"]
    for c, r in rows.items():
        f = r["fit_by_level"]
        L.append(f"| {c} | {r['reject_discarded'][0]}/{r['reject_discarded'][1]} | "
                 f"{r['useful_kept'][0]}/{r['useful_kept'][1]} | {f['useful']} / {f['unsure']} / {f['reject']} | "
                 f"{r['contam_exact']} / {r['contam_near']} |")
    L += ["", f"Cost (prices read 2026-11-05; pass 1 = {PASS1_GLYPHS} glyphs):\n",
          "| config | input tok | output tok | thinking tok | seconds | $/glyph | pass 1 $ | pass 1 batch $ |",
          "|---|---|---|---|---|---|---|---|"]
    for c, r in rows.items():
        t = r["tokens"]
        if t:
            L.append(f"| {c} | {t['prompt']:.0f} | {t['output']:.0f} | {t['thoughts']:.0f} | {t['seconds']:.0f} | "
                     f"{r['usd_glyph']:.4f} | {r['usd_glyph'] * PASS1_GLYPHS:.0f} | "
                     f"{r['usd_glyph_batch'] * PASS1_GLYPHS:.0f} |")
    return "\n".join(L) + "\n"


def report_html(rows, sample, verdicts, textq):
    e = html.escape
    v1 = {}
    for s in sample:
        p = config.ICON_DESC_DIR / f"{s['icon']}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            v1[s["icon"]] = (", ".join(d.get("visual_concepts", [])[:6]),
                             d.get("discard"), d.get("icon_usefulness"))
    cards = []
    for s in sample:
        icon = s["icon"]
        img = base64.b64encode(dv2.glyph_png(icon)).decode()
        cols = []
        for c in CONFIGS:
            r = load_raw(c, icon)
            vd = verdicts.get((icon, c), {})
            verdict = vd.get("verdict", "—")
            badge = f'<span class="v v-{e(verdict)}">{e(verdict)}</span>'
            if vd.get("source") == "judgments_user.jsonl":
                badge += ' <span class="who">(yours)</span>'
            note = f'<div class="note">{e(vd.get("note", ""))}</div>' if vd.get("note") else ""
            if not r or not r.get("data"):
                cols.append(f'<div class="cfg"><h4>{e(c)}</h4>{badge}<p class="muted">no output: '
                            f'{e((r or {}).get("error", "not run"))}</p></div>')
                continue
            d = r["data"]
            dep = d["depiction"]
            tq = textq.get((icon, c))
            tqs = (f'<div class="muted">text: realism {tq["realism"]}, fit {tq["fit"]}, '
                   f'near-miss {tq["near_miss"]}</div>') if tq else ""
            ex = "".join(f"<li>{'<i>[' + e(x['parent']) + ']</i> ' if x.get('parent') else ''}{e(x['task'])}"
                         f" <span class='muted'>{e(x['domain'])}</span></li>" for x in d["example_tasks"])
            ml = "".join(f"<li>{e(t)}</li>" for t in d["example_tasks_es"] + d["example_tasks_ru"])
            ctx = "".join(f"<li>{e(x['task'])} <span class='muted'>central: {e(x['central'])}</span></li>"
                          for x in d["context_tasks"])
            nm = "".join(f"<li>{e(x['task'])} → <b>{e(x['better'])}</b> <span class='muted'>{e(x['trap'])}</span></li>"
                         for x in d["near_misses"])
            sc = d["scores"]
            cols.append(
                f'<div class="cfg"><h4>{e(c)}</h4>{badge}{note}'
                f'<p><b>{e(dep["primary_object"])}</b>'
                f'{" + " + e("; ".join(dep["modifiers"])) if dep.get("modifiers") else ""}</p>'
                f'<p>{"<br>".join(e(x["meaning"]) + (" <span class=muted>(2nd)</span>" if x["strength"] != "primary" else "") for x in d["readings"])}</p>'
                f'<p class="muted">{e(d.get("ambiguity") or "")}</p>'
                f'<p class="muted">R{sc["recognizability"]} F{sc["task_fit"]} D{sc["distinctiveness"]}'
                f'{" · DISCARD" if d["discard"] else ""}</p>{tqs}'
                f'<details><summary>examples, near-misses</summary><ul>{ex}</ul><ul>{ml}</ul>'
                f'<p class="muted">context</p><ul>{ctx}</ul><p class="muted">near-misses</p><ul>{nm}</ul>'
                f'<p class="muted">{e(d["reasoning"])}</p></details></div>')
        v1c, v1d, v1u = v1.get(icon, ("", None, None))
        cards.append(
            f'<section class="card"><div class="head"><img src="data:image/png;base64,{img}" alt="">'
            f'<div><h3>{e(icon)}</h3><div class="muted">your review: <b>{e(s["level"])}</b>'
            f'{" · known v1 misread" if s["known_v1_misread"] else ""}</div>'
            f'<div class="muted">v1: {e(v1c)}{" · discarded" if v1d else ""} · usefulness {v1u}</div></div></div>'
            f'<div class="grid">{"".join(cols)}</div></section>')
    md = report_md(rows, len(sample))
    table = md_tables_to_html(md)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>v2 pilot report</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d9d9de; --card:#f7f7f9;
  --ok:#1f7a3a; --part:#a86400; --bad:#b3261e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --fg:#ececf0; --muted:#9a9aa2;
  --line:#34343a; --card:#1d1d21; --ok:#5fc27a; --part:#e0a23a; --bad:#ff7b72; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, system-ui, sans-serif;
  margin:0 auto; padding:16px; max-width:1500px; }}
table {{ border-collapse:collapse; margin:8px 0 20px; font-size:13px; display:block; overflow-x:auto; }}
td, th {{ border:1px solid var(--line); padding:4px 8px; text-align:left; white-space:nowrap; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; margin:14px 0; }}
.head {{ display:flex; gap:14px; align-items:center; }}
.head img {{ width:96px; height:96px; background:#fff; border-radius:8px; }}
h3 {{ margin:0 0 4px; }} h4 {{ margin:0 0 4px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:10px; margin-top:10px; }}
.cfg {{ border-top:1px solid var(--line); padding-top:6px; }}
.cfg p {{ margin:4px 0; }} ul {{ margin:4px 0; padding-left:18px; }}
.muted {{ color:var(--muted); font-size:12px; }} .note {{ font-size:12px; font-style:italic; }}
.v {{ font-weight:600; font-size:12px; }} .v-correct {{ color:var(--ok); }}
.v-partly {{ color:var(--part); }} .v-wrong {{ color:var(--bad); }} .who {{ font-size:11px; }}
</style></head><body>
<h1>v2 description pilot</h1>
<p class="muted">Verdicts are blind (model hidden while judging). Glyph images are the 256 px renders the models saw.
To override a verdict, note the icon and config; it goes to judgments_user.jsonl.</p>
{table}
{"".join(cards)}
</body></html>"""


def md_tables_to_html(md):
    out, rows = [], []
    for line in md.splitlines() + [""]:
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue
            rows.append([c.strip() for c in line.strip("|").split("|")])
            continue
        if rows:
            head, *body = rows
            out.append("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</table>")
            rows = []
        if line.startswith("# "):
            continue
        if line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    p = sub.add_parser("run")
    p.add_argument("--configs", nargs="*", choices=list(CONFIGS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true", help="redo glyphs that already have output")
    sub.add_parser("sheets")
    sub.add_parser("report")
    args = ap.parse_args()
    {"sample": cmd_sample, "run": cmd_run, "sheets": cmd_sheets, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
