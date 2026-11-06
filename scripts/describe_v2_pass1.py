#!/usr/bin/env python3
"""Pass 1 of the v2 icon descriptions: describe every kept glyph with Gemini 4 Argon,
as one Vertex AI batch job (docs/icon-descriptions.md §5.3).

Which glyphs: every distinct glyph (font aliases collapsed) except the ones rejected
in the manual review (data/labels/icon_curation.jsonl), v1-discarded ones included.
The request is tcr/describe_v2.py's, the same as in the model pilot; it never contains
an icon's name (CLAUDE.md hard rule 1). Requests and responses are matched by the
glyph image's hash, so the name never travels with the request at all.

  prepare   build the requests                    -> .cache/v2_pass1/<run>/input.jsonl
  submit    upload them and start the batch job   (bucket gs://<project>-tcr-batch)
  status    print the job's state
  wait      poll until the job ends (safe to stop and rerun)
  collect   download, validate, write data/icon_descriptions_v2/<name>.json (+ aliases)
  retry     describe what failed or is missing, synchronously
  run       all of the above in order; resumable: a submitted job is never resubmitted

Batch jobs run on Google's side: after `submit`, the computer may sleep; `wait` /
`collect` can run later from anywhere with the same .cache/v2_pass1/<run>/state.json.

Run: python scripts/describe_v2_pass1.py run [--run pass1] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import describe_v2 as dv2  # noqa: E402
from tcr import vertex_batch as vb  # noqa: E402
from tcr.data import glyph_hashes, icon_labels, load_icons  # noqa: E402
from tcr.vertex_batch import log  # noqa: E402

MODEL = "gemini-4-argon"
TEMPERATURE = 1.0
OUT_DIR = config.DATA_DIR / "icon_descriptions_v2"
WORK = config.ROOT / ".cache" / "v2_pass1"
# USD per 1M tokens, batch (input, output incl. thinking); promo price until 2026-11-30,
# then (2.00, 10.00). See scripts/v2_pilot.py PRICES.
BATCH_PRICE = (1.00, 5.00)


def run_of(args):
    return vb.Run(WORK, args.run, "v2_pass1", MODEL)


# --- which glyphs ------------------------------------------------------------------

def glyphs_to_describe():
    """[(representative name, glyph hash, [alias names])], human rejects excluded."""
    hashes = glyph_hashes()
    rejected = {hashes.get(n) for n, lab in icon_labels().items() if lab == "reject"}
    aliases = {}
    for name, h in hashes.items():
        if h:
            aliases.setdefault(h, []).append(name)
    out = []
    for ic in load_icons(include_discarded=True, version="v1"):   # one representative per glyph
        h = hashes[ic.name]
        if h not in rejected:
            out.append((ic.name, h, sorted(n for n in aliases[h] if n != ic.name)))
    return sorted(out)


# --- prepare -----------------------------------------------------------------------

def request_json(name, glyph):
    """The REST body of one generateContent request, as the online call sends it."""
    system, prompt, images = dv2.build_request(name, key=glyph)
    return vb.request_json(system, [*images, prompt], dv2.RESPONSE_SCHEMA, TEMPERATURE)


def cmd_prepare(args):
    items = glyphs_to_describe()
    if args.only_missing:
        items = [it for it in items if not (OUT_DIR / f"{it[0]}.json").exists()]
    if args.limit:
        items = items[: args.limit]
    run = run_of(args)
    manifest = {}
    with open(run.input, "w", encoding="utf-8") as f:
        for name, glyph, al in items:
            req = request_json(name, glyph)
            k = vb.image_key(req)
            if k in manifest:
                # Two glyphs that differ only by position in the 128 px render (e.g.
                # arrow_back_ios / arrow_back_ios_new) are one image once centred:
                # describe it once, write it under every name.
                manifest[k]["aliases"] += [name] + al
                continue
            manifest[k] = {"icon": name, "glyph": glyph, "aliases": al}
            f.write(json.dumps({"request": req}, ensure_ascii=False) + "\n")
    run.save_manifest(manifest)
    mb = run.input.stat().st_size / 1e6
    est = len(items) * (3900 * BATCH_PRICE[0] + 1650 * BATCH_PRICE[1]) / 1e6
    log(f"{len(items)} glyphs ({sum(len(v['aliases']) for v in manifest.values())} aliases) "
        f"-> {run.input} ({mb:.1f} MB); estimated batch cost ${est:.0f}")


# --- submit / status / wait ---------------------------------------------------------

def cmd_submit(args):
    run_of(args).submit(f"tcr-v2-{args.run}")


def cmd_status(args):
    log(vb.describe_job(run_of(args).job()))


def cmd_wait(args):
    return run_of(args).wait(args.poll)


# --- collect -----------------------------------------------------------------------

def write_description(entry, data, usage, mode):
    meta = {"model": MODEL, "prompt_version": dv2.PROMPT_VERSION, "pass": "describe",
            "mode": mode, "glyph": entry["glyph"], "usage": usage,
            "soft_problems": dv2.soft_problems(data),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [entry["icon"]] + entry["aliases"]:
        doc = {"schema_version": 2, **data,
               "_meta": {**meta, "alias_of": None if name == entry["icon"] else entry["icon"]}}
        (OUT_DIR / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_collect(args):
    run = run_of(args)
    state = run.state()
    log(f"job {vb.describe_job(run.job())}")
    manifest = run.manifest()
    seen, failures, usage_sum = set(), Counter(), Counter()
    for rec in run.download():
        entry = manifest.get(vb.image_key(rec["request"]))
        if entry is None:
            failures["unmatched"] += 1
            continue
        seen.add(entry["icon"])
        resp = rec.get("response")
        if not resp or rec.get("status"):
            failures["api error"] += 1
            continue
        u = vb.usage_of(resp)
        usage_sum.update(u)
        try:
            data = dv2._parse_or_none(vb.response_text(resp))
        except (KeyError, IndexError):
            data = None
        probs = dv2.hard_problems(data) if data is not None else ["unparseable"]
        if probs:
            failures[probs[0].split(" of ")[0]] += 1
            continue
        write_description(entry, data, u, "batch")
    missing = [v["icon"] for v in manifest.values() if v["icon"] not in seen]
    failures["missing"] += len(missing)
    todo = [v["icon"] for v in manifest.values() if not (OUT_DIR / f"{v['icon']}.json").exists()]
    state["collected"] = datetime.now(timezone.utc).isoformat()
    state["todo"] = todo
    run.save_state(state)
    cost = vb.cost(usage_sum, BATCH_PRICE)
    log(f"wrote {len(manifest) - len(todo)} of {len(manifest)} glyphs to {OUT_DIR}; "
        f"failures {dict(failures) or 'none'}; to retry: {len(todo)}; "
        f"tokens {dict(usage_sum)}; cost ≈ ${cost:.2f}")


# --- retry -------------------------------------------------------------------------

def cmd_retry(args):
    from tcr.llm import LLMError, complete_json
    run = run_of(args)
    state = run.state()
    manifest = {v["icon"]: v for v in run.manifest().values()}
    todo = [i for i in state.get("todo", manifest) if not (OUT_DIR / f"{i}.json").exists()]
    log(f"{len(todo)} glyphs to describe synchronously")

    def one(icon):
        e = manifest[icon]
        system, prompt, images = dv2.build_request(icon, key=e["glyph"])
        res = complete_json(prompt, system=system, schema=dv2.RESPONSE_SCHEMA, backend="vertex",
                            models=[MODEL], temperature=TEMPERATURE, images=images,
                            validate=lambda d: not dv2.hard_problems(d), retries=4, tag="pass1")
        write_description(e, res["data"], res.get("usage"), "online")

    left = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, i): i for i in todo}
        for f in as_completed(futs):
            try:
                f.result()
                log(f"ok {futs[f]}")
            except LLMError as e:
                left.append(futs[f])
                log(f"FAILED {futs[f]}: {str(e)[:120]}")
    state["todo"] = left
    run.save_state(state)
    log(f"done; {len(left)} still failing" + (f": {left}" if left else ""))


def cmd_run(args):
    if not run_of(args).input.exists():
        cmd_prepare(args)
    cmd_submit(args)
    job = cmd_wait(args)
    if vb.succeeded(job):
        cmd_collect(args)
        cmd_retry(args)
    else:
        log("job did not succeed; see `status`. Collect what exists with `collect`, then `retry`.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "submit", "status", "wait", "collect", "retry", "run"])
    ap.add_argument("--run", default="pass1", help="run name (state in .cache/v2_pass1/<run>/)")
    ap.add_argument("--limit", type=int, default=0, help="prepare: only the first N glyphs (tests)")
    ap.add_argument("--only-missing", action="store_true",
                    help="prepare: skip glyphs that already have a v2 description")
    ap.add_argument("--poll", type=int, default=300, help="wait: seconds between checks")
    ap.add_argument("--workers", type=int, default=2, help="retry: parallel requests")
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
