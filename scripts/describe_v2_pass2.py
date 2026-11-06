#!/usr/bin/env python3
"""Pass 2 of the v2 icon descriptions: contrast among look-alikes, with Gemini 4 Argon
as one Vertex AI batch job (docs/icon-descriptions.md §5.4).

Look-alike groups come from image embeddings of the rendered glyphs only
(tcr/glyphs.py `lookalike_groups`, CLIP cosine ≥ THRESHOLD, groups of 2–8), over the
glyphs that v2 keeps. Each request shows a group's glyphs side by side, each under a
letter with its pass-1 description, and rewrites per glyph: `readings`, `ambiguity`,
`near_misses`, `scores.distinctiveness`, plus a new `distinguishing` sentence. Names
never travel with a request; requests and responses are matched by their images.

The merge keeps the replaced pass-1 values in `_meta.pass1`, adds `contrast` to
`_meta.passes`, and records the group's other members in `_meta.lookalikes` (names as
IDs only). Rerun scripts/v2_guard.py afterwards: the near-misses are new text.

  groups    print the groups' size mix (no API calls)
  prepare   build the requests for groups not contrasted yet  -> .cache/v2_pass2/<run>/
  test      contrast the first N of those groups online (default 3), and write them
  submit / status / wait / collect / retry / run   as in describe_v2_pass1.py

Run: python scripts/describe_v2_pass2.py run [--run pass2]
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
from tcr.data import load_icons  # noqa: E402
from tcr.glyphs import glyph_embeddings, lookalike_groups  # noqa: E402
from tcr.vertex_batch import log  # noqa: E402

MODEL = "gemini-4-argon"
TEMPERATURE = 1.0
THRESHOLD = 0.975          # ~33% of kept glyphs in 350 groups (checked on contact sheets)
DESC_DIR = config.ICON_DESC_V2_DIR
WORK = config.ROOT / ".cache" / "v2_pass2"
BATCH_PRICE = (1.00, 5.00)  # promo until 2026-11-30, then (2.00, 10.00)
REPLACED = ("readings", "ambiguity", "near_misses")


def run_of(args):
    return vb.Run(WORK, args.run, "v2_pass2", MODEL)


def read(name):
    return json.loads((DESC_DIR / f"{name}.json").read_text(encoding="utf-8"))


def aliases():
    """{representative: [alias names]} from the v2 files."""
    out = {}
    for p in DESC_DIR.glob("*.json"):
        a = json.loads(p.read_text(encoding="utf-8"))["_meta"].get("alias_of")
        if a:
            out.setdefault(a, []).append(p.stem)
    return out


def groups():
    names = [ic.name for ic in load_icons(version="v2")]   # kept, one per glyph
    return [[names[i] for i in g] for g in lookalike_groups(glyph_embeddings(names), THRESHOLD)]


def pass1_view(d):
    """The pass-1 description, also when a contrast was already merged into `d`."""
    old = d["_meta"].get("pass1") or {}
    v = {**d, **{k: old[k] for k in REPLACED if k in old}}
    if "distinctiveness" in old:
        v["scores"] = {**d["scores"], "distinctiveness": old["distinctiveness"]}
    return v


def done(group):
    return all("contrast" in (read(n)["_meta"].get("passes") or []) for n in group)


def request_for(group):
    system, prompt, parts = dv2.build_contrast_request(group, [pass1_view(read(n)) for n in group])
    return system, prompt, parts


# --- groups / prepare / test ---------------------------------------------------------

def cmd_groups(args):
    gs = groups()
    n = sum(len(g) for g in gs)
    log(f"{len(gs)} groups, {n} glyphs; sizes {sorted(Counter(len(g) for g in gs).items())}; "
        f"{sum(done(g) for g in gs)} groups already contrasted")


def todo_groups(args):
    gs = [g for g in groups() if not done(g)]
    return gs[: args.limit] if args.limit else gs


def cmd_prepare(args):
    run = run_of(args)
    gs = todo_groups(args)
    manifest = {}
    with open(run.input, "w", encoding="utf-8") as f:
        for g in gs:
            system, prompt, parts = request_for(g)
            req = vb.request_json(system, [*parts, prompt], dv2.CONTRAST_SCHEMA, TEMPERATURE)
            manifest[vb.image_key(req)] = {"group": g}
            f.write(json.dumps({"request": req}, ensure_ascii=False) + "\n")
    run.save_manifest(manifest)
    n = sum(len(g) for g in gs)
    est = (n * 1650 + len(gs) * 1500) * BATCH_PRICE[0] / 1e6 + n * 900 * BATCH_PRICE[1] / 1e6
    log(f"{len(gs)} groups ({n} glyphs) -> {run.input} ({run.input.stat().st_size / 1e6:.1f} MB); "
        f"estimated batch cost ${est:.1f}")


def online(group, tag="pass2"):
    from tcr.llm import complete_json
    system, prompt, parts = request_for(group)
    res = complete_json(prompt, system=system, schema=dv2.CONTRAST_SCHEMA, backend="vertex",
                        models=[MODEL], temperature=TEMPERATURE, images=parts,
                        validate=lambda d: not dv2.contrast_hard_problems(d, len(group)),
                        retries=4, tag=tag)
    merge(group, res["data"], res.get("usage"), "online")
    return res


def cmd_test(args):
    for g in todo_groups(argparse.Namespace(limit=args.limit or 3)):
        res = online(g)
        log(f"ok {g}: {res.get('usage')}")
        for e in res["data"]["icons"]:
            log(f"  {e['id']} {e['distinguishing']} | soft: {dv2.contrast_soft_problems(e)}")


# --- merge ---------------------------------------------------------------------------

def merge(group, data, usage, mode):
    """Write one group's contrast into its glyphs' files (and their aliases)."""
    by_id = {e["id"]: e for e in data["icons"]}
    al = aliases()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for letter, name in zip(dv2.LETTERS, group):
        e = by_id[letter]
        for fname in [name] + sorted(al.get(name, [])):
            path = DESC_DIR / f"{fname}.json"
            d = json.loads(path.read_text(encoding="utf-8"))
            meta = d["_meta"]
            if "pass1" not in meta:   # the first contrast keeps what it replaces
                meta["pass1"] = {**{k: d[k] for k in REPLACED},
                                 "distinctiveness": d["scores"]["distinctiveness"]}
            d.update(readings=e["readings"], ambiguity=e["ambiguity"],
                     near_misses=e["near_misses"], distinguishing=e["distinguishing"])
            d["scores"]["distinctiveness"] = e["distinctiveness"]
            passes = meta.get("passes") or [meta.get("pass", "describe")]
            meta["passes"] = passes + ["contrast"] if "contrast" not in passes else passes
            meta["lookalikes"] = [n for n in group if n != name]
            meta["contrast"] = {"model": MODEL, "prompt_version": dv2.CONTRAST_VERSION,
                                "mode": mode, "group_size": len(group), "usage": usage,
                                "soft_problems": dv2.contrast_soft_problems(e), "created": now}
            path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


# --- batch ---------------------------------------------------------------------------

def cmd_submit(args):
    run_of(args).submit(f"tcr-v2-{args.run}")


def cmd_status(args):
    log(vb.describe_job(run_of(args).job()))


def cmd_wait(args):
    return run_of(args).wait(args.poll)


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
        g = entry["group"]
        seen.add(tuple(g))
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
        probs = dv2.contrast_hard_problems(data, len(g)) if data is not None else ["unparseable"]
        if probs:
            failures[probs[0].split(":")[0] if len(probs[0]) < 40 else "invalid"] += 1
            continue
        merge(g, data, u, "batch")
    todo = [v["group"] for v in manifest.values() if not done(v["group"])]
    state.update(collected=datetime.now(timezone.utc).isoformat(), todo=todo)
    run.save_state(state)
    log(f"merged {len(manifest) - len(todo)} of {len(manifest)} groups; failures "
        f"{dict(failures) or 'none'}; to retry: {len(todo)}; tokens {dict(usage_sum)}; "
        f"cost ≈ ${vb.cost(usage_sum, BATCH_PRICE):.2f}")


def cmd_retry(args):
    from tcr.llm import LLMError
    run = run_of(args)
    state = run.state()
    todo = [g for g in state.get("todo", []) if not done(g)]
    log(f"{len(todo)} groups to contrast synchronously")
    left = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(online, g): g for g in todo}
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
    ap.add_argument("cmd", choices=["groups", "prepare", "test", "submit", "status", "wait",
                                    "collect", "retry", "run"])
    ap.add_argument("--run", default="pass2", help="run name (state in .cache/v2_pass2/<run>/)")
    ap.add_argument("--limit", type=int, default=0, help="prepare/test: only the first N groups")
    ap.add_argument("--poll", type=int, default=300, help="wait: seconds between checks")
    ap.add_argument("--workers", type=int, default=2, help="retry: parallel requests")
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
