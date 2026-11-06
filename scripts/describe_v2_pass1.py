#!/usr/bin/env python3
"""Pass 1 of the v2 icon descriptions: describe every kept glyph with Gemini 4 Argon,
as one Vertex AI batch job (docs/icon-descriptions-v2.md §5).

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
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tcr import config  # noqa: E402
from tcr import describe_v2 as dv2  # noqa: E402
from tcr.data import glyph_hashes, icon_labels, load_icons  # noqa: E402

MODEL = "gemini-4-argon"
TEMPERATURE = 1.0
OUT_DIR = config.DATA_DIR / "icon_descriptions_v2"
WORK = config.ROOT / ".cache" / "v2_pass1"
# USD per 1M tokens, batch (input, output incl. thinking); promo price until 2026-11-30,
# then (2.00, 10.00). See scripts/v2_pilot.py PRICES.
BATCH_PRICE = (1.00, 5.00)
ENDED = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
         "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


def env():
    from dotenv import load_dotenv
    load_dotenv(config.ROOT / ".env")
    project = os.environ["VERTEX_PROJECT"]
    # gemini-4-argon is served on the global endpoint only (us-central1: 404).
    location = os.environ.get("VERTEX_BATCH_LOCATION", "global")
    bucket = os.environ.get("VERTEX_BATCH_BUCKET", f"{project}-tcr-batch")
    return project, location, bucket


BUCKET_LOCATION = "us-central1"   # a bucket needs a real region; global jobs read it fine


_CLIENT = None


def client():
    """One client for the process: a Client that is garbage-collected closes its HTTP
    connection, which breaks `client().batches…` chains."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        project, location, _ = env()
        _CLIENT = genai.Client(vertexai=True, project=project, location=location)
    return _CLIENT


def run_dir(run):
    d = WORK / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_state(run):
    p = run_dir(run) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(run, state):
    (run_dir(run) / "state.json").write_text(json.dumps(state, indent=1))


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def gcloud(*args, check=True):
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=check)


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
    for ic in load_icons(include_discarded=True):   # one representative per glyph
        h = hashes[ic.name]
        if h not in rejected:
            out.append((ic.name, h, sorted(n for n in aliases[h] if n != ic.name)))
    return sorted(out)


# --- prepare -----------------------------------------------------------------------

def request_json(name, glyph):
    """The REST body of one generateContent request, as the online call sends it."""
    system, prompt, images = dv2.build_request(name, key=glyph)
    (png, mime), = images
    return {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": mime, "data": base64.b64encode(png).decode()}},
            {"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": TEMPERATURE, "responseMimeType": "application/json",
                             "responseJsonSchema": dv2.RESPONSE_SCHEMA},
    }


def image_key(req):
    return hashlib.sha1(req["contents"][0]["parts"][0]["inlineData"]["data"].encode()).hexdigest()


def cmd_prepare(args):
    items = glyphs_to_describe()
    if args.only_missing:
        items = [it for it in items if not (OUT_DIR / f"{it[0]}.json").exists()]
    if args.limit:
        items = items[: args.limit]
    d = run_dir(args.run)
    manifest = {}
    with open(d / "input.jsonl", "w", encoding="utf-8") as f:
        for name, glyph, al in items:
            req = request_json(name, glyph)
            k = image_key(req)
            if k in manifest:
                # Two glyphs that differ only by position in the 128 px render (e.g.
                # arrow_back_ios / arrow_back_ios_new) are one image once centred:
                # describe it once, write it under every name.
                manifest[k]["aliases"] += [name] + al
                continue
            manifest[k] = {"icon": name, "glyph": glyph, "aliases": al}
            f.write(json.dumps({"request": req}, ensure_ascii=False) + "\n")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=0))
    mb = (d / "input.jsonl").stat().st_size / 1e6
    est = len(items) * (3900 * BATCH_PRICE[0] + 1650 * BATCH_PRICE[1]) / 1e6
    log(f"{len(items)} glyphs ({sum(len(v['aliases']) for v in manifest.values())} aliases) "
        f"-> {d / 'input.jsonl'} ({mb:.1f} MB); estimated batch cost ${est:.0f}")


# --- submit / status / wait ---------------------------------------------------------

def ensure_bucket(bucket):
    if gcloud("storage", "buckets", "describe", f"gs://{bucket}", check=False).returncode == 0:
        return
    log(f"creating bucket gs://{bucket} in {BUCKET_LOCATION}")
    gcloud("storage", "buckets", "create", f"gs://{bucket}", f"--location={BUCKET_LOCATION}",
           "--uniform-bucket-level-access", "--public-access-prevention")


def cmd_submit(args):
    state = load_state(args.run)
    if state.get("job"):
        log(f"already submitted: {state['job']} (use `status`); not resubmitting")
        return
    d = run_dir(args.run)
    if not (d / "input.jsonl").exists():
        sys.exit("no input.jsonl: run `prepare` first")
    project, location, bucket = env()
    ensure_bucket(bucket)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = f"gs://{bucket}/v2_pass1/{args.run}-{stamp}"
    log(f"uploading input to {prefix}/input.jsonl")
    gcloud("storage", "cp", str(d / "input.jsonl"), f"{prefix}/input.jsonl")
    from google.genai import types
    job = client().batches.create(
        model=MODEL, src=f"{prefix}/input.jsonl",
        config=types.CreateBatchJobConfig(dest=f"{prefix}/output", display_name=f"tcr-v2-{args.run}"))
    state.update(job=job.name, prefix=prefix, submitted=stamp, location=location)
    save_state(args.run, state)
    log(f"submitted {job.name} ({job.state})")


def job_info(run):
    state = load_state(run)
    if not state.get("job"):
        sys.exit("nothing submitted for this run")
    return state, client().batches.get(name=state["job"])


def describe_job(job):
    s = job.completion_stats
    stats = (f"ok {s.successful_count}, failed {s.failed_count}, incomplete {s.incomplete_count}"
             if s else "no counts yet")
    err = f" error: {job.error}" if job.error else ""
    return f"{job.state} ({stats}){err}"


def cmd_status(args):
    _, job = job_info(args.run)
    log(describe_job(job))


def cmd_wait(args):
    while True:
        _, job = job_info(args.run)
        log(describe_job(job))
        if str(job.state).split(".")[-1] in ENDED or str(job.state) in ENDED:
            return job
        time.sleep(args.poll)


# --- collect -----------------------------------------------------------------------

def response_text(resp):
    parts = resp["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def usage_of(resp):
    um = resp.get("usageMetadata", {})
    return {"prompt": um.get("promptTokenCount", 0), "output": um.get("candidatesTokenCount", 0),
            "thoughts": um.get("thoughtsTokenCount", 0), "total": um.get("totalTokenCount", 0)}


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
    state, job = job_info(args.run)
    d = run_dir(args.run)
    out = d / "output"
    out.mkdir(exist_ok=True)
    log(f"job {describe_job(job)}; downloading {state['prefix']}/output")
    gcloud("storage", "cp", "-r", f"{state['prefix']}/output/*", str(out))
    manifest = json.loads((d / "manifest.json").read_text())
    seen, failures, usage_sum = set(), Counter(), Counter()
    for path in sorted(out.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            entry = manifest.get(image_key(rec["request"]))
            if entry is None:
                failures["unmatched"] += 1
                continue
            seen.add(entry["icon"])
            resp = rec.get("response")
            if not resp or rec.get("status"):
                failures["api error"] += 1
                continue
            u = usage_of(resp)
            usage_sum.update(u)
            try:
                data = dv2._parse_or_none(response_text(resp))
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
    save_state(args.run, state)
    cost = (usage_sum["prompt"] * BATCH_PRICE[0]
            + (usage_sum["output"] + usage_sum["thoughts"]) * BATCH_PRICE[1]) / 1e6
    log(f"wrote {len(manifest) - len(todo)} of {len(manifest)} glyphs to {OUT_DIR}; "
        f"failures {dict(failures) or 'none'}; to retry: {len(todo)}; "
        f"tokens {dict(usage_sum)}; cost ≈ ${cost:.2f}")


# --- retry -------------------------------------------------------------------------

def cmd_retry(args):
    from tcr.llm import LLMError, complete_json
    state = load_state(args.run)
    manifest = {v["icon"]: v for v in json.loads((run_dir(args.run) / "manifest.json").read_text()).values()}
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
    save_state(args.run, state)
    log(f"done; {len(left)} still failing" + (f": {left}" if left else ""))


def cmd_run(args):
    if not (run_dir(args.run) / "input.jsonl").exists():
        cmd_prepare(args)
    cmd_submit(args)
    job = cmd_wait(args)
    if "SUCCEEDED" in str(job.state):
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
