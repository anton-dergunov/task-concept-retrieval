"""Vertex AI batch jobs for Gemini: submit a JSONL of requests, wait, download.

Shared by the v2 description passes (scripts/describe_v2_pass1.py, _pass2.py). A
run's state lives in `<work>/<run>/state.json`, so a submitted job is never
resubmitted and `wait` / `collect` can resume from any later session. Batch input
and output sit in a private bucket, gs://<project>-tcr-batch (us-central1).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

ENDED = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
         "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
BUCKET_LOCATION = "us-central1"   # a bucket needs a real region; global jobs read it fine


def env():
    from dotenv import load_dotenv
    load_dotenv(config.ROOT / ".env")
    project = os.environ["VERTEX_PROJECT"]
    # gemini-4-argon is served on the global endpoint only (us-central1: 404).
    location = os.environ.get("VERTEX_BATCH_LOCATION", "global")
    bucket = os.environ.get("VERTEX_BATCH_BUCKET", f"{project}-tcr-batch")
    return project, location, bucket


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


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def gcloud(*args, check=True):
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=check)


# --- requests and responses ----------------------------------------------------------

def request_json(system: str, parts, schema: dict, temperature: float = 1.0) -> dict:
    """The REST body of one generateContent request. `parts` is a list of str (text) and
    (bytes, mime type) (inline images), in order."""
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append({"text": p})
        else:
            out.append({"inlineData": {"mimeType": p[1], "data": base64.b64encode(p[0]).decode()}})
    return {
        "contents": [{"role": "user", "parts": out}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json",
                             "responseJsonSchema": schema},
    }


def image_key(req: dict) -> str:
    """Identity of a request by its images (the names never travel with a request)."""
    # Output records echo the request with null fields on text parts.
    data = [p["inlineData"]["data"] for p in req["contents"][0]["parts"] if p.get("inlineData")]
    return hashlib.sha1("".join(data).encode()).hexdigest()


def response_text(resp: dict) -> str:
    parts = resp["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def usage_of(resp: dict) -> dict:
    um = resp.get("usageMetadata", {})
    return {"prompt": um.get("promptTokenCount", 0), "output": um.get("candidatesTokenCount", 0),
            "thoughts": um.get("thoughtsTokenCount", 0), "total": um.get("totalTokenCount", 0)}


def cost(usage: dict, price) -> float:
    return (usage.get("prompt", 0) * price[0]
            + (usage.get("output", 0) + usage.get("thoughts", 0)) * price[1]) / 1e6


# --- a run ---------------------------------------------------------------------------

class Run:
    """One batch run: `<work>/<name>/{input.jsonl, manifest.json, state.json, output/}`,
    uploaded under gs://<bucket>/<gcs_folder>/<name>-<stamp>/."""

    def __init__(self, work: Path, name: str, gcs_folder: str, model: str):
        self.dir = work / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name, self.gcs_folder, self.model = name, gcs_folder, model

    @property
    def input(self) -> Path:
        return self.dir / "input.jsonl"

    def state(self) -> dict:
        p = self.dir / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save_state(self, state: dict) -> None:
        (self.dir / "state.json").write_text(json.dumps(state, indent=1))

    def manifest(self) -> dict:
        return json.loads((self.dir / "manifest.json").read_text())

    def save_manifest(self, manifest: dict) -> None:
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=0))

    def submit(self, display_name: str) -> None:
        state = self.state()
        if state.get("job"):
            log(f"already submitted: {state['job']} (use `status`); not resubmitting")
            return
        if not self.input.exists():
            sys.exit("no input.jsonl: run `prepare` first")
        _, location, bucket = env()
        ensure_bucket(bucket)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        prefix = f"gs://{bucket}/{self.gcs_folder}/{self.name}-{stamp}"
        log(f"uploading input to {prefix}/input.jsonl")
        gcloud("storage", "cp", str(self.input), f"{prefix}/input.jsonl")
        from google.genai import types
        job = client().batches.create(
            model=self.model, src=f"{prefix}/input.jsonl",
            config=types.CreateBatchJobConfig(dest=f"{prefix}/output", display_name=display_name))
        state.update(job=job.name, prefix=prefix, submitted=stamp, location=location)
        self.save_state(state)
        log(f"submitted {job.name} ({job.state})")

    def job(self):
        state = self.state()
        if not state.get("job"):
            sys.exit("nothing submitted for this run")
        return client().batches.get(name=state["job"])

    def wait(self, poll: int):
        while True:
            job = self.job()
            log(describe_job(job))
            if ended(job):
                return job
            time.sleep(poll)

    def download(self):
        """Download the job's output; yield every output record (request + response)."""
        state = self.state()
        out = self.dir / "output"
        out.mkdir(exist_ok=True)
        log(f"downloading {state['prefix']}/output")
        gcloud("storage", "cp", "-r", f"{state['prefix']}/output/*", str(out))
        for path in sorted(out.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)


def ensure_bucket(bucket: str) -> None:
    if gcloud("storage", "buckets", "describe", f"gs://{bucket}", check=False).returncode == 0:
        return
    log(f"creating bucket gs://{bucket} in {BUCKET_LOCATION}")
    gcloud("storage", "buckets", "create", f"gs://{bucket}", f"--location={BUCKET_LOCATION}",
           "--uniform-bucket-level-access", "--public-access-prevention")


def describe_job(job) -> str:
    s = job.completion_stats
    stats = (f"ok {s.successful_count}, failed {s.failed_count}, incomplete {s.incomplete_count}"
             if s else "no counts yet")
    err = f" error: {job.error}" if job.error else ""
    return f"{job.state} ({stats}){err}"


def ended(job) -> bool:
    return str(job.state).split(".")[-1] in ENDED


def succeeded(job) -> bool:
    return "SUCCEEDED" in str(job.state)
