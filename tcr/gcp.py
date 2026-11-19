"""Vertex AI runs only on the Google account and project pinned in .env.

Every Vertex call in this repo (tcr.llm's backend "vertex", tcr.vertex_batch) calls
`verify()` first, once per process. It stops the process unless all of these agree:

- `.env`'s VERTEX_ACCOUNT, VERTEX_PROJECT, GOOGLE_APPLICATION_CREDENTIALS (an ADC file)
  and CLOUDSDK_ACTIVE_CONFIG_NAME (a gcloud configuration), with no shell variable
  overriding them (python-dotenv never overrides the shell, so an exported
  VERTEX_PROJECT would otherwise win silently);
- the ADC file's account, as Google's tokeninfo reports it, and its quota project;
- the gcloud configuration's account and project.

The checked credentials go to the genai client explicitly, so no other ADC source is
used; the gcloud CLI always gets --account and --project; batch files must live in
gs://<project>-tcr-batch and batch jobs must belong to the project's number. The account
itself lives only in the gitignored .env, never in code. `scripts/gcp_check.py` runs the
same checks and prints them.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import NamedTuple

from . import config

KEYS = ("VERTEX_ACCOUNT", "VERTEX_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS",
        "CLOUDSDK_ACTIVE_CONFIG_NAME")
TOKENINFO = "https://oauth2.googleapis.com/tokeninfo"


class Refused(SystemExit):
    def __init__(self, why: str):
        super().__init__(f"GCP guard (tcr/gcp.py): {why}\nNothing was sent to Google Cloud.")


class Pin(NamedTuple):
    account: str
    project: str
    number: str          # the project's number, the prefix of its batch job names
    adc: str
    gcloud_config: str
    bucket: str
    credentials: object  # google.auth credentials, checked


def _env_file() -> dict:
    from dotenv import dotenv_values
    return dict(dotenv_values(config.ROOT / ".env"))


def pinned(file: dict = None) -> dict:
    """The pin from .env, checked against the shell; no network. Puts .env's values in
    os.environ, so everything downstream sees them."""
    file = _env_file() if file is None else file
    missing = [k for k in KEYS if not file.get(k)]
    if missing:
        raise Refused(f".env lacks {', '.join(missing)}; Vertex needs an explicit pin.")
    account, project = file["VERTEX_ACCOUNT"], file["VERTEX_PROJECT"]
    expected = {k: file[k] for k in KEYS}
    # Variables that would redirect gcloud or a client if set in the shell.
    expected.update({k: v for k, v in {"VERTEX_BATCH_BUCKET": file.get("VERTEX_BATCH_BUCKET"),
                                      "GOOGLE_CLOUD_PROJECT": project,
                                      "CLOUDSDK_CORE_PROJECT": project,
                                      "CLOUDSDK_CORE_ACCOUNT": account}.items() if v})
    for key, want in expected.items():
        have = os.environ.get(key)
        if have is not None and have != want:
            raise Refused(f"{key} is {have!r} in the environment but .env pins {want!r}; "
                          f"unset it (`unset {key}`) or fix .env.")
    bucket = file.get("VERTEX_BATCH_BUCKET") or f"{project}-tcr-batch"
    if bucket != f"{project}-tcr-batch":
        raise Refused(f"VERTEX_BATCH_BUCKET {bucket!r} is not the project's bucket "
                      f"{project}-tcr-batch.")
    if not Path(file["GOOGLE_APPLICATION_CREDENTIALS"]).is_file():
        raise Refused(f"no ADC file at {file['GOOGLE_APPLICATION_CREDENTIALS']}.")
    os.environ.update({k: file[k] for k in KEYS})
    return {"account": account, "project": project, "adc": file["GOOGLE_APPLICATION_CREDENTIALS"],
            "gcloud_config": file["CLOUDSDK_ACTIVE_CONFIG_NAME"], "bucket": bucket}


def _tokeninfo(token: str) -> dict:
    import requests
    r = requests.get(TOKENINFO, params={"access_token": token}, timeout=30)
    r.raise_for_status()
    return r.json()


def _credentials(adc: str, account: str, project: str):
    import google.auth
    from google.auth.transport.requests import Request
    creds, _ = google.auth.load_credentials_from_file(adc)
    quota = getattr(creds, "quota_project_id", None)
    if quota != project:
        raise Refused(f"the ADC file {adc} bills quota project {quota!r}, not {project!r}.")
    sa = getattr(creds, "service_account_email", None)
    if sa:
        email = sa
    else:
        try:
            creds.refresh(Request())
            email = _tokeninfo(creds.token).get("email")
        except Exception as exc:  # noqa: BLE001
            raise Refused(f"could not confirm the ADC file's account ({exc}); run "
                          f"`gcloud auth application-default login --account {account}` "
                          f"and save the file to {adc}.") from exc
    if email != account:
        raise Refused(f"the ADC file {adc} belongs to {email!r}, not the pinned {account!r}.")
    return creds


def _gcloud(args, gcloud_config: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CLOUDSDK_ACTIVE_CONFIG_NAME": gcloud_config}
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, env=env)


def _check_gcloud(account: str, project: str, gcloud_config: str) -> str:
    """The gcloud configuration's account and project; returns the project number."""
    for prop, want in (("account", account), ("project", project)):
        have = _gcloud(["config", "get", prop], gcloud_config).stdout.strip()
        if have != want:
            raise Refused(f"gcloud configuration {gcloud_config!r} has {prop} {have!r}, "
                          f"not {want!r}.")
    r = _gcloud(["projects", "describe", project, f"--account={account}",
                 "--format=value(projectNumber)"], gcloud_config)
    number = r.stdout.strip()
    if r.returncode or not number.isdigit():
        raise Refused(f"cannot read project {project}'s number as {account}: "
                      f"{r.stderr.strip()[-300:]}")
    return number


@functools.lru_cache(maxsize=None)
def verify() -> Pin:
    p = pinned()
    creds = _credentials(p["adc"], p["account"], p["project"])
    number = _check_gcloud(p["account"], p["project"], p["gcloud_config"])
    return Pin(p["account"], p["project"], number, p["adc"], p["gcloud_config"], p["bucket"], creds)


def gcloud_flags() -> list:
    pin = verify()
    return [f"--account={pin.account}", f"--project={pin.project}"]


def check_gs(path: str) -> str:
    """A gs:// path must lie in the pinned project's bucket."""
    bucket = verify().bucket
    if path.startswith("gs://") and not (path == f"gs://{bucket}" or path.startswith(f"gs://{bucket}/")):
        raise Refused(f"{path} is outside the pinned bucket gs://{bucket}.")
    return path


def check_job(name: str) -> str:
    """A batch job name must belong to the pinned project (projects/<number>/…)."""
    number = verify().number
    if not name.startswith(f"projects/{number}/"):
        raise Refused(f"batch job {name} is not in project {verify().project} "
                      f"(number {number}).")
    return name
