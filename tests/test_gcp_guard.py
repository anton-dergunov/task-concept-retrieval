"""The Vertex guard (tcr/gcp.py) refuses anything but the pinned account and project.

No network: tokeninfo, the ADC loader and gcloud are replaced by stubs.
"""

import subprocess
import types

import pytest

from tcr import gcp

ACCOUNT, PROJECT = "pinned@example.com", "pinned-project"
REDIRECTS = ("VERTEX_ACCOUNT", "VERTEX_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS",
             "CLOUDSDK_ACTIVE_CONFIG_NAME", "VERTEX_BATCH_BUCKET", "GOOGLE_CLOUD_PROJECT",
             "CLOUDSDK_CORE_PROJECT", "CLOUDSDK_CORE_ACCOUNT")


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for k in REDIRECTS:
        monkeypatch.delenv(k, raising=False)
    adc = tmp_path / "adc.json"
    adc.write_text("{}")
    return {"VERTEX_ACCOUNT": ACCOUNT, "VERTEX_PROJECT": PROJECT,
            "GOOGLE_APPLICATION_CREDENTIALS": str(adc), "CLOUDSDK_ACTIVE_CONFIG_NAME": "cfg"}


def test_pin_from_env_file(env_file):
    p = gcp.pinned(env_file)
    assert (p["account"], p["project"], p["bucket"]) == (ACCOUNT, PROJECT, f"{PROJECT}-tcr-batch")


@pytest.mark.parametrize("key", ["VERTEX_ACCOUNT", "VERTEX_PROJECT",
                                 "GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_ACTIVE_CONFIG_NAME"])
def test_missing_variable_refused(env_file, key):
    del env_file[key]
    with pytest.raises(gcp.Refused, match=key):
        gcp.pinned(env_file)


@pytest.mark.parametrize("key,value", [("VERTEX_PROJECT", "other-project"),
                                       ("GOOGLE_CLOUD_PROJECT", "other-project"),
                                       ("CLOUDSDK_CORE_ACCOUNT", "other@example.com"),
                                       ("GOOGLE_APPLICATION_CREDENTIALS", "/elsewhere.json")])
def test_shell_override_refused(env_file, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(gcp.Refused, match=key):
        gcp.pinned(env_file)


def test_shell_with_same_value_is_fine(env_file, monkeypatch):
    monkeypatch.setenv("VERTEX_PROJECT", PROJECT)
    gcp.pinned(env_file)


def test_foreign_bucket_refused(env_file):
    env_file["VERTEX_BATCH_BUCKET"] = "someone-elses-bucket"
    with pytest.raises(gcp.Refused, match="bucket"):
        gcp.pinned(env_file)


def _fake_creds(monkeypatch, quota=PROJECT, email=ACCOUNT):
    creds = types.SimpleNamespace(quota_project_id=quota, token="t", refresh=lambda req: None)
    monkeypatch.setattr("google.auth.load_credentials_from_file", lambda path: (creds, None))
    monkeypatch.setattr(gcp, "_tokeninfo", lambda token: {"email": email})
    return creds


def test_adc_of_the_pinned_account_passes(monkeypatch):
    creds = _fake_creds(monkeypatch)
    assert gcp._credentials("adc.json", ACCOUNT, PROJECT) is creds


def test_adc_of_another_account_refused(monkeypatch):
    _fake_creds(monkeypatch, email="other@example.com")
    with pytest.raises(gcp.Refused, match="belongs to 'other@example.com'"):
        gcp._credentials("adc.json", ACCOUNT, PROJECT)


def test_adc_billing_another_project_refused(monkeypatch):
    _fake_creds(monkeypatch, quota="other-project")
    with pytest.raises(gcp.Refused, match="quota project"):
        gcp._credentials("adc.json", ACCOUNT, PROJECT)


def _fake_gcloud(monkeypatch, account=ACCOUNT, project=PROJECT, number="123"):
    def run(args, cfg):
        out = {("config", "get", "account"): account, ("config", "get", "project"): project}
        return subprocess.CompletedProcess(args, 0, out.get(tuple(args[:3]), number) + "\n", "")
    monkeypatch.setattr(gcp, "_gcloud", run)


def test_gcloud_config_of_another_account_refused(monkeypatch):
    _fake_gcloud(monkeypatch, account="other@example.com")
    with pytest.raises(gcp.Refused, match="account 'other@example.com'"):
        gcp._check_gcloud(ACCOUNT, PROJECT, "cfg")


def test_gcloud_config_returns_project_number(monkeypatch):
    _fake_gcloud(monkeypatch)
    assert gcp._check_gcloud(ACCOUNT, PROJECT, "cfg") == "123"


@pytest.fixture
def pin(monkeypatch):
    p = gcp.Pin(ACCOUNT, PROJECT, "123", "adc.json", "cfg", f"{PROJECT}-tcr-batch", None)
    monkeypatch.setattr(gcp, "verify", lambda: p)
    return p


def test_jobs_of_other_projects_refused(pin):
    assert gcp.check_job("projects/123/locations/global/batchPredictionJobs/9")
    with pytest.raises(gcp.Refused, match="not in project"):
        gcp.check_job("projects/77349788575/locations/global/batchPredictionJobs/9")


def test_paths_outside_the_bucket_refused(pin):
    assert gcp.check_gs(f"gs://{PROJECT}-tcr-batch/judge/x/input.jsonl")
    assert gcp.check_gs("local/file.jsonl") == "local/file.jsonl"
    for path in ("gs://valid-octagon-514423-n9-tcr-batch/judge/x", f"gs://{PROJECT}-tcr-batch-evil/x"):
        with pytest.raises(gcp.Refused, match="outside"):
            gcp.check_gs(path)


def test_gcloud_gets_account_and_project(pin, monkeypatch):
    from tcr import vertex_batch
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.setdefault("cmd", cmd))
    vertex_batch.gcloud("storage", "cp", "in.jsonl", f"gs://{PROJECT}-tcr-batch/x/in.jsonl")
    assert seen["cmd"][-2:] == [f"--account={ACCOUNT}", f"--project={PROJECT}"]
    with pytest.raises(gcp.Refused):
        vertex_batch.gcloud("storage", "cp", "-r", "gs://valid-octagon-514423-n9-tcr-batch/o/*", "out")
