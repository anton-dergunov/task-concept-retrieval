"""Preflight for Vertex jobs: is everything pinned to the one account in .env?

Runs tcr/gcp.py's checks (shell overrides, ADC account and quota project, gcloud
configuration, project number), then shows whether billing and the Vertex AI API are on
and whether the batch bucket exists. `--ping` also sends one tiny Flash request (a fraction
of a cent) through the same client the jobs use. Exit code 0 only if every check passes.

    .venv/bin/python scripts/gcp_check.py [--ping]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tcr import gcp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ping", action="store_true", help="send one tiny Flash request")
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    pin = gcp.verify()           # raises gcp.Refused (exit 1, message) on any mismatch
    print(f"account   {pin.account}  (ADC file and gcloud agree)")
    print(f"project   {pin.project}  (number {pin.number})")
    print(f"ADC file  {pin.adc}")
    print(f"gcloud    configuration {pin.gcloud_config}")

    ok = True
    flags = gcp.gcloud_flags()
    r = gcp._gcloud(["billing", "projects", "describe", pin.project, *flags,
                     "--format=value(billingEnabled,billingAccountName)"], pin.gcloud_config)
    billing = r.stdout.split()
    print(f"billing   {' '.join(billing) or r.stderr.strip()[-200:]}")
    ok &= bool(billing) and billing[0] == "True"

    r = gcp._gcloud(["services", "list", "--enabled", *flags,
                     "--filter=config.name=aiplatform.googleapis.com", "--format=value(config.name)"],
                    pin.gcloud_config)
    api = r.stdout.strip() == "aiplatform.googleapis.com"
    print(f"Vertex AI API  {'enabled' if api else 'NOT enabled: ' + r.stderr.strip()[-200:]}")
    ok &= api

    r = gcp._gcloud(["storage", "buckets", "describe", f"gs://{pin.bucket}", *flags,
                     "--format=value(location)"], pin.gcloud_config)
    print(f"bucket    gs://{pin.bucket}: "
          f"{r.stdout.strip() if r.returncode == 0 else 'not there yet (the first batch job creates it)'}")

    if args.ping:
        from google import genai
        client = genai.Client(vertexai=True, project=pin.project, location="global",
                              credentials=pin.credentials)
        try:
            reply = client.models.generate_content(model=args.model, contents="Reply with: ok")
            print(f"ping      {args.model}: {(reply.text or '').strip()[:40]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"ping      FAILED: {exc}")
            ok = False

    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
