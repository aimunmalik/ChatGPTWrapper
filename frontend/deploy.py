"""Build the SPA and deploy it to S3 + invalidate CloudFront.

Reads the target bucket and CloudFront distribution ID from Terraform
outputs in `infra/envs/dev`. Requires:

  - AWS_PROFILE set (e.g. 'anna-chat') or default creds available
  - terraform + aws CLI on PATH
  - node + npm on PATH

Usage:
  python deploy.py            # build + deploy
  python deploy.py --no-build  # redeploy existing dist/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# On Windows, npm and aws are .cmd/.exe shims that Python subprocess can't
# resolve without shell=True. Everywhere else, keep shell=False.
SHELL = os.name == "nt"

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parent
TF_ENV = ROOT / "infra" / "envs" / "dev"
DIST = FRONTEND / "dist"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=SHELL,
    )
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def tf_output(key: str) -> str:
    result = subprocess.run(
        ["terraform", "output", "-raw", key],
        cwd=str(TF_ENV),
        capture_output=True,
        text=True,
        check=True,
        shell=SHELL,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-build", action="store_true", help="Skip npm build; reuse dist/.")
    args = parser.parse_args()

    print("[1/5] reading Terraform outputs")
    bucket = tf_output("spa_bucket_name")
    distribution = tf_output("cloudfront_distribution_id")
    cf_url = tf_output("cloudfront_url")
    print(f"       bucket       = {bucket}")
    print(f"       distribution = {distribution}")
    print(f"       url          = {cf_url}")

    if not args.no_build:
        print("[2/5] building frontend (npm run build)")
        run(["npm", "run", "build"], cwd=FRONTEND)
    else:
        print("[2/5] skipping build (--no-build)")
        if not DIST.is_dir():
            raise SystemExit(f"dist/ not found at {DIST}; run without --no-build first.")

    print("[3/5] uploading hashed assets with long-lived cache")
    assets_dir = DIST / "assets"
    if assets_dir.is_dir():
        run([
            "aws", "s3", "sync",
            str(assets_dir),
            f"s3://{bucket}/assets/",
            "--delete",
            "--cache-control", "public, max-age=31536000, immutable",
        ])

    print("[4/5] uploading root files with no-cache")
    for file in DIST.iterdir():
        if file.is_file():
            cache_control = "no-cache, must-revalidate"
            content_type = _content_type(file.name)
            run([
                "aws", "s3", "cp",
                str(file),
                f"s3://{bucket}/{file.name}",
                "--cache-control", cache_control,
                "--content-type", content_type,
            ])

    print("[5/5] creating CloudFront invalidation for /*")
    result = subprocess.run(
        [
            "aws", "cloudfront", "create-invalidation",
            "--distribution-id", distribution,
            "--paths", "/*",
        ],
        capture_output=True,
        text=True,
        check=True,
        shell=SHELL,
    )
    invalidation = json.loads(result.stdout)["Invalidation"]["Id"]
    print(f"       invalidation {invalidation} created (takes 1-5 min)")

    print()
    print(f"Deployed to {cf_url}")
    return 0


def _content_type(filename: str) -> str:
    if filename.endswith(".html"):
        return "text/html; charset=utf-8"
    if filename.endswith(".css"):
        return "text/css; charset=utf-8"
    if filename.endswith(".js"):
        return "application/javascript; charset=utf-8"
    if filename.endswith(".json"):
        return "application/json"
    if filename.endswith(".png"):
        return "image/png"
    if filename.endswith(".svg"):
        return "image/svg+xml"
    if filename.endswith(".ico"):
        return "image/x-icon"
    return "application/octet-stream"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
