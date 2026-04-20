"""Package the Lambda deployment zip.

Lambda runs on Amazon Linux 2 x86_64. When building on Windows or macOS we
must force pip to fetch manylinux wheels, not whatever the host uses. The zip
lands at backend/lambda.zip, which Terraform picks up.

Run with: python build.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "lambda_build"
SRC_PKG = ROOT / "src" / "anna_chat"
REQUIREMENTS = ROOT / "requirements.txt"
ZIP_BASENAME = ROOT / "lambda"
ZIP_PATH = ROOT / "lambda.zip"


def main() -> int:
    if not SRC_PKG.is_dir():
        print(f"error: source package missing at {SRC_PKG}", file=sys.stderr)
        return 1
    if not REQUIREMENTS.is_file():
        print(f"error: requirements.txt missing at {REQUIREMENTS}", file=sys.stderr)
        return 1

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print(f"[1/3] installing deps from {REQUIREMENTS.name} into {BUILD_DIR.name}/ (linux x86_64)")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--target",
            str(BUILD_DIR),
            "--platform",
            "manylinux2014_x86_64",
            "--implementation",
            "cp",
            "--python-version",
            "3.12",
            "--only-binary=:all:",
            "--upgrade",
            "-r",
            str(REQUIREMENTS),
        ],
        check=False,
    )
    if result.returncode != 0:
        return result.returncode

    print(f"[2/3] copying src/anna_chat/ into {BUILD_DIR.name}/anna_chat/")
    shutil.copytree(SRC_PKG, BUILD_DIR / "anna_chat")

    _strip_unneeded(BUILD_DIR)

    print(f"[3/3] zipping {BUILD_DIR.name}/ -> {ZIP_PATH.name}")
    shutil.make_archive(str(ZIP_BASENAME), "zip", BUILD_DIR)

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"built {ZIP_PATH.name} ({size_mb:.1f} MB)")
    return 0


def _strip_unneeded(root: Path) -> None:
    # These are safe to drop; they only serve development-time tooling.
    patterns = ["*.dist-info", "*.pyi", "__pycache__", "tests", "test"]
    count = 0
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                count += 1
            elif path.is_file():
                path.unlink(missing_ok=True)
                count += 1
    if count:
        print(f"       pruned {count} dev-time artifact(s)")


if __name__ == "__main__":
    raise SystemExit(main())
