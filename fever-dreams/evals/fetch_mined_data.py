"""Download and unpack the Fever Dreams 449-shot mined eval dataset from GitHub Releases.

Keeps ~727 MB of PNG test images out of Git history while making the eval benchmark
100% reproducible on any machine with one command:

    uv run python evals/fetch_mined_data.py

To publish the local zip to GitHub Releases first:
    gh release create fever-dreams-eval-v1 evals/mined_data_v1.zip \
        --title "Fever Dreams Eval Dataset v1 (449 Human-Verified Shots)" \
        --notes "676 lossless PNGs (449 shots, 112 contact sheets, 115 canonical refs) + sidecars."
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
DATA_DIR = EVALS_DIR / "mined_data"
DEFAULT_REPO = "wblakecaldwell/weird-ai"
DEFAULT_TAG = "fever-dreams-eval-v1"
DEFAULT_ASSET = "mined_data_v1.zip"
EXPECTED_SHOTS = 449


def current_shot_count(data_dir: Path) -> int:
    shots_dir = data_dir / "shots"
    return len(list(shots_dir.glob("*.png"))) if shots_dir.exists() else 0


def download_with_progress(url: str, dest: Path) -> None:
    headers = {"User-Agent": "fever-dreams-eval-fetcher"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        chunk = 1024 * 1024  # 1 MB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if total > 0:
                pct = done * 100.0 / total
                print(f"\r  Downloading {dest.name}: {done / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r  Downloading {dest.name}: {done / 1e6:.1f} MB", end="", flush=True)
    print()


def fetch_dataset(*, repo: str, tag: str, asset: str, data_dir: Path, force: bool, keep_zip: bool) -> None:
    existing = current_shot_count(data_dir)
    if existing >= EXPECTED_SHOTS and not force:
        print(f"[fetch] Dataset already present in {data_dir} ({existing} shot PNGs). Use --force to re-download.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = EVALS_DIR / asset

    if not zip_path.exists() or force:
        url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
        print(f"[fetch] Fetching {url} ...")
        try:
            download_with_progress(url, zip_path)
        except Exception as exc:
            # Fallback to `gh release download` (handles private repos automatically)
            if shutil.which("gh"):
                print(f"[fetch] Direct HTTP failed ({exc}); trying `gh release download`...")
                subprocess.run(
                    ["gh", "release", "download", tag, "--repo", repo, "--pattern", asset, "--dir", str(EVALS_DIR), "--clobber"],
                    check=True,
                )
            else:
                raise

    # Preserve committed gold_manifest.json if it already exists locally
    manifest_p = data_dir / "gold_manifest.json"
    saved_manifest = manifest_p.read_bytes() if manifest_p.exists() else None

    print(f"[fetch] Extracting {zip_path.name} -> {data_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    if saved_manifest is not None:
        manifest_p.write_bytes(saved_manifest)

    if not keep_zip and zip_path.exists():
        zip_path.unlink()

    final_cnt = current_shot_count(data_dir)
    print(f"[fetch] Ready! {final_cnt} shot PNGs verified in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Fever Dreams mined eval PNG dataset from GitHub Releases.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download even if dataset already exists")
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded zip after extracting")
    args = parser.parse_args()

    fetch_dataset(
        repo=args.repo,
        tag=args.tag,
        asset=args.asset,
        data_dir=args.data_dir,
        force=args.force,
        keep_zip=args.keep_zip,
    )


if __name__ == "__main__":
    main()
