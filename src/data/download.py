"""Idempotent download + unzip of MovieLens datasets."""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from src.config import DATASET_URLS, RAW_DIR, ensure_dirs

CHUNK = 1 << 20  # 1 MiB


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[get ] {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r       {done/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:5.1f}%)", end="")
        print()


def _unzip(zip_path: Path, target_marker: Path) -> None:
    if target_marker.exists():
        print(f"[skip] {zip_path.stem} already extracted")
        return
    print(f"[unz ] {zip_path} -> {RAW_DIR}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)


def download_all() -> None:
    """Download + unzip both MovieLens datasets into data/raw/."""
    ensure_dirs()
    for name, url in DATASET_URLS.items():
        zip_path = RAW_DIR / f"{name}.zip"
        _download(url, zip_path)
        marker = RAW_DIR / name / ("ratings.dat" if name == "ml-1m" else "ratings.csv")
        _unzip(zip_path, marker)
        if not marker.exists():
            raise RuntimeError(f"Expected {marker} after unzip but it is missing.")
    print("[done] raw data ready in", RAW_DIR)
