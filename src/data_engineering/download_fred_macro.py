"""Download FRED macro series for US VAR/VECM modeling.

Uses the FRED REST API (free key from https://fred.stlouisfed.org/docs/api/api_key.html).
Set FRED_API_KEY env var before running.

Usage:
    python -m data_engineering.download_fred_macro
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .config import REPO_ROOT, ensure_parent, load_registry

FRED_BASE = "https://api.stlouisfed.org/fred"


def _fred_api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Set FRED_API_KEY environment variable. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "fed-prediction-agent/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_series(series_id: str, target_dir: Path, api_key: str) -> Path:
    """Download a single FRED series and save as JSON."""
    params = urllib.parse.urlencode({
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2000-01-01",
        "sort_order": "asc",
    })
    url = f"{FRED_BASE}/series/observations?{params}"
    data = _fetch_json(url)
    out_path = target_dir / f"{series_id}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def run() -> None:
    registry = load_registry()
    api_key = _fred_api_key()
    target_dir = ensure_parent(REPO_ROOT / "raw" / "market" / "fred" / "_placeholder")
    target_dir = REPO_ROOT / "raw" / "market" / "fred"
    target_dir.mkdir(parents=True, exist_ok=True)

    series_list = registry.get("fred_macro_series", [])
    print(f"Downloading {len(series_list)} FRED macro series...")
    for item in series_list:
        sid = item["id"]
        try:
            path = download_series(sid, target_dir, api_key)
            print(f"  OK {sid} ({item['name'][:50]}...) -> {path.name}")
        except Exception as exc:
            print(f"  ERROR {sid}: {exc}")
        time.sleep(0.3)  # FRED rate limit: 120/min

    # --- build an inventory manifest ---
    manifest = []
    for p in sorted(target_dir.glob("*.json")):
        manifest.append({"series_id": p.stem, "file": str(p.relative_to(REPO_ROOT))})
    manifest_path = target_dir / "fred_macro_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest → {manifest_path.relative_to(REPO_ROOT)} ({len(manifest)} series)")


if __name__ == "__main__":
    run()
