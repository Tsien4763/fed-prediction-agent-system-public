from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import ensure_parent, load_registry
from .http_utils import ranged_download, stream_download, validate_file


def download_bulk(ids: set[str] | None = None, skip_existing: bool = True) -> list[dict[str, object]]:
    registry = load_registry()
    results: list[dict[str, object]] = []

    for item in registry["bulk_downloads"]:
        if ids and item["id"] not in ids:
            continue
        if item.get("method") == "data360_api":
            results.append(
                {
                    "id": item["id"],
                    "ok": True,
                    "status": "handled_by_pipeline",
                    "url": item["url"],
                    "final_url": "",
                    "target_path": item["target_path"],
                    "bytes": "",
                    "sha256": "",
                    "content_type": "",
                    "elapsed_sec": "",
                    "error": "",
                }
            )
            continue
        target = ensure_parent(item["target_path"])
        min_bytes = int(item.get("min_bytes", 0) or 0)
        validate_zip = bool(item.get("validate_zip", False))
        target_ok, validation_error = validate_file(target, min_bytes=min_bytes, validate_zip=validate_zip)
        if skip_existing and target_ok:
            results.append(
                {
                    "id": item["id"],
                    "ok": True,
                    "status": "existing",
                    "url": item["url"],
                    "final_url": "",
                    "target_path": str(target),
                    "bytes": target.stat().st_size,
                    "sha256": "",
                    "content_type": "",
                    "elapsed_sec": 0,
                    "error": "",
                }
            )
            continue
        if item.get("range_chunk_bytes"):
            result = ranged_download(item["url"], target, int(item["range_chunk_bytes"]))
        else:
            result = stream_download(item["url"], target)
        target_ok, validation_error = validate_file(target, min_bytes=min_bytes, validate_zip=validate_zip)
        if not target_ok:
            result["ok"] = False
            result["error"] = result.get("error") or validation_error
        result["id"] = item["id"]
        results.append(result)

    status_path = ensure_parent(registry["outputs"]["download_status"])
    fieldnames = [
        "id",
        "ok",
        "status",
        "url",
        "final_url",
        "target_path",
        "bytes",
        "sha256",
        "content_type",
        "elapsed_sec",
        "error",
    ]
    with status_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WDI, WEO, and BIS bulk source files.")
    parser.add_argument("--ids", nargs="*", help="Optional source IDs to download.")
    parser.add_argument("--force", action="store_true", help="Redownload even if target files exist.")
    args = parser.parse_args()

    ids = set(args.ids) if args.ids else None
    results = download_bulk(ids=ids, skip_existing=not args.force)
    for row in results:
        marker = "OK" if row["ok"] else "FAIL"
        print(f"{marker} {row['id']} -> {row['target_path']} ({row.get('bytes', 0)} bytes)")
        if row.get("error"):
            print(f"  error: {row['error']}")


if __name__ == "__main__":
    main()
