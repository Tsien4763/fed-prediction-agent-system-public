from __future__ import annotations

import hashlib
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def stream_download(
    url: str,
    target: Path,
    timeout: int = 120,
    chunk_size: int = 1024 * 1024,
    retries: int = 2,
) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".part")
    last_error = ""

    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "*/*",
                },
            )
            sha256 = hashlib.sha256()
            bytes_written = 0
            started = time.time()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                final_url = response.geturl()
                content_type = response.headers.get("Content-Type", "")
                expected_length = response.headers.get("Content-Length")
                with tmp_target.open("wb") as fh:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        fh.write(chunk)
                        sha256.update(chunk)
                        bytes_written += len(chunk)
            if expected_length and expected_length.isdigit() and bytes_written != int(expected_length):
                raise OSError(
                    f"incomplete download: got {bytes_written} bytes, "
                    f"expected {expected_length}"
                )
            tmp_target.replace(target)
            return {
                "ok": True,
                "status": status,
                "url": url,
                "final_url": final_url,
                "target_path": str(target),
                "bytes": bytes_written,
                "sha256": sha256.hexdigest(),
                "content_type": content_type,
                "elapsed_sec": round(time.time() - started, 2),
                "error": "",
            }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            if tmp_target.exists():
                tmp_target.unlink()
            if attempt < retries:
                time.sleep(2 + attempt)

    return {
        "ok": False,
        "status": "",
        "url": url,
        "final_url": "",
        "target_path": str(target),
        "bytes": 0,
        "sha256": "",
        "content_type": "",
        "elapsed_sec": "",
        "error": last_error,
    }


def ranged_download(
    url: str,
    target: Path,
    chunk_size: int,
    timeout: int = 120,
    retries: int = 4,
) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".part")
    if tmp_target.exists():
        tmp_target.unlink()

    total_bytes: int | None = None
    bytes_written = 0
    sha256 = hashlib.sha256()
    started = time.time()
    last_error = ""

    def fetch_range(start: int, end: int | None) -> tuple[bytes, int]:
        range_value = f"bytes={start}-{'' if end is None else end}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                "Range": range_value,
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 206:
                raise OSError(f"expected HTTP 206 for range {range_value}, got {status}")
            content_range = response.headers.get("Content-Range", "")
            match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+)", content_range)
            if not match:
                raise OSError(f"missing or invalid Content-Range: {content_range}")
            response_start, response_end, response_total = map(int, match.groups())
            if response_start != start:
                raise OSError(f"range start mismatch: got {response_start}, expected {start}")
            expected_length = response_end - response_start + 1
            data = response.read()
            if len(data) != expected_length:
                raise OSError(
                    f"incomplete range {range_value}: got {len(data)} bytes, expected {expected_length}"
                )
            return data, response_total

    while total_bytes is None or bytes_written < total_bytes:
        end = bytes_written + chunk_size - 1
        if total_bytes is not None:
            end = min(end, total_bytes - 1)
        for attempt in range(retries + 1):
            try:
                data, total = fetch_range(bytes_written, end)
                total_bytes = total
                with tmp_target.open("ab") as fh:
                    fh.write(data)
                sha256.update(data)
                bytes_written += len(data)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
                if attempt >= retries:
                    if tmp_target.exists():
                        tmp_target.unlink()
                    return {
                        "ok": False,
                        "status": "",
                        "url": url,
                        "final_url": "",
                        "target_path": str(target),
                        "bytes": bytes_written,
                        "sha256": "",
                        "content_type": "",
                        "elapsed_sec": round(time.time() - started, 2),
                        "error": last_error,
                    }
                time.sleep(1 + attempt)

    tmp_target.replace(target)
    return {
        "ok": True,
        "status": "206",
        "url": url,
        "final_url": "",
        "target_path": str(target),
        "bytes": bytes_written,
        "sha256": sha256.hexdigest(),
        "content_type": "application/octet-stream",
        "elapsed_sec": round(time.time() - started, 2),
        "error": "",
    }


def validate_file(path: Path, min_bytes: int = 0, validate_zip: bool = False) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if min_bytes and size < min_bytes:
        return False, f"too small: {size} < {min_bytes}"
    if validate_zip and not zipfile.is_zipfile(path):
        return False, "invalid zip"
    if validate_zip:
        try:
            with zipfile.ZipFile(path) as zf:
                bad_member = zf.testzip()
            if bad_member:
                return False, f"corrupt zip member: {bad_member}"
        except Exception as exc:
            return False, f"zip validation failed: {exc}"
    return True, ""
