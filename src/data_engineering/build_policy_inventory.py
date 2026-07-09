from __future__ import annotations

import argparse
import csv
import html
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from .config import ensure_parent, load_registry
from .http_utils import fetch_bytes

csv.field_size_limit(1024 * 1024 * 1024)


DATE_PATTERNS = [
    re.compile(r"(?P<year>20\d{2}|19\d{2})(?P<month>\d{2})(?P<day>\d{2})"),
    re.compile(r"(?P<year>20\d{2}|19\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"),
    re.compile(
        r"(?P<month_name>January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(?P<day>\d{1,2}),\s+(?P<year>20\d{2}|19\d{2})",
        re.I,
    ),
]

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

DOC_KEYWORDS = {
    "minutes": "minutes",
    "statement": "statement",
    "press release": "statement",
    "monetary policy summary": "decision",
    "rate": "decision",
    "key rate": "decision",
    "speech": "speech",
    "testimony": "testimony",
    "report": "report",
    "projection": "projection",
    "sep": "projection",
    "transcript": "transcript",
    "press conference": "press_conference",
    "open market": "operation",
    "reserve requirement": "operation",
    "lpr": "operation",
    "mlf": "operation",
}

COMMON_NAV_TITLES = {
    "skip to main content",
    "back to home",
    "subscribe to rss",
    "subscribe to email",
    "recent postings",
    "calendar",
    "publications",
    "site map",
    "a-z index",
    "careers",
    "faqs",
    "videos",
    "contact",
    "advanced",
}

ACTOR_PATTERNS = [
    ("CHN", "People's Bank of China", ["people's bank of china", "people’s bank of china", "pboc", "pbc"]),
    ("GBR", "Bank of England", ["bank of england", "monetary policy committee", " mpc "]),
    ("USA", "Federal Reserve", ["federal reserve", "fomc", "board of governors"]),
    (
        "RUS",
        "Bank of Russia",
        ["bank of russia", "central bank of the russian federation", "central bank of russia"],
    ),
    ("FRA", "Banque de France", ["banque de france", "bank of france"]),
    ("FRA", "European Central Bank", ["european central bank", " ecb "]),
]


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href")
            if href:
                self._href = href
                self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href:
            text = " ".join(" ".join(self._text).split())
            if text:
                self.links.append({"href": self._href, "text": text})
            self._href = None
            self._text = []


@dataclass
class InventoryRow:
    source_id: str
    country: str
    actor: str
    doc_type: str
    date: str
    year: int | None
    title: str
    url: str
    local_path: str
    provenance: str
    raw_metadata: str


def infer_date(text: str, fallback_year: int | None = None) -> tuple[str, int | None]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groupdict()
        year = int(groups["year"])
        if groups.get("month_name"):
            month = MONTHS[groups["month_name"].lower()]
        else:
            month = int(groups["month"])
        day = int(groups["day"])
        try:
            return datetime(year, month, day).date().isoformat(), year
        except ValueError:
            continue
    if fallback_year:
        return f"{fallback_year}-01-01", fallback_year
    return "", None


def infer_doc_type(text: str, source_type: str = "") -> str:
    lowered = text.lower()
    source_lowered = source_type.lower()
    if "reserve_requirement" in source_lowered:
        return "operation"
    if "open_market" in source_lowered or "operation" in source_lowered:
        return "operation"
    if "interest_rate" in source_lowered or "key_rate" in source_lowered:
        return "decision"
    if "report" in source_lowered:
        return "report"
    if "committee" in source_lowered or "meeting" in source_lowered:
        return "minutes"
    if re.search(r"monetary20\d{6}a", lowered):
        return "statement"
    if "fomcminutes" in lowered:
        return "minutes"
    if "fomcproj" in lowered or "projection" in lowered:
        return "projection"
    if "fomcpresconf" in lowered or "press conference" in lowered:
        return "press_conference"
    for keyword, doc_type in DOC_KEYWORDS.items():
        if keyword in lowered:
            return doc_type
    return "unknown"


def decode_response(raw: bytes) -> str:
    head = raw[:4096].decode("ascii", errors="ignore").lower()
    match = re.search(r"charset=['\"]?([a-z0-9_-]+)", head)
    encodings = []
    if match:
        encodings.append(match.group(1))
    encodings.extend(["utf-8", "gb18030", "gbk"])
    seen = set()
    for encoding in encodings:
        if encoding in seen:
            continue
        seen.add(encoding)
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def infer_actor_country(text: str, default_country: str = "MULTI", default_actor: str = "BIS") -> tuple[str, str]:
    padded = f" {text.lower()} "
    for country, actor, patterns in ACTOR_PATTERNS:
        if any(pattern in padded for pattern in patterns):
            return country, actor
    return default_country, default_actor


def is_valid_link_for_source(source: dict[str, object], url: str, text: str, doc_type: str) -> bool:
    source_id = str(source.get("id", ""))
    lowered_url = url.lower()
    if source_id in {"fed_fomc_historical", "fed_fomc_recent"}:
        return any(
            pattern in lowered_url
            for pattern in [
                "fomcminutes",
                "pressreleases/monetary20",
                "files/monetary20",
                "fomcprojtabl",
                "fomcpresconf",
                "fomc20",
            ]
        )
    return doc_type != "unknown"


def normalize_title(text: str, url: str, doc_type: str, date: str) -> str:
    short = text.strip()
    lowered_url = url.lower()
    suffix = ""
    if short.lower() in {"html", "pdf"}:
        suffix = f" ({short.upper()})"
    if "fomcminutes" in lowered_url:
        return f"FOMC minutes {date or ''}{suffix}".strip()
    if "pressreleases/monetary20" in lowered_url or "files/monetary20" in lowered_url:
        if "implementation" in short.lower() or lowered_url.endswith("b.htm"):
            return f"FOMC implementation note {date or ''}{suffix}".strip()
        return f"FOMC statement {date or ''}{suffix}".strip()
    if "fomcpresconf" in lowered_url:
        return f"FOMC press conference {date or ''}{suffix}".strip()
    if "meeting.pdf" in lowered_url:
        return f"FOMC transcript {date or ''}{suffix}".strip()
    if "sep" in lowered_url or "fomcprojtabl" in lowered_url or "fomcprojtable" in lowered_url:
        return f"FOMC SEP {date or ''}{suffix}".strip()
    return short


def parse_html_links(source: dict[str, object], url: str, fallback_year: int | None = None) -> list[InventoryRow]:
    try:
        html_text = decode_response(fetch_bytes(url, timeout=45))
    except Exception as exc:
        return [
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type="fetch_error",
                date="",
                year=fallback_year,
                title=f"FETCH ERROR: {exc}",
                url=url,
                local_path="",
                provenance="official_html_index",
                raw_metadata=json.dumps({"error": str(exc)}, ensure_ascii=False),
            )
        ]

    parser = LinkParser()
    parser.feed(html_text)
    rows: list[InventoryRow] = []
    for link in parser.links:
        href = link["href"]
        text = link["text"]
        if text.strip().lower() in COMMON_NAV_TITLES:
            continue
        absolute = urljoin(url, href)
        candidate = f"{text} {href}"
        doc_type = infer_doc_type(candidate, str(source.get("type", "")))
        if doc_type == "unknown":
            continue
        if not is_valid_link_for_source(source, absolute, text, doc_type):
            continue
        date, year = infer_date(candidate, fallback_year=fallback_year)
        title = normalize_title(text, absolute, doc_type, date)
        rows.append(
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type=doc_type,
                date=date,
                year=year,
                title=title,
                url=absolute,
                local_path="",
                provenance="official_html_index",
                raw_metadata=json.dumps({"href": href}, ensure_ascii=False),
            )
        )
    return rows


def parse_pboc_list_pages(source: dict[str, object]) -> list[InventoryRow]:
    first_url = str(source["url"])
    rows: list[InventoryRow] = []
    seen_urls: set[str] = set()
    page_urls = [first_url]

    try:
        first_html = decode_response(fetch_bytes(first_url, timeout=60))
    except Exception as exc:
        return [
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type="fetch_error",
                date="",
                year=None,
                title=f"FETCH ERROR: {exc}",
                url=first_url,
                local_path="",
                provenance="official_pboc_list",
                raw_metadata=json.dumps({"error": str(exc)}, ensure_ascii=False),
            )
        ]

    page_urls.extend(_pboc_pagination_urls(first_url, first_html))
    for url in page_urls:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            page_html = first_html if url == first_url else decode_response(fetch_bytes(url, timeout=60))
        except Exception:
            continue
        for href, title, date_text in _extract_pboc_items(page_html):
            absolute = urljoin(first_url, href)
            date, year = infer_date(date_text)
            if not year:
                continue
            rows.append(
                InventoryRow(
                    source_id=str(source["id"]),
                    country=str(source.get("country", "")),
                    actor=str(source.get("actor", "")),
                    doc_type=infer_doc_type(title, str(source.get("type", ""))),
                    date=date,
                    year=year,
                    title=html.unescape(" ".join(title.split())),
                    url=absolute,
                    local_path="",
                    provenance="official_pboc_list",
                    raw_metadata=json.dumps({"page_url": url}, ensure_ascii=False),
                )
            )
    return rows


def _pboc_pagination_urls(first_url: str, page_html: str) -> list[str]:
    match = re.search(r"tagname=\"([^\"]+?-(\d+)\.html)\"[^>]*>\s*尾页", page_html)
    if not match:
        match = re.search(r"queryArticleByCondition\(this,'([^']+?-(\d+)\.html)'\)[^>]*>\s*尾页", page_html)
    if not match:
        return []
    last_path = match.group(1)
    last_page = int(match.group(2))
    prefix = re.sub(r"-\d+\.html$", "", last_path)
    return [urljoin(first_url, f"{prefix}-{page}.html") for page in range(2, last_page + 1)]


def _extract_pboc_items(page_html: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(
        r"<a\s+[^>]*href=\"(?P<href>[^\"]+)\"[^>]*title=\"(?P<title>[^\"]*)\"[^>]*>(?P<inner>.*?)</a>\s*"
        r"(?:</font>)?\s*<span\s+class=\"hui12\">(?P<date>\d{4}-\d{2}-\d{2})</span>",
        re.S,
    )
    items = []
    for match in pattern.finditer(page_html):
        title = html.unescape(match.group("title")).strip()
        if title.lower() in {"", "true", "false"}:
            title = re.sub(r"<[^>]+>", " ", match.group("inner"))
            title = html.unescape(" ".join(title.split()))
        items.append((match.group("href"), title, match.group("date")))
    return items


def _parse_number(value: object) -> float | None:
    text = str(value).replace(",", "").strip()
    if not text or text == "-" or text.lower() == "nan":
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def _direction(current: float, previous: float | None) -> str:
    if previous is None:
        return "initial"
    if current > previous:
        return "increased"
    if current < previous:
        return "decreased"
    return "held"


def parse_boe_bank_rate_series(source: dict[str, object]) -> list[InventoryRow]:
    url = str(source["url"])
    try:
        page_html = decode_response(fetch_bytes(url, timeout=90))
    except Exception as exc:
        return [
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type="fetch_error",
                date="",
                year=None,
                title=f"FETCH ERROR: {exc}",
                url=url,
                local_path="",
                provenance="official_boe_bank_rate",
                raw_metadata=json.dumps({"error": str(exc)}, ensure_ascii=False),
            )
        ]

    tables = pd.read_html(StringIO(page_html))
    rate_table = next((table for table in tables if {"Date Changed", "Rate"}.issubset(table.columns)), None)
    if rate_table is None:
        return []

    parsed_rows: list[tuple[str, int, float]] = []
    for record in rate_table.to_dict(orient="records"):
        rate = _parse_number(record.get("Rate"))
        if rate is None:
            continue
        try:
            parsed_date = datetime.strptime(str(record.get("Date Changed")).strip(), "%d %b %y").date()
        except ValueError:
            continue
        parsed_rows.append((parsed_date.isoformat(), parsed_date.year, rate))

    rows: list[InventoryRow] = []
    previous_rate: float | None = None
    for date, year, rate in sorted(parsed_rows):
        direction = _direction(rate, previous_rate)
        rows.append(
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type="decision",
                date=date,
                year=year,
                title=f"Bank of England Bank Rate {direction} to {rate:.2f}%",
                url=url,
                local_path="",
                provenance="official_boe_bank_rate_series",
                raw_metadata=json.dumps({"rate": rate, "previous_rate": previous_rate}, ensure_ascii=False),
            )
        )
        previous_rate = rate
    return rows


def parse_ecb_key_rate_series(source: dict[str, object]) -> list[InventoryRow]:
    url = str(source["url"])
    try:
        page_html = decode_response(fetch_bytes(url, timeout=90))
    except Exception as exc:
        return [
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type="fetch_error",
                date="",
                year=None,
                title=f"FETCH ERROR: {exc}",
                url=url,
                local_path="",
                provenance="official_ecb_key_rates",
                raw_metadata=json.dumps({"error": str(exc)}, ensure_ascii=False),
            )
        ]

    tables = pd.read_html(StringIO(page_html))
    if not tables:
        return []
    rate_table = tables[0]
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    parsed_rows: list[dict[str, object]] = []
    for _, record in rate_table.iterrows():
        year_match = re.search(r"\d{4}", str(record.iloc[0]))
        date_match = re.search(r"(\d{1,2})\s+([A-Za-z]{3})", str(record.iloc[1]))
        if not year_match or not date_match:
            continue
        year = int(year_match.group(0))
        month = month_map.get(date_match.group(2).lower())
        if not month:
            continue
        date = datetime(year, month, int(date_match.group(1))).date().isoformat()
        deposit = _parse_number(record.iloc[2])
        main_refi = _parse_number(record.iloc[3])
        marginal = _parse_number(record.iloc[5])
        policy_rate = main_refi if main_refi is not None else deposit
        if policy_rate is None:
            continue
        parsed_rows.append(
            {
                "date": date,
                "year": year,
                "deposit_facility": deposit,
                "main_refinancing_operations": main_refi,
                "marginal_lending_facility": marginal,
                "policy_rate": policy_rate,
            }
        )

    rows: list[InventoryRow] = []
    previous_rate: float | None = None
    for item in sorted(parsed_rows, key=lambda value: str(value["date"])):
        rate = float(item["policy_rate"])
        direction = _direction(rate, previous_rate)
        rows.append(
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type="decision",
                date=str(item["date"]),
                year=int(item["year"]),
                title=(
                    f"ECB key rates {direction}: deposit {item['deposit_facility']}%, "
                    f"main refinancing {item['main_refinancing_operations']}%, "
                    f"marginal lending {item['marginal_lending_facility']}%"
                ),
                url=url,
                local_path="",
                provenance="official_ecb_key_rate_series",
                raw_metadata=json.dumps(item, ensure_ascii=False),
            )
        )
        previous_rate = rate
    return rows


def parse_cbr_key_rate_series(source: dict[str, object]) -> list[InventoryRow]:
    url = str(source["url"])
    try:
        page_html = decode_response(fetch_bytes(url, timeout=90))
    except Exception as exc:
        return [
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type="fetch_error",
                date="",
                year=None,
                title=f"FETCH ERROR: {exc}",
                url=url,
                local_path="",
                provenance="official_cbr_key_rate",
                raw_metadata=json.dumps({"error": str(exc)}, ensure_ascii=False),
            )
        ]

    categories_match = re.search(r"\"categories\":\[(?P<dates>.*?)\]", page_html, re.S)
    data_match = re.search(r"\"data\":\[(?P<rates>[\d\.,\s]+)\]", page_html, re.S)
    if not categories_match or not data_match:
        return []
    date_strings = re.findall(r"\"(\d{2}\.\d{2}\.\d{4})\"", categories_match.group("dates"))
    rates = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", data_match.group("rates"))]
    rows: list[InventoryRow] = []
    previous_rate: float | None = None
    for date_text, rate in zip(date_strings, rates):
        day, month, year_text = date_text.split(".")
        date = f"{year_text}-{month}-{day}"
        year = int(year_text)
        if previous_rate is not None and rate == previous_rate:
            continue
        direction = "initial" if previous_rate is None else ("increased" if rate > previous_rate else "decreased")
        rows.append(
            InventoryRow(
                source_id=str(source["id"]),
                country=str(source.get("country", "")),
                actor=str(source.get("actor", "")),
                doc_type="decision",
                date=date,
                year=year,
                title=f"Bank of Russia key rate {direction} to {rate:.2f}%",
                url=url,
                local_path="",
                provenance="official_cbr_key_rate_series",
                raw_metadata=json.dumps({"rate": rate, "previous_rate": previous_rate}, ensure_ascii=False),
            )
        )
        previous_rate = rate
    return rows


def parse_zip_inventory(source: dict[str, object]) -> list[InventoryRow]:
    local_path = Path(str(source["local_path"]))
    if not local_path.exists():
        return []
    rows: list[InventoryRow] = []
    with zipfile.ZipFile(local_path) as zf:
        names = zf.namelist()
        csv_names = [name for name in names if name.lower().endswith(".csv")]
        if csv_names:
            rows.extend(_parse_bis_csv_members(zf, csv_names, source))
        if rows:
            return rows
        for name in names:
            if name.endswith("/"):
                continue
            date, year = infer_date(name)
            rows.append(
                InventoryRow(
                    source_id=str(source["id"]),
                    country="MULTI",
                    actor="BIS",
                    doc_type="speech",
                    date=date,
                    year=year,
                    title=Path(name).stem,
                    url="",
                    local_path=f"{local_path}!{name}",
                    provenance="bis_zip_member",
                    raw_metadata=json.dumps({"zip_member": name}, ensure_ascii=False),
                )
            )
    return rows


def _parse_bis_csv_members(
    zf: zipfile.ZipFile, csv_names: list[str], source: dict[str, object]
) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for name in csv_names:
        with zf.open(name) as fh:
            sample = fh.read(4096).decode("utf-8", errors="replace")
        delimiter = "|" if sample.count("|") > sample.count(",") else ","
        with zf.open(name) as fh:
            text_iter = (line.decode("utf-8", errors="replace") for line in fh)
            reader = csv.DictReader(text_iter, delimiter=delimiter)
            for record in reader:
                lower_record = {str(k).lower(): v for k, v in record.items()}
                date_text = (
                    lower_record.get("date")
                    or lower_record.get("published")
                    or lower_record.get("publication_date")
                    or lower_record.get("year")
                    or ""
                )
                title = lower_record.get("title") or lower_record.get("speech_title") or ""
                speaker = lower_record.get("speaker") or lower_record.get("author") or ""
                description = lower_record.get("description") or ""
                institution = (
                    lower_record.get("institution")
                    or lower_record.get("central_bank")
                    or lower_record.get("cb")
                    or lower_record.get("country")
                    or "BIS"
                )
                url = lower_record.get("url") or lower_record.get("link") or ""
                actor_text = f"{institution} {speaker} {title} {description}"
                country, actor = infer_actor_country(actor_text)
                date, year = infer_date(f"{date_text} {title}")
                metadata = {k: v for k, v in record.items() if str(k).lower() != "text"}
                metadata["text_chars"] = len(lower_record.get("text") or "")
                rows.append(
                    InventoryRow(
                        source_id=str(source["id"]),
                        country=country,
                        actor=actor,
                        doc_type="speech",
                        date=date,
                        year=year,
                        title=title or speaker or Path(name).stem,
                        url=url,
                        local_path=f"{source['local_path']}!{name}",
                        provenance="bis_zip_csv",
                        raw_metadata=json.dumps(metadata, ensure_ascii=False),
                    )
                )
    return rows


def build_policy_inventory(limit_sources: set[str] | None = None) -> pd.DataFrame:
    registry = load_registry()
    rows: list[InventoryRow] = []
    for source in registry["policy_sources"]:
        if limit_sources and source["id"] not in limit_sources:
            continue
        method = source["method"]
        if method == "html_index_yearly":
            years = source.get("years", {})
            for year in range(int(years["start"]), int(years["end"]) + 1):
                url = str(source["url_template"]).format(year=year)
                rows.extend(parse_html_links(source, url, fallback_year=year))
        elif method == "html_index":
            rows.extend(parse_html_links(source, str(source["url"])))
        elif method == "pboc_list_pages":
            rows.extend(parse_pboc_list_pages(source))
        elif method == "boe_bank_rate_series":
            rows.extend(parse_boe_bank_rate_series(source))
        elif method == "ecb_key_rate_series":
            rows.extend(parse_ecb_key_rate_series(source))
        elif method == "cbr_key_rate_series":
            rows.extend(parse_cbr_key_rate_series(source))
        elif method == "zip_inventory":
            rows.extend(parse_zip_inventory(source))

    df = pd.DataFrame([row.__dict__ for row in rows])
    if not df.empty:
        df = df.drop_duplicates(subset=["source_id", "title", "url", "local_path", "date"])
        df = df.sort_values(["country", "actor", "year", "date", "doc_type"], na_position="last")
    out_path = ensure_parent(registry["outputs"]["policy_document_inventory"])
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build five-power policy document inventory.")
    parser.add_argument("--sources", nargs="*", help="Optional policy source IDs to scan.")
    args = parser.parse_args()

    df = build_policy_inventory(set(args.sources) if args.sources else None)
    print(f"Wrote {len(df)} inventory rows")


if __name__ == "__main__":
    main()
