from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import urllib.robotparser
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urldefrag, urljoin, urlparse

from .config import REPO_ROOT, ensure_parent


@dataclass(frozen=True)
class PolicySourceSpec:
    source_id: str
    url: str
    parser: str = "generic_html"
    cadence: str = "daily"
    enabled: bool = True
    crawl_enabled: bool = True
    max_pages: int = 8
    max_depth: int = 1
    same_domain_only: bool = True
    allowed_url_patterns: list[str] = field(default_factory=list)
    denied_url_patterns: list[str] = field(default_factory=list)
    respect_robots_txt: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedPolicyDocument:
    source_id: str
    url: str
    title: str
    published_at: str | None
    fetched_at: str
    text: str
    text_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        *,
        source_id: str,
        url: str,
        title: str,
        text: str,
        published_at: str | None = None,
        fetched_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "NormalizedPolicyDocument":
        normalized = normalize_text(text)
        return cls(
            source_id=source_id,
            url=url,
            title=title.strip() or source_id,
            published_at=published_at,
            fetched_at=fetched_at or utc_now(),
            text=normalized,
            text_hash=hash_text(normalized),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ParserFn = Callable[[str, PolicySourceSpec], list[NormalizedPolicyDocument]]
HttpGetFn = Callable[[str], str]
PARSER_REGISTRY: dict[str, ParserFn] = {}


class TextExtractingHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.time_value: str | None = None
        self._tag_stack: list[str] = []
        self._capture_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag_stack.append(tag)
        self._capture_title = tag == "title"
        if tag == "time":
            attr_map = {key: value for key, value in attrs}
            self.time_value = attr_map.get("datetime") or self.time_value

    def handle_endtag(self, tag: str) -> None:
        if self._tag_stack:
            self._tag_stack.pop()
        if tag == "title":
            self._capture_title = False

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if not stripped:
            return
        if self._capture_title:
            self.title_parts.append(stripped)
        if self._tag_stack and self._tag_stack[-1] in {"p", "h1", "h2", "h3", "li", "td"}:
            self.text_parts.append(stripped)


class LinkExtractingHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = {key.lower(): value for key, value in attrs if value}
        href = attr_map.get("href")
        if not href:
            return
        absolute = urljoin(self.base_url, href)
        clean, _fragment = urldefrag(absolute)
        self.links.append(clean)


def register_parser(name: str, parser: ParserFn | None = None):
    def decorator(fn: ParserFn) -> ParserFn:
        PARSER_REGISTRY[name] = fn
        return fn

    if parser is not None:
        return decorator(parser)
    return decorator


def default_policy_sources() -> list[PolicySourceSpec]:
    return [
        PolicySourceSpec(
            source_id="fed_fomc_press_releases",
            url="https://www.federalreserve.gov/newsevents/pressreleases/monetary.htm",
            parser="generic_html",
            cadence="daily",
            allowed_url_patterns=[r"/newsevents/pressreleases/monetary\d{8}a\.htm$"],
            max_pages=12,
            max_depth=1,
            metadata={"institution": "Federal Reserve", "document_type": "fomc_statement"},
        ),
        PolicySourceSpec(
            source_id="fed_speeches",
            url="https://www.federalreserve.gov/newsevents/speech.htm",
            parser="generic_html",
            cadence="daily",
            allowed_url_patterns=[r"/newsevents/speech/.*\.htm$"],
            max_pages=12,
            max_depth=1,
            metadata={"institution": "Federal Reserve", "document_type": "speech"},
        ),
        PolicySourceSpec(
            source_id="bis_central_bank_speeches",
            url="https://www.bis.org/cbspeeches/",
            parser="generic_html",
            cadence="daily",
            allowed_url_patterns=[r"/review/.*\.htm$", r"/speeches/.*\.htm$"],
            max_pages=12,
            max_depth=1,
            metadata={"institution": "BIS", "document_type": "central_bank_speech"},
        ),
    ]


def load_policy_source_specs(path: str | Path | None = None) -> list[PolicySourceSpec]:
    resolved = Path(path) if path else REPO_ROOT / "configs" / "policy_sources.json"
    if not resolved.exists():
        return default_policy_sources()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    rows = payload.get("sources", payload if isinstance(payload, list) else [])
    return [PolicySourceSpec(**row) for row in rows]


def fetch_source(
    spec: PolicySourceSpec,
    *,
    http_get: HttpGetFn | None = None,
    fetched_at: str | None = None,
) -> list[NormalizedPolicyDocument]:
    crawl_result = crawl_policy_source(spec, http_get=http_get, fetched_at=fetched_at)
    return crawl_result["documents"]


def fetch_single_page(
    spec: PolicySourceSpec,
    *,
    http_get: HttpGetFn | None = None,
    fetched_at: str | None = None,
) -> list[NormalizedPolicyDocument]:
    if not spec.enabled:
        return []
    getter = http_get or default_http_get
    parser = PARSER_REGISTRY.get(spec.parser)
    if parser is None:
        raise KeyError(f"Unknown parser: {spec.parser}")
    html = getter(spec.url)
    docs = parser(html, spec)
    fetched = fetched_at or utc_now()
    return [
        NormalizedPolicyDocument.from_text(
            source_id=doc.source_id,
            url=doc.url,
            title=doc.title,
            published_at=doc.published_at,
            fetched_at=fetched,
            text=doc.text,
            metadata=doc.metadata,
        )
        for doc in docs
    ]


def crawl_policy_source(
    spec: PolicySourceSpec,
    *,
    http_get: HttpGetFn | None = None,
    fetched_at: str | None = None,
    robot_allowed: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Crawl a policy source with bounded BFS link discovery.

    The crawler starts from a registry seed URL, extracts links, filters them
    by domain and allow/deny patterns, and parses each accepted page into the
    normalized document schema. It is intentionally bounded by depth and page
    count so it can run safely in notebooks and CI smoke tests.
    """
    if not spec.enabled:
        return {"source_id": spec.source_id, "documents": [], "pages_fetched": 0, "visited_urls": []}
    if not spec.crawl_enabled:
        docs = fetch_single_page(spec, http_get=http_get, fetched_at=fetched_at)
        return {
            "source_id": spec.source_id,
            "documents": docs,
            "pages_fetched": len(docs),
            "visited_urls": [doc.url for doc in docs],
        }

    getter = http_get or default_http_get
    queue: deque[tuple[str, int]] = deque([(spec.url, 0)])
    visited: set[str] = set()
    documents: list[NormalizedPolicyDocument] = []
    fetch_errors: dict[str, str] = {}
    fetched = fetched_at or utc_now()

    while queue and len(visited) < max(1, spec.max_pages):
        url, depth = queue.popleft()
        url = canonicalize_url(url)
        if not should_visit_url(url, spec, depth, visited):
            continue
        if spec.respect_robots_txt and not is_robot_allowed(url, robot_allowed=robot_allowed):
            continue
        visited.add(url)
        try:
            html = getter(url)
        except Exception as exc:
            fetch_errors[url] = type(exc).__name__
            continue

        page_spec = replace(spec, url=url)
        documents.extend(fetch_single_page(page_spec, http_get=lambda _url, html=html: html, fetched_at=fetched))
        if depth >= spec.max_depth:
            continue
        for link in extract_links(html, url):
            clean = canonicalize_url(link)
            if should_visit_url(clean, spec, depth + 1, visited):
                queue.append((clean, depth + 1))

    return {
        "source_id": spec.source_id,
        "documents": documents,
        "pages_fetched": len(visited),
        "visited_urls": sorted(visited),
        "fetch_errors": fetch_errors,
    }


def incremental_fetch(
    specs: list[PolicySourceSpec] | None = None,
    *,
    store_path: str | Path | None = None,
    http_get: HttpGetFn | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    specs = specs or load_policy_source_specs()
    out_path = Path(store_path) if store_path else REPO_ROOT / "data" / "processed" / "policy_documents.jsonl"
    seen = load_seen_hashes(out_path)
    new_docs: list[NormalizedPolicyDocument] = []
    duplicate_docs = 0
    skipped_sources = 0

    for spec in specs:
        if not spec.enabled:
            skipped_sources += 1
            continue
        crawl_result = crawl_policy_source(spec, http_get=http_get)
        for doc in crawl_result["documents"]:
            if doc.text_hash in seen:
                duplicate_docs += 1
                continue
            seen.add(doc.text_hash)
            new_docs.append(doc)

    if new_docs and not dry_run:
        ensure_parent(out_path)
        with out_path.open("a", encoding="utf-8") as fh:
            for doc in new_docs:
                fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    return {
        "sources_checked": sum(1 for item in specs if item.enabled),
        "sources_skipped": skipped_sources,
        "new_documents": len(new_docs),
        "duplicate_documents": duplicate_docs,
        "store_path": str(out_path),
        "dry_run": dry_run,
        "documents": [doc.to_dict() for doc in new_docs],
    }


def extract_links(html: str, base_url: str) -> list[str]:
    parser = LinkExtractingHTMLParser(base_url)
    parser.feed(html)
    return parser.links


def canonicalize_url(url: str) -> str:
    clean, _fragment = urldefrag(url.strip())
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"}:
        return clean
    path = parsed.path or "/"
    normalized = parsed._replace(path=path, fragment="")
    return normalized.geturl()


def should_visit_url(url: str, spec: PolicySourceSpec, depth: int, visited: set[str]) -> bool:
    if not url or url in visited:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if depth > spec.max_depth:
        return False
    if spec.same_domain_only and urlparse(spec.url).netloc != parsed.netloc:
        return False
    if any(re.search(pattern, url) for pattern in spec.denied_url_patterns):
        return False
    if spec.allowed_url_patterns and url != spec.url:
        return any(re.search(pattern, url) for pattern in spec.allowed_url_patterns)
    return True


def is_robot_allowed(url: str, *, robot_allowed: Callable[[str], bool] | None = None) -> bool:
    if robot_allowed is not None:
        return bool(robot_allowed(url))
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception:
        return True
    return bool(parser.can_fetch("MAE-CPS/1.0", url))


@register_parser("generic_html")
def parse_generic_html(html: str, spec: PolicySourceSpec) -> list[NormalizedPolicyDocument]:
    parser = TextExtractingHTMLParser()
    parser.feed(html)
    title = " ".join(parser.title_parts).strip() or spec.source_id
    text = "\n".join(parser.text_parts).strip()
    if not text:
        text = strip_tags(html)
    published_at = parser.time_value or extract_date(html)
    return [
        NormalizedPolicyDocument.from_text(
            source_id=spec.source_id,
            url=spec.url,
            title=title,
            published_at=published_at,
            fetched_at=utc_now(),
            text=text,
            metadata=spec.metadata,
        )
    ]


def default_http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "MAE-CPS/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def load_seen_hashes(path: str | Path) -> set[str]:
    resolved = Path(path)
    if not resolved.exists():
        return set()
    hashes: set[str] = set()
    with resolved.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = payload.get("text_hash")
            if value:
                hashes.add(str(value))
    return hashes


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def strip_tags(html: str) -> str:
    return normalize_text(re.sub(r"<[^>]+>", " ", html))


def extract_date(text: str) -> str | None:
    match = re.search(r"\b(20\d{2}|19\d{2})[-/](\d{2})[-/](\d{2})\b", text)
    return match.group(0) if match else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
