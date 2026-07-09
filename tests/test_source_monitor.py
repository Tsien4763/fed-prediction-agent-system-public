from __future__ import annotations

from pathlib import Path

from data_engineering.source_monitor import PolicySourceSpec, crawl_policy_source, incremental_fetch


def test_crawler_normalizes_policy_documents_and_skips_duplicates(tmp_path: Path) -> None:
    pages = {
        "https://example.test/fomc": """
        <html>
          <head><title>FOMC index</title></head>
          <body>
            <a href="/fomc/20260617a.htm">June statement</a>
            <a href="/fomc/20260729a.htm">July statement</a>
            <a href="https://outside.test/ignore.htm">outside</a>
          </body>
        </html>
        """,
        "https://example.test/fomc/20260617a.htm": """
        <html><head><title>June FOMC statement</title></head>
          <body><time datetime="2026-06-17"></time>
          <article><p>Inflation remains elevated and policy remains data dependent.</p></article></body></html>
        """,
        "https://example.test/fomc/20260729a.htm": """
        <html><head><title>July FOMC statement</title></head>
          <body><time datetime="2026-07-29"></time>
          <article><p>The Committee will assess incoming data and risks.</p></article></body></html>
        """,
    }
    spec = PolicySourceSpec(
        source_id="fed_test",
        url="https://example.test/fomc",
        parser="generic_html",
        max_pages=5,
        max_depth=1,
        allowed_url_patterns=[r"/fomc/\d{8}a\.htm$"],
        metadata={"institution": "Federal Reserve"},
    )
    fake_get = lambda url: pages[url]

    crawled = crawl_policy_source(spec, http_get=fake_get)
    assert crawled["pages_fetched"] == 3
    assert len(crawled["documents"]) == 3
    assert "https://example.test/fomc/20260617a.htm" in crawled["visited_urls"]

    store = tmp_path / "policy_documents.jsonl"
    first = incremental_fetch([spec], store_path=store, http_get=fake_get)
    second = incremental_fetch([spec], store_path=store, http_get=fake_get)

    assert first["new_documents"] == 3
    assert second["duplicate_documents"] == 3
    doc = first["documents"][0]
    assert set(doc) >= {
        "source_id",
        "url",
        "title",
        "published_at",
        "fetched_at",
        "text",
        "text_hash",
        "metadata",
    }
    assert doc["metadata"]["institution"] == "Federal Reserve"

