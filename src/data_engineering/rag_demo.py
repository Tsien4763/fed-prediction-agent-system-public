from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import date

from .config import load_registry


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_index() -> list[dict[str, object]]:
    registry = load_registry()
    path = registry["outputs"]["rag_index"]
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def search(as_of: str, query: str, country: str | None = None, top_k: int = 8) -> list[dict[str, object]]:
    as_of_date = date.fromisoformat(as_of)
    rows = []
    for item in load_index():
        item_date = str(item.get("date") or "")
        if not item_date:
            continue
        try:
            if date.fromisoformat(item_date) > as_of_date:
                continue
        except ValueError:
            continue
        if country and str(item.get("country")) != country:
            continue
        rows.append(item)

    query_tokens = Counter(tokenize(query))
    scored = []
    for item in rows:
        text_tokens = Counter(tokenize(str(item.get("search_text") or item.get("text", ""))))
        overlap = sum(min(count, text_tokens[token]) for token, count in query_tokens.items())
        recency_bonus = 0.0
        try:
            age_days = (as_of_date - date.fromisoformat(str(item["date"]))).days
            recency_bonus = 2.0 / math.sqrt((max(age_days, 0) / 30.0) + 1.0)
        except Exception:
            pass
        score = overlap + recency_bonus
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG demo: query as-of-visible policy context.")
    parser.add_argument("--as-of", required=True, help="As-of date YYYY-MM-DD.")
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--country", help="Optional ISO3 country filter, e.g. USA.")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    results = search(args.as_of, args.query, args.country, args.top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
