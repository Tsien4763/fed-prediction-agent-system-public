from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import repo_path


@dataclass
class RagRecord:
    doc_id: str
    date: str
    country: str
    actor: str
    strategy_key: str
    title: str
    url: str
    text: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RagRecord":
        return cls(
            doc_id=str(row.get("doc_id", "")),
            date=str(row.get("date", "")),
            country=str(row.get("country", "")),
            actor=str(row.get("actor", "")),
            strategy_key=str(row.get("strategy_key", "")),
            title=str(row.get("title", "")),
            url=str(row.get("url", "")),
            text=str(row.get("text") or row.get("search_text") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "date": self.date,
            "country": self.country,
            "actor": self.actor,
            "strategy_key": self.strategy_key,
            "title": self.title,
            "url": self.url,
            "text": self.text[:1200],
        }


def read_jsonl(path: str | Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    resolved = repo_path(path)
    if not resolved.exists():
        return
    with resolved.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any] | str], *, append: bool = False) -> int:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    mode = "a" if append else "w"
    with resolved.open(mode, encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, str):
                fh.write(row.rstrip("\n") + "\n")
            else:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


class RagIndex:
    def __init__(self, path: str | Path, scan_limit: int | None = None) -> None:
        self.path = repo_path(path)
        self.scan_limit = scan_limit
        self._records: list[RagRecord] | None = None

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in text.lower().replace("_", " ").replace("-", " ").split() if len(token) > 2}

    def search(
        self,
        query: str,
        *,
        country: str | None = None,
        before_date: str | None = None,
        top_k: int = 5,
    ) -> list[RagRecord]:
        query_tokens = self._tokens(query)
        scored: list[tuple[float, RagRecord]] = []
        for record in self.records:
            if country and record.country != country:
                continue
            if before_date and record.date > before_date:
                continue
            haystack = " ".join([record.country, record.actor, record.strategy_key, record.title])
            overlap = len(query_tokens & self._tokens(haystack))
            if overlap == 0:
                continue
            recency_bonus = 0.01 if before_date and record.date[:4] == before_date[:4] else 0.0
            scored.append((overlap + recency_bonus, record))
            scored.sort(key=lambda item: item[0], reverse=True)
            if len(scored) > top_k * 8:
                scored = scored[: top_k * 4]
        return [record for _, record in scored[:top_k]]

    @property
    def records(self) -> list[RagRecord]:
        if self._records is None:
            self._records = [RagRecord.from_dict(row) for row in read_jsonl(self.path, limit=self.scan_limit)]
        return self._records


def load_context_snapshots(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    resolved = repo_path(path)
    try:
        import pandas as pd

        df = pd.read_parquet(resolved)
        if limit is not None:
            df = df.head(limit)
        return df.to_dict(orient="records")
    except Exception:
        fallback = repo_path("data/index/policy_context_index.jsonl")
        rows = []
        for row in read_jsonl(fallback, limit=limit):
            rows.append(
                {
                    "snapshot_id": f"{row.get('doc_id')}_fallback_ctx",
                    "as_of_date": row.get("date"),
                    "target_event_id": row.get("doc_id"),
                    "target_country": row.get("country"),
                    "target_actor": row.get("actor"),
                    "target_strategy_key": row.get("strategy_key"),
                    "own_previous_strategy_sequence": "[]",
                    "other_p4_previous_strategy_sequence": "[]",
                    "macro_context": "{}",
                    "rag_text": row.get("search_text") or row.get("title") or "",
                }
            )
        return rows


def load_strategy_cards(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    return list(read_jsonl(path, limit=limit))
