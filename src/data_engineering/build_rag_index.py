from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import ensure_parent, load_registry


def build_rag_index() -> int:
    registry = load_registry()
    events = pd.read_parquet(registry["outputs"]["policy_events"])
    contexts = pd.read_parquet(registry["outputs"]["context_snapshots"])
    context_by_event = contexts.set_index("target_event_id").to_dict(orient="index") if not contexts.empty else {}
    out_path = ensure_parent(registry["outputs"]["rag_index"])
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in events.itertuples(index=False):
            ctx = context_by_event.get(row.event_id, {})
            text = "\n".join(
                [
                    f"date: {row.date}",
                    f"country: {row.country}",
                    f"actor: {row.actor}",
                    f"strategy: {row.strategy_key}",
                    f"title: {row.title}",
                    f"url: {row.url}",
                    f"own_prior: {ctx.get('own_previous_strategy_sequence', '[]')}",
                    f"other_p4_prior: {ctx.get('other_p4_previous_strategy_sequence', '[]')}",
                    f"macro_context: {ctx.get('macro_context', '{}')}",
                ]
            )
            record = {
                "doc_id": row.event_id,
                "date": row.date,
                "country": row.country,
                "actor": row.actor,
                "strategy_key": row.strategy_key,
                "title": row.title,
                "url": row.url,
                "search_text": " ".join(
                    [
                        str(row.date),
                        str(row.country),
                        str(row.actor),
                        str(row.strategy_key),
                        str(row.title),
                    ]
                ),
                "text": text,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    argparse.ArgumentParser(description="Build lightweight JSONL RAG index.").parse_args()
    count = build_rag_index()
    print(f"Wrote RAG index records: {count}")


if __name__ == "__main__":
    main()
