from __future__ import annotations

import argparse

from .build_context_snapshots import build_context_snapshots
from .build_coverage_audit import build_coverage_audit
from .build_policy_events import build_policy_events
from .build_policy_inventory import build_policy_inventory
from .build_rag_index import build_rag_index
from .build_strategy_cards import build_strategy_cards
from .build_weo_panel import build_weo_panel
from .build_wdi_top100 import build_wdi_top100
from .download_bulk import download_bulk


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data engineering MVP pipeline.")
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Build derived artifacts from existing raw files without downloading bulk files.",
    )
    parser.add_argument(
        "--download-ids",
        nargs="*",
        help="Optional bulk source IDs to download before building artifacts.",
    )
    args = parser.parse_args()

    if not args.skip_downloads:
        print("Downloading bulk sources...")
        results = download_bulk(ids=set(args.download_ids) if args.download_ids else None)
        for row in results:
            marker = "OK" if row["ok"] else "FAIL"
            print(f"  {marker} {row['id']}: {row.get('bytes', 0)} bytes")

    print("Building policy inventory...")
    inventory = build_policy_inventory()
    print(f"  inventory rows: {len(inventory)}")

    print("Building coverage audit...")
    audit = build_coverage_audit()
    print(f"  audit rows: {len(audit)}")

    print("Building policy events...")
    events = build_policy_events()
    print(f"  event rows: {len(events)}")

    print("Building WDI top 100 panel...")
    wdi_indicators, wdi_panel = build_wdi_top100()
    print(f"  WDI indicators: {len(wdi_indicators)}")
    print(f"  WDI panel rows: {len(wdi_panel)}")

    print("Building WEO panel...")
    weo_panel = build_weo_panel()
    print(f"  WEO panel rows: {len(weo_panel)}")

    print("Building strategy cards...")
    cards = build_strategy_cards()
    print(f"  strategy cards: {len(cards)}")

    print("Building context snapshots...")
    contexts = build_context_snapshots()
    print(f"  context snapshots: {len(contexts)}")

    print("Building RAG index...")
    index_count = build_rag_index()
    print(f"  RAG records: {index_count}")


if __name__ == "__main__":
    main()
