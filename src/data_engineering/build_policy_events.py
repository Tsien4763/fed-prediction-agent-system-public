from __future__ import annotations

import argparse
import hashlib
import re

import pandas as pd

from .config import ensure_parent, load_registry


STANCE_KEYWORDS = {
    "tightening": [
        "increase",
        "increased",
        "raise",
        "raised",
        "hike",
        "restrictive",
        "inflation",
        "tightening",
        "vigilant",
    ],
    "easing": [
        "decrease",
        "decreased",
        "cut",
        "lower",
        "reduced",
        "accommodative",
        "stimulus",
        "easing",
        "support growth",
    ],
    "hold": [
        "maintain",
        "unchanged",
        "held",
        "hold",
        "pause",
        "kept",
    ],
}

INSTRUMENT_KEYWORDS = {
    "policy_rate": ["rate", "bank rate", "federal funds", "key rate", "lpr"],
    "asset_purchase": ["asset purchase", "qe", "quantitative easing", "purchase programme", "balance sheet"],
    "reserve_requirement": ["reserve requirement", "rrr"],
    "forward_guidance": ["forward guidance", "guidance", "communication"],
    "macroprudential": ["macroprudential", "financial stability", "capital buffer"],
}


def _classify_strategy(title: str, doc_type: str) -> tuple[str, str, str]:
    text = f"{title} {doc_type}".lower()
    scores = {
        stance: sum(1 for keyword in keywords if keyword in text)
        for stance, keywords in STANCE_KEYWORDS.items()
    }
    stance = max(scores, key=scores.get)
    if scores[stance] == 0:
        stance = "communication"

    instrument = "communication"
    for candidate, keywords in INSTRUMENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            instrument = candidate
            break

    if doc_type in {"speech", "testimony", "press_conference"}:
        action = "signal"
    elif stance == "tightening":
        action = "tighten"
    elif stance == "easing":
        action = "ease"
    elif stance == "hold":
        action = "hold"
    else:
        action = "describe"
    return stance, instrument, action


def _extract_magnitude_bp(title: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*(basis points|bp)", title, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d(?:\.\d+)?)\s*%", title)
    if match:
        return int(float(match.group(1)) * 100)
    return None


def _canonical_event_title(title: str) -> str:
    canonical = re.sub(r"\s+\((?:PDF|HTML)\)\s*$", "", title, flags=re.I)
    canonical = re.sub(r":\s*(?:PDF|HTML)\s*$", "", canonical, flags=re.I)
    return " ".join(canonical.split())


def _preferred_document_rank(row: pd.Series) -> int:
    url = str(row.get("url", "")).lower()
    title = str(row.get("title", "")).lower()
    if url.endswith(".htm") or url.endswith(".html") or "(html)" in title:
        return 0
    if url.endswith(".pdf") or "(pdf)" in title:
        return 1
    return 2


def build_policy_events() -> pd.DataFrame:
    registry = load_registry()
    inventory = pd.read_parquet(registry["outputs"]["policy_document_inventory"])
    start = registry["project"]["date_range"]["start_year"]
    end = registry["project"]["date_range"]["end_year"]
    countries = {item["code"] for item in registry["project"]["countries"]}
    inventory["year"] = pd.to_numeric(inventory["year"], errors="coerce").astype("Int64")
    inventory = inventory[inventory["year"].between(start, end)]
    inventory = inventory[inventory["country"].isin(countries)]
    inventory = inventory[inventory["doc_type"] != "fetch_error"].copy()
    inventory["canonical_event_title"] = inventory["title"].map(lambda value: _canonical_event_title(str(value)))
    inventory["document_rank"] = inventory.apply(_preferred_document_rank, axis=1)
    inventory = inventory.sort_values(["document_rank", "source_id", "url", "local_path"])
    inventory = inventory.drop_duplicates(
        subset=["date", "country", "actor", "doc_type", "canonical_event_title"],
        keep="first",
    )

    events = []
    for row in inventory.itertuples(index=False):
        stance, instrument, action = _classify_strategy(str(row.title), str(row.doc_type))
        stable = f"{row.source_id}|{row.date}|{row.actor}|{row.title}|{row.url}|{row.local_path}"
        event_id = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:16]
        events.append(
            {
                "event_id": event_id,
                "date": row.date,
                "year": row.year,
                "country": row.country,
                "actor": row.actor,
                "doc_type": row.doc_type,
                "instrument": instrument,
                "action": action,
                "stance": stance,
                "magnitude_bp": _extract_magnitude_bp(str(row.title)),
                "strategy_key": f"{instrument}.{action}.{stance}",
                "title": row.title,
                "url": row.url,
                "source_id": row.source_id,
                "provenance": row.provenance,
            }
        )
    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values(["date", "country", "actor", "doc_type"])
    out_path = ensure_parent(registry["outputs"]["policy_events"])
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    argparse.ArgumentParser(description="Extract normalized policy events from inventory.").parse_args()
    df = build_policy_events()
    print(f"Wrote policy events: {len(df)}")


if __name__ == "__main__":
    main()
