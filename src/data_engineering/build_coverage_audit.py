from __future__ import annotations

import argparse

import pandas as pd

from .config import ensure_parent, load_registry


def build_coverage_audit() -> pd.DataFrame:
    registry = load_registry()
    inventory_path = registry["outputs"]["policy_document_inventory"]
    df = pd.read_parquet(inventory_path)
    start = registry["project"]["date_range"]["start_year"]
    end = registry["project"]["date_range"]["end_year"]
    countries = [item["code"] for item in registry["project"]["countries"]]

    clean = df.copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce").astype("Int64")
    clean = clean[clean["year"].between(start, end)]

    rows = []
    for country in countries:
        country_df = clean[(clean["country"] == country) | ((country == "FRA") & clean["actor"].str.contains("ECB|France", case=False, na=False))]
        for year in range(start, end + 1):
            year_df = country_df[country_df["year"] == year]
            row = {
                "country": country,
                "year": year,
                "total_documents": int(len(year_df)),
                "decisions": int(year_df["doc_type"].isin(["decision", "statement"]).sum()),
                "speeches": int((year_df["doc_type"] == "speech").sum()),
                "reports": int((year_df["doc_type"] == "report").sum()),
                "minutes": int((year_df["doc_type"] == "minutes").sum()),
                "operations": int((year_df["doc_type"] == "operation").sum()),
                "fetch_errors": int((year_df["doc_type"] == "fetch_error").sum()),
            }
            row["coverage_grade"] = "ok" if row["total_documents"] >= 5 else "thin"
            rows.append(row)
    audit = pd.DataFrame(rows)
    out_path = ensure_parent(registry["outputs"]["coverage_audit"])
    audit.to_csv(out_path, index=False)
    return audit


def main() -> None:
    argparse.ArgumentParser(description="Build policy inventory coverage audit.").parse_args()
    audit = build_coverage_audit()
    print(f"Wrote coverage audit rows: {len(audit)}")


if __name__ == "__main__":
    main()

