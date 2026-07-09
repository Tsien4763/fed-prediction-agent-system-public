"""Historical FOMC decision labels (2000-2026).

Each entry: (meeting_date, decision, ff_target_lower, ff_target_upper, description)
  decision: +1 = hike, 0 = hold, -1 = cut
  rates in percent

Source: Federal Reserve historical materials, FRED FEDFUNDS.
"""
from __future__ import annotations

import pandas as pd

# Major FOMC decisions since 2000, simplified to the dominant action per meeting.
# For multi-move meetings (rare), we use the net direction.
FOMC_LABELS: list[dict] = [
    # 2000
    {"date": "2000-02-02", "decision": 1, "ff_lower": 5.50, "ff_upper": 5.75, "desc": "hike 25bp"},
    {"date": "2000-03-21", "decision": 1, "ff_lower": 5.75, "ff_upper": 6.00, "desc": "hike 25bp"},
    {"date": "2000-05-16", "decision": 1, "ff_lower": 6.00, "ff_upper": 6.50, "desc": "hike 50bp"},
    {"date": "2000-06-28", "decision": 0, "ff_lower": 6.50, "ff_upper": 6.50, "desc": "hold"},
    {"date": "2000-08-22", "decision": 0, "ff_lower": 6.50, "ff_upper": 6.50, "desc": "hold"},
    {"date": "2000-10-03", "decision": 0, "ff_lower": 6.50, "ff_upper": 6.50, "desc": "hold"},
    {"date": "2000-11-15", "decision": 0, "ff_lower": 6.50, "ff_upper": 6.50, "desc": "hold"},
    {"date": "2000-12-19", "decision": 0, "ff_lower": 6.50, "ff_upper": 6.50, "desc": "hold"},

    # 2001 — easing cycle (dot-com bust + 9/11)
    {"date": "2001-01-03", "decision": -1, "ff_lower": 6.00, "ff_upper": 6.00, "desc": "cut 50bp (inter-meeting)"},
    {"date": "2001-01-31", "decision": -1, "ff_lower": 5.50, "ff_upper": 5.50, "desc": "cut 50bp"},
    {"date": "2001-03-20", "decision": -1, "ff_lower": 5.00, "ff_upper": 5.00, "desc": "cut 50bp"},
    {"date": "2001-04-18", "decision": -1, "ff_lower": 4.50, "ff_upper": 4.50, "desc": "cut 50bp (inter-meeting)"},
    {"date": "2001-05-15", "decision": -1, "ff_lower": 4.00, "ff_upper": 4.00, "desc": "cut 50bp"},
    {"date": "2001-06-27", "decision": -1, "ff_lower": 3.75, "ff_upper": 3.75, "desc": "cut 25bp"},
    {"date": "2001-08-21", "decision": -1, "ff_lower": 3.50, "ff_upper": 3.50, "desc": "cut 25bp"},
    {"date": "2001-09-17", "decision": -1, "ff_lower": 3.00, "ff_upper": 3.00, "desc": "cut 50bp (inter-meeting, post-9/11)"},
    {"date": "2001-10-02", "decision": -1, "ff_lower": 2.50, "ff_upper": 2.50, "desc": "cut 50bp"},
    {"date": "2001-11-06", "decision": -1, "ff_lower": 2.00, "ff_upper": 2.00, "desc": "cut 50bp"},
    {"date": "2001-12-11", "decision": -1, "ff_lower": 1.75, "ff_upper": 1.75, "desc": "cut 25bp"},

    # 2002 — hold at 1.75%
    {"date": "2002-01-30", "decision": 0, "ff_lower": 1.75, "ff_upper": 1.75, "desc": "hold"},
    {"date": "2002-03-19", "decision": 0, "ff_lower": 1.75, "ff_upper": 1.75, "desc": "hold"},
    {"date": "2002-05-07", "decision": 0, "ff_lower": 1.75, "ff_upper": 1.75, "desc": "hold"},
    {"date": "2002-06-26", "decision": 0, "ff_lower": 1.75, "ff_upper": 1.75, "desc": "hold"},
    {"date": "2002-08-13", "decision": 0, "ff_lower": 1.75, "ff_upper": 1.75, "desc": "hold"},
    {"date": "2002-09-24", "decision": 0, "ff_lower": 1.75, "ff_upper": 1.75, "desc": "hold"},
    {"date": "2002-11-06", "decision": -1, "ff_lower": 1.25, "ff_upper": 1.25, "desc": "cut 50bp"},
    {"date": "2002-12-10", "decision": 0, "ff_lower": 1.25, "ff_upper": 1.25, "desc": "hold"},

    # 2003-2004 — hold at 1.00%
    {"date": "2003-01-29", "decision": 0, "ff_lower": 1.25, "ff_upper": 1.25, "desc": "hold"},
    {"date": "2003-06-25", "decision": -1, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "cut 25bp"},
    {"date": "2003-08-12", "decision": 0, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "hold"},
    {"date": "2003-09-16", "decision": 0, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "hold"},
    {"date": "2003-10-28", "decision": 0, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "hold"},
    {"date": "2003-12-09", "decision": 0, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "hold"},
    {"date": "2004-01-28", "decision": 0, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "hold"},
    {"date": "2004-03-16", "decision": 0, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "hold"},
    {"date": "2004-05-04", "decision": 0, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "hold"},

    # 2004-2006 — tightening cycle
    {"date": "2004-06-30", "decision": 1, "ff_lower": 1.25, "ff_upper": 1.25, "desc": "hike 25bp"},
    {"date": "2004-08-10", "decision": 1, "ff_lower": 1.50, "ff_upper": 1.50, "desc": "hike 25bp"},
    {"date": "2004-09-21", "decision": 1, "ff_lower": 1.75, "ff_upper": 1.75, "desc": "hike 25bp"},
    {"date": "2004-11-10", "decision": 1, "ff_lower": 2.00, "ff_upper": 2.00, "desc": "hike 25bp"},
    {"date": "2004-12-14", "decision": 1, "ff_lower": 2.25, "ff_upper": 2.25, "desc": "hike 25bp"},
    # 2005 — 8 hikes of 25bp each
    {"date": "2005-02-02", "decision": 1, "ff_lower": 2.50, "ff_upper": 2.50, "desc": "hike 25bp"},
    {"date": "2005-03-22", "decision": 1, "ff_lower": 2.75, "ff_upper": 2.75, "desc": "hike 25bp"},
    {"date": "2005-05-03", "decision": 1, "ff_lower": 3.00, "ff_upper": 3.00, "desc": "hike 25bp"},
    {"date": "2005-06-30", "decision": 1, "ff_lower": 3.25, "ff_upper": 3.25, "desc": "hike 25bp"},
    {"date": "2005-08-09", "decision": 1, "ff_lower": 3.50, "ff_upper": 3.50, "desc": "hike 25bp"},
    {"date": "2005-09-20", "decision": 1, "ff_lower": 3.75, "ff_upper": 3.75, "desc": "hike 25bp"},
    {"date": "2005-11-01", "decision": 1, "ff_lower": 4.00, "ff_upper": 4.00, "desc": "hike 25bp"},
    {"date": "2005-12-13", "decision": 1, "ff_lower": 4.25, "ff_upper": 4.25, "desc": "hike 25bp"},
    # 2006
    {"date": "2006-01-31", "decision": 1, "ff_lower": 4.50, "ff_upper": 4.50, "desc": "hike 25bp"},
    {"date": "2006-03-28", "decision": 1, "ff_lower": 4.75, "ff_upper": 4.75, "desc": "hike 25bp"},
    {"date": "2006-05-10", "decision": 1, "ff_lower": 5.00, "ff_upper": 5.00, "desc": "hike 25bp"},
    {"date": "2006-06-29", "decision": 1, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hike 25bp"},
    {"date": "2006-08-08", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},
    {"date": "2006-09-20", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},
    {"date": "2006-10-25", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},
    {"date": "2006-12-12", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},

    # 2007 — easing begins
    {"date": "2007-01-31", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},
    {"date": "2007-03-21", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},
    {"date": "2007-05-09", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},
    {"date": "2007-06-28", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},
    {"date": "2007-08-07", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.25, "desc": "hold"},
    {"date": "2007-09-18", "decision": -1, "ff_lower": 4.75, "ff_upper": 4.75, "desc": "cut 50bp"},
    {"date": "2007-10-31", "decision": -1, "ff_lower": 4.50, "ff_upper": 4.50, "desc": "cut 25bp"},
    {"date": "2007-12-11", "decision": -1, "ff_lower": 4.25, "ff_upper": 4.25, "desc": "cut 25bp"},

    # 2008 — GFC
    {"date": "2008-01-22", "decision": -1, "ff_lower": 3.50, "ff_upper": 3.50, "desc": "cut 75bp (inter-meeting)"},
    {"date": "2008-01-30", "decision": -1, "ff_lower": 3.00, "ff_upper": 3.00, "desc": "cut 50bp"},
    {"date": "2008-03-18", "decision": -1, "ff_lower": 2.25, "ff_upper": 2.25, "desc": "cut 75bp"},
    {"date": "2008-04-30", "decision": -1, "ff_lower": 2.00, "ff_upper": 2.00, "desc": "cut 25bp"},
    {"date": "2008-06-25", "decision": 0, "ff_lower": 2.00, "ff_upper": 2.00, "desc": "hold"},
    {"date": "2008-08-05", "decision": 0, "ff_lower": 2.00, "ff_upper": 2.00, "desc": "hold"},
    {"date": "2008-09-16", "decision": 0, "ff_lower": 2.00, "ff_upper": 2.00, "desc": "hold (Lehman weekend)"},
    {"date": "2008-10-08", "decision": -1, "ff_lower": 1.50, "ff_upper": 1.50, "desc": "cut 50bp (coordinated)"},
    {"date": "2008-10-29", "decision": -1, "ff_lower": 1.00, "ff_upper": 1.00, "desc": "cut 50bp"},
    {"date": "2008-12-16", "decision": -1, "ff_lower": 0.00, "ff_upper": 0.25, "desc": "cut to ZLB"},

    # 2009-2015 — ZIRP
    {"date": "2009-01-28", "decision": 0, "ff_lower": 0.00, "ff_upper": 0.25, "desc": "hold (ZIRP)"},
    {"date": "2015-12-16", "decision": 1, "ff_lower": 0.25, "ff_upper": 0.50, "desc": "hike 25bp (liftoff)"},

    # 2016-2018 — slow normalization
    {"date": "2016-12-14", "decision": 1, "ff_lower": 0.50, "ff_upper": 0.75, "desc": "hike 25bp"},
    {"date": "2017-03-15", "decision": 1, "ff_lower": 0.75, "ff_upper": 1.00, "desc": "hike 25bp"},
    {"date": "2017-06-14", "decision": 1, "ff_lower": 1.00, "ff_upper": 1.25, "desc": "hike 25bp"},
    {"date": "2017-12-13", "decision": 1, "ff_lower": 1.25, "ff_upper": 1.50, "desc": "hike 25bp"},
    {"date": "2018-03-21", "decision": 1, "ff_lower": 1.50, "ff_upper": 1.75, "desc": "hike 25bp"},
    {"date": "2018-06-13", "decision": 1, "ff_lower": 1.75, "ff_upper": 2.00, "desc": "hike 25bp"},
    {"date": "2018-09-26", "decision": 1, "ff_lower": 2.00, "ff_upper": 2.25, "desc": "hike 25bp"},
    {"date": "2018-12-19", "decision": 1, "ff_lower": 2.25, "ff_upper": 2.50, "desc": "hike 25bp"},

    # 2019 — easing
    {"date": "2019-07-31", "decision": -1, "ff_lower": 2.00, "ff_upper": 2.25, "desc": "cut 25bp"},
    {"date": "2019-09-18", "decision": -1, "ff_lower": 1.75, "ff_upper": 2.00, "desc": "cut 25bp"},
    {"date": "2019-10-30", "decision": -1, "ff_lower": 1.50, "ff_upper": 1.75, "desc": "cut 25bp"},

    # 2020 — COVID emergency cuts
    {"date": "2020-03-03", "decision": -1, "ff_lower": 1.00, "ff_upper": 1.25, "desc": "cut 50bp (inter-meeting)"},
    {"date": "2020-03-15", "decision": -1, "ff_lower": 0.00, "ff_upper": 0.25, "desc": "cut 100bp to ZLB"},

    # 2020-2021 — ZIRP
    {"date": "2020-06-10", "decision": 0, "ff_lower": 0.00, "ff_upper": 0.25, "desc": "hold (ZIRP)"},

    # 2022 — aggressive tightening
    {"date": "2022-03-16", "decision": 1, "ff_lower": 0.25, "ff_upper": 0.50, "desc": "hike 25bp (liftoff)"},
    {"date": "2022-05-04", "decision": 1, "ff_lower": 0.75, "ff_upper": 1.00, "desc": "hike 50bp"},
    {"date": "2022-06-15", "decision": 1, "ff_lower": 1.50, "ff_upper": 1.75, "desc": "hike 75bp"},
    {"date": "2022-07-27", "decision": 1, "ff_lower": 2.25, "ff_upper": 2.50, "desc": "hike 75bp"},
    {"date": "2022-09-21", "decision": 1, "ff_lower": 3.00, "ff_upper": 3.25, "desc": "hike 75bp"},
    {"date": "2022-11-02", "decision": 1, "ff_lower": 3.75, "ff_upper": 4.00, "desc": "hike 75bp"},
    {"date": "2022-12-14", "decision": 1, "ff_lower": 4.25, "ff_upper": 4.50, "desc": "hike 50bp"},

    # 2023
    {"date": "2023-02-01", "decision": 1, "ff_lower": 4.50, "ff_upper": 4.75, "desc": "hike 25bp"},
    {"date": "2023-03-22", "decision": 1, "ff_lower": 4.75, "ff_upper": 5.00, "desc": "hike 25bp"},
    {"date": "2023-05-03", "decision": 1, "ff_lower": 5.00, "ff_upper": 5.25, "desc": "hike 25bp"},
    {"date": "2023-06-14", "decision": 0, "ff_lower": 5.00, "ff_upper": 5.25, "desc": "hold (skip)"},
    {"date": "2023-07-26", "decision": 1, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hike 25bp (terminal)"},
    {"date": "2023-09-20", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hold"},
    {"date": "2023-11-01", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hold"},
    {"date": "2023-12-13", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hold"},

    # 2024
    {"date": "2024-01-31", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hold"},
    {"date": "2024-03-20", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hold"},
    {"date": "2024-05-01", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hold"},
    {"date": "2024-06-12", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hold"},
    {"date": "2024-07-31", "decision": 0, "ff_lower": 5.25, "ff_upper": 5.50, "desc": "hold"},
    {"date": "2024-09-18", "decision": -1, "ff_lower": 5.00, "ff_upper": 5.25, "desc": "cut 25bp"},
    {"date": "2024-11-07", "decision": -1, "ff_lower": 4.75, "ff_upper": 5.00, "desc": "cut 25bp"},
    {"date": "2024-12-18", "decision": -1, "ff_lower": 4.50, "ff_upper": 4.75, "desc": "cut 25bp"},

    # 2025
    {"date": "2025-01-29", "decision": -1, "ff_lower": 4.25, "ff_upper": 4.50, "desc": "cut 25bp"},
    {"date": "2025-03-19", "decision": -1, "ff_lower": 4.00, "ff_upper": 4.25, "desc": "cut 25bp"},
    {"date": "2025-05-07", "decision": -1, "ff_lower": 3.75, "ff_upper": 4.00, "desc": "cut 25bp"},
    {"date": "2025-06-18", "decision": -1, "ff_lower": 3.50, "ff_upper": 3.75, "desc": "cut 25bp"},
    {"date": "2025-07-30", "decision": 0, "ff_lower": 3.50, "ff_upper": 3.75, "desc": "hold (Waller+Bowman dissent)"},
    {"date": "2025-09-17", "decision": -1, "ff_lower": 3.25, "ff_upper": 3.50, "desc": "cut 25bp"},
    {"date": "2025-10-29", "decision": 0, "ff_lower": 3.25, "ff_upper": 3.50, "desc": "hold"},
    {"date": "2025-12-10", "decision": 0, "ff_lower": 3.25, "ff_upper": 3.50, "desc": "hold (cut signaled but not executed?)"},

    # 2026 (Jan-Jun, from Fed release records and local scenario labels)
    {"date": "2026-01-28", "decision": 0, "ff_lower": 3.50, "ff_upper": 3.75, "desc": "hold (Waller+Miran dissent for cut)"},
    {"date": "2026-03-18", "decision": 0, "ff_lower": 3.50, "ff_upper": 3.75, "desc": "hold (Miran dissent for cut)"},
    {"date": "2026-04-29", "decision": 0, "ff_lower": 3.50, "ff_upper": 3.75, "desc": "hold (4 dissenters)"},
    {"date": "2026-06-17", "decision": 0, "ff_lower": 3.50, "ff_upper": 3.75, "desc": "hold (Warsh first meeting, hawkish SEP)"},
]


def build_fomc_label_df() -> pd.DataFrame:
    """Build a DataFrame of FOMC labels with quarter-aligned dates for merging."""
    df = pd.DataFrame(FOMC_LABELS)
    df["date"] = pd.to_datetime(df["date"])
    # Align to quarter-end for merging with macro panel
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp(how="end")
    return df
