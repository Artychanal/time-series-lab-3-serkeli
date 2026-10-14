from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_END = pd.Timestamp("2026-08-01")


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    required = {"date", "positive_reviews", "negative_reviews", "total_reviews", "positive_share"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    numeric = ["positive_reviews", "negative_reviews", "total_reviews", "positive_share"]
    values = frame[numeric].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or frame["date"].isna().any():
        raise ValueError("Input contains missing or nonfinite values")
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate calendar months")
    if not (frame["date"] == frame["date"].dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Dates must be the first day of each month")
    expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="MS")
    if len(expected.difference(frame["date"])):
        raise ValueError("Calendar months are missing")
    counts = values[:, :3]
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Review counts must be nonnegative integers")
    if not np.equal(counts[:, 0] + counts[:, 1], counts[:, 2]).all():
        raise ValueError("Inconsistent positive + negative totals")
    shares = np.divide(counts[:, 0], counts[:, 2], out=np.zeros(len(frame)), where=counts[:, 2] > 0)
    if not np.allclose(shares, values[:, 3], atol=1e-9, rtol=0):
        raise ValueError("positive_share does not agree with review counts")
    selected = frame.loc[frame["date"] <= ANALYSIS_END].copy().reset_index(drop=True)
    if len(selected) < 48:
        raise ValueError("At least 48 complete monthly observations are required")
    selected["log_reviews"] = np.log1p(selected["total_reviews"].to_numpy(float))
    metadata_path = path.with_name("cyberpunk_2077_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    report = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_observations": len(frame), "observations": len(selected),
        "period_start": selected["date"].iloc[0].date().isoformat(),
        "period_end": selected["date"].iloc[-1].date().isoformat(),
        "excluded_months": frame.loc[frame["date"] > ANALYSIS_END, "date"].dt.strftime("%Y-%m").tolist(),
        "missing_values": 0, "duplicates": 0, "missing_months": [],
        "totals_consistent": True, "shares_consistent": True,
        "collected_at_utc": metadata.get("collected_at_utc"),
        "source_url": metadata.get("histogram_url"),
        "analysis_cutoff": str(ANALYSIS_END.date()),
        "excluded_reason": "Frozen snapshot: September 2026 was incomplete at collection",
    }
    return selected, report
