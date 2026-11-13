"""Production-intelligence helpers with explicit boundaries around unavailable plant data.

Prediction history is session-scoped in Streamlit Community Cloud. Downtime
uploads are descriptive only; this module does not claim to train or infer
machine-failure risk from synthetic or unverified records.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REQUIRED_DOWNTIME_COLUMNS = (
    "Timestamp",
    "Machine_ID",
    "Downtime_min",
    "Fault_Code",
)
OPTIONAL_DOWNTIME_COLUMNS = ("Shift", "Production_Order_ID", "Operator_ID")


def append_prediction_history(
    history: Sequence[Mapping[str, object]],
    customer_inputs: Mapping[str, object],
    predictions: Mapping[str, float],
    generated_at_utc: str,
    max_entries: int = 100,
) -> list[dict[str, object]]:
    """Return a bounded, newest-first in-memory history without mutating inputs."""
    if max_entries < 1:
        raise ValueError("max_entries must be at least one.")
    entry: dict[str, object] = {
        "Generated_at_UTC": generated_at_utc,
        **{f"Input_{key}": value for key, value in customer_inputs.items()},
        **{f"Prediction_{key}": float(value) for key, value in predictions.items()},
    }
    return [entry, *[dict(item) for item in history[: max_entries - 1]]]


def history_frame(history: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Convert session history to a display/export dataframe."""
    return pd.DataFrame([dict(item) for item in history])


def audit_downtime_records(frame: pd.DataFrame) -> dict[str, object]:
    """Validate uploaded downtime records and return descriptive summaries only."""
    missing = [column for column in REQUIRED_DOWNTIME_COLUMNS if column not in frame.columns]
    issues: list[str] = []
    if frame.empty:
        issues.append("The uploaded file contains no records.")
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}.")
    if not missing and not frame.empty:
        timestamps = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
        downtime = pd.to_numeric(frame["Downtime_min"], errors="coerce")
        if timestamps.isna().any():
            issues.append("Timestamp contains missing or unparseable values.")
        if downtime.isna().any() or not np.isfinite(downtime.dropna()).all():
            issues.append("Downtime_min contains missing, non-numeric or non-finite values.")
        elif (downtime < 0).any():
            issues.append("Downtime_min cannot contain negative values.")
        if frame["Machine_ID"].isna().any() or frame["Machine_ID"].astype("string").str.strip().eq("").any():
            issues.append("Machine_ID contains missing or blank values.")
        if frame["Fault_Code"].isna().any() or frame["Fault_Code"].astype("string").str.strip().eq("").any():
            issues.append("Fault_Code contains missing or blank values.")

    if issues:
        return {
            "passed": False,
            "issues": issues,
            "rows": int(len(frame)),
            "machine_count": 0,
            "total_downtime_min": None,
            "by_machine": pd.DataFrame(),
            "by_fault": pd.DataFrame(),
        }

    clean = frame.copy()
    clean["Timestamp"] = pd.to_datetime(clean["Timestamp"], errors="raise", utc=True)
    clean["Downtime_min"] = pd.to_numeric(clean["Downtime_min"], errors="raise")
    by_machine = (
        clean.groupby("Machine_ID", dropna=False)["Downtime_min"]
        .agg(Events="count", Total_downtime_min="sum", Mean_event_min="mean")
        .reset_index()
        .sort_values("Total_downtime_min", ascending=False)
    )
    by_fault = (
        clean.groupby("Fault_Code", dropna=False)["Downtime_min"]
        .agg(Events="count", Total_downtime_min="sum")
        .reset_index()
        .sort_values("Total_downtime_min", ascending=False)
    )
    clean["Date"] = clean["Timestamp"].dt.floor("D")
    by_day = (
        clean.groupby("Date", as_index=False)["Downtime_min"]
        .sum()
        .rename(columns={"Downtime_min": "Total_downtime_min"})
        .sort_values("Date")
    )
    return {
        "passed": True,
        "issues": [],
        "rows": int(len(clean)),
        "machine_count": int(clean["Machine_ID"].nunique()),
        "total_downtime_min": float(clean["Downtime_min"].sum()),
        "by_machine": by_machine,
        "by_fault": by_fault,
        "by_day": by_day,
    }
