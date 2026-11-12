"""Non-destructive quality gates for the shared extrusion training dataset.

The checks deliberately validate the existing data rather than repairing,
imputing, deduplicating, or generating rows silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    EFFICIENCY_TARGET,
    MODEL1_FEATURES,
    PROCESS_TARGETS,
    SCRAP_TARGET,
)

REQUIRED_MODEL_COLUMNS = (
    MODEL1_FEATURES + PROCESS_TARGETS + [EFFICIENCY_TARGET, SCRAP_TARGET]
)


def audit_modeling_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Return a compact, deterministic audit summary without mutating frame."""
    issues: list[str] = []
    if frame.empty:
        issues.append("dataset contains no rows")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        issues.append(f"duplicate column names: {duplicates}")

    missing_columns = [name for name in REQUIRED_MODEL_COLUMNS if name not in frame.columns]
    if missing_columns:
        issues.append(f"missing required model columns: {missing_columns}")

    missing_cells: dict[str, int] = {}
    non_numeric: dict[str, int] = {}
    non_finite: dict[str, int] = {}
    blank_categories: dict[str, int] = {}

    present_required = [name for name in REQUIRED_MODEL_COLUMNS if name in frame.columns]
    for name in present_required:
        count = int(frame[name].isna().sum())
        if count:
            missing_cells[name] = count

    for name in CATEGORICAL_FEATURES:
        if name not in frame:
            continue
        values = frame[name].astype("string").str.strip()
        count = int((values.isna() | values.eq("")).sum())
        if count:
            blank_categories[name] = count

    numeric_columns = CUSTOMER_NUMERIC_FEATURES + PROCESS_TARGETS + [
        EFFICIENCY_TARGET,
        SCRAP_TARGET,
    ]
    for name in numeric_columns:
        if name not in frame:
            continue
        converted = pd.to_numeric(frame[name], errors="coerce")
        invalid_count = int((converted.isna() & frame[name].notna()).sum())
        if invalid_count:
            non_numeric[name] = invalid_count
        finite_count = int((~np.isfinite(converted.dropna().to_numpy(dtype=float))).sum())
        if finite_count:
            non_finite[name] = finite_count

    if missing_cells:
        issues.append(f"missing required values: {missing_cells}")
    if blank_categories:
        issues.append(f"blank categorical values: {blank_categories}")
    if non_numeric:
        issues.append(f"non-numeric values in numeric fields: {non_numeric}")
    if non_finite:
        issues.append(f"non-finite numeric values: {non_finite}")

    duplicates = int(frame.duplicated().sum()) if not frame.empty else 0
    return {
        "passed": not issues,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows_reported": duplicates,
        "required_columns": len(REQUIRED_MODEL_COLUMNS),
        "missing_columns": missing_columns,
        "missing_cells": missing_cells,
        "blank_categories": blank_categories,
        "non_numeric_values": non_numeric,
        "non_finite_values": non_finite,
        "issues": issues,
    }


def validate_modeling_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Fail closed on structural/data-type issues and return the audit summary."""
    report = audit_modeling_dataset(frame)
    if not report["passed"]:
        raise ValueError(
            "Training dataset failed quality validation; no values were repaired. "
            + " | ".join(report["issues"])
        )
    return report
