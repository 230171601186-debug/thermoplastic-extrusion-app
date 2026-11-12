"""Capture immutable baseline evidence before experimental data generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN_DIR = (
    Path(__file__).resolve().parents[1]
    / "model2_generator_experiment_2026-10-11"
)
DATASET = ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
METADATA = ROOT / "models/model_metadata.json"
SPLIT = ROOT / "models/data_split.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    df = pd.read_csv(DATASET, keep_default_na=False)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    artifact_paths = [
        METADATA,
        SPLIT,
        *sorted((ROOT / "models").rglob("*.pkl")),
        *sorted((ROOT / "training/evaluation_results/controlled_stacking/model2_candidate/candidate_artifacts").glob("*.pkl")),
        *sorted((ROOT / "training/evaluation_results/controlled_stacking/model3_candidate/candidate_artifacts").glob("*.pkl")),
        ROOT / "training/results/model1_metrics.csv",
        ROOT / "training/results/model2_metrics.csv",
        ROOT / "training/results/model3_metrics.csv",
        ROOT / "training/evaluation_results/controlled_stacking/metrics_by_fold.csv",
        ROOT / "training/evaluation_results/controlled_stacking/metrics_summary.csv",
        ROOT / "training/evaluation_results/controlled_stacking/model2_candidate/test_comparison.csv",
        ROOT / "training/evaluation_results/controlled_stacking/model2_candidate/nested_cv_summary.csv",
    ]
    artifact_hashes = {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in artifact_paths
        if path.exists()
    }

    identities = {
        "actual_length_vs_speed_runtime_max_abs": float(
            (df.Actual_Processing_Length_m - df.Line_Speed_m_min * df.Run_Time_min).abs().max()
        ),
        "good_plus_scrap_vs_actual_max_abs": float(
            (df.Good_Length_m + df.Scrap_Length_m - df.Actual_Processing_Length_m).abs().max()
        ),
        "availability_vs_runtime_ratio_max_abs": float(
            (df.Availability - df.Run_Time_min / df.Planned_Production_Time_min).abs().max()
        ),
        "oee_vs_product_max_abs": float(
            (df.OEE - df.Availability * df.Performance * df.Quality).abs().max()
        ),
        "efficiency_vs_oee_pct_max_abs": float(
            (df.Operational_Efficiency_pct - df.OEE * 100.0).abs().max()
        ),
    }
    completed = df.Order_Status == "Completed"
    good_output_below_requirement = (
        completed & (df.Good_Length_m < df.Customer_Required_Length_m)
    )
    target_correlations = df[
        [
            "Operational_Efficiency_pct",
            "Availability",
            "Performance",
            "Quality",
            "Downtime_min",
            "Scrap_Rate",
            "Line_Speed_m_min",
            "Screw_RPM",
            "Extrusion_Speed_kg_hr",
        ]
    ].corr()["Operational_Efficiency_pct"].round(6).to_dict()
    fault_summary = (
        df.groupby("Fault_Code").Operational_Efficiency_pct.agg(["count", "mean", "std"])
        .round(6)
        .to_dict(orient="index")
    )
    baseline = {
        "dataset_path": DATASET.relative_to(ROOT).as_posix(),
        "dataset_sha256": digest(DATASET),
        "dataset_shape": list(df.shape),
        "dataset_columns": list(df.columns),
        "metadata_path": METADATA.relative_to(ROOT).as_posix(),
        "model_metadata_sha256": digest(METADATA),
        "split_path": SPLIT.relative_to(ROOT).as_posix(),
        "split_sha256": digest(SPLIT),
        "split_definition": {
            "test_size": split["test_size"],
            "random_state": split["random_state"],
            "train_rows": len(split["train_index"]),
            "test_rows": len(split["test_index"]),
            "train_indices": split["train_index"],
            "test_indices": split["test_index"],
        },
        "model2_production_metrics": metadata["model2"],
        "model2_controlled_test_comparison_path": "training/evaluation_results/controlled_stacking/model2_candidate/test_comparison.csv",
        "model2_controlled_nested_cv_summary_path": "training/evaluation_results/controlled_stacking/model2_candidate/nested_cv_summary.csv",
        "model2_controlled_nested_cv_metrics_path": "training/evaluation_results/controlled_stacking/metrics_summary.csv",
        "accounting_identity_max_residuals": identities,
        "efficiency_correlations": target_correlations,
        "efficiency_by_fault_code": fault_summary,
        "fault_rows": int((df.Fault_Code != "No Fault").sum()),
        "braid_none_rows": int((df.Braid_Type == "None").sum()),
        "completed_rows": int(completed.sum()),
        "completed_good_length_below_customer_requirement_rows": int(
            good_output_below_requirement.sum()
        ),
        "completed_good_length_below_customer_requirement_fraction": float(
            good_output_below_requirement.sum() / max(1, completed.sum())
        ),
        "missing_cells": int(df.isna().sum().sum()),
        "unique_record_ids": int(df.Record_ID.nunique()),
        "artifact_hashes": artifact_hashes,
        "source_audit_notes": [
            "Process targets use explicit material, grade, tube geometry/type, braid, and machine-profile terms plus noise.",
            "Downtime is fault-specific; runtime is planned time less downtime; actual length is line speed times runtime.",
            "Performance in the existing generator uses actual length over a denominator containing planned time; availability separately uses runtime/planned time and is multiplied again in OEE.",
            "The existing generator also applies a direct risk-score performance multiplier before calculating OEE. The experiment changes this definition only, not OEE accounting identity.",
            "Generator bounds and synthetic profiles are not validated Polyhose manufacturing specifications.",
        ],
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "baseline_snapshot.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    print(RUN_DIR / "baseline_snapshot.json", "written")
    print("dataset_sha256", baseline["dataset_sha256"], "shape", df.shape)
    print("model2_test", metadata["model2"]["test_metrics"])
    print("model2_cv", metadata["model2"]["cv_metrics"])
    print("artifact_hash_count", len(artifact_hashes))
    print("accounting_residuals", identities)


if __name__ == "__main__":
    main()
