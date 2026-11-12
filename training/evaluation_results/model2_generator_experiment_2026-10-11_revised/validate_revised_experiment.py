"""Validate reproducibility, schema, accounting, and production preservation."""

from __future__ import annotations

import hashlib
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
PREVIOUS_DIR = ROOT / "training/evaluation_results/model2_generator_experiment_2026-10-11"
PRODUCTION_DATASET = ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
PRESERVATION_SNAPSHOT = EXPERIMENT_DIR / "previous_experiment_preservation_snapshot.json"

sys.path.insert(0, str(EXPERIMENT_DIR))
import experimental_generator as generator  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-preservation", action="store_true")
    args = parser.parse_args()
    if args.capture_preservation:
        snapshot = {
            "directory": PREVIOUS_DIR.relative_to(ROOT).as_posix(),
            "files": file_hashes(PREVIOUS_DIR),
        }
        PRESERVATION_SNAPSHOT.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"Captured {len(snapshot['files'])} prior experiment file hashes")
        return

    csv_path = EXPERIMENT_DIR / "experimental_dataset.csv"
    saved = pd.read_csv(csv_path, keep_default_na=False)
    regenerated = generator.generate_dataset(seed=generator.SEED, row_count=generator.ROW_COUNT)
    if not regenerated.equals(generator.generate_dataset(seed=generator.SEED, row_count=generator.ROW_COUNT)):
        raise AssertionError("Fixed-seed generator output is not exactly reproducible")
    if list(saved.columns) != generator.COLUMNS or saved.shape != (5000, 40):
        raise AssertionError("Saved dataset does not preserve the 5000 x 40 contract")
    if list(regenerated.columns) != list(saved.columns):
        raise AssertionError("Regenerated schema differs from saved dataset")
    if not np.allclose(
        saved.select_dtypes(include=[np.number]).to_numpy(dtype=float),
        regenerated.select_dtypes(include=[np.number]).to_numpy(dtype=float),
        atol=1e-10,
    ):
        raise AssertionError("Saved dataset differs from fixed-seed regeneration")

    previous = pd.read_csv(PREVIOUS_DIR / "experimental_dataset.csv", keep_default_na=False)
    categorical = [
        "Material_Type", "Material_Grade", "Tube_Type", "Braid_Type",
        "Machine_ID", "Shift", "Fault_Code", "Order_Status",
    ]
    for column in categorical:
        if set(saved[column].astype(str)) != set(previous[column].astype(str)):
            raise AssertionError(f"Categorical labels changed for {column}")

    audit = generator.validate_dataset(saved)
    positive = saved.Actual_Processing_Length_m > 0
    residuals = {
        "actual_length": float(
            (saved.Actual_Processing_Length_m - saved.Line_Speed_m_min * saved.Run_Time_min).abs().max()
        ),
        "good_plus_scrap": float(
            (saved.Good_Length_m + saved.Scrap_Length_m - saved.Actual_Processing_Length_m).abs().max()
        ),
        "scrap_length": float(
            (saved.Scrap_Length_m - saved.Actual_Processing_Length_m * saved.Scrap_Rate).abs().max()
        ),
        "good_length": float(
            (saved.Good_Length_m - (saved.Actual_Processing_Length_m - saved.Scrap_Length_m)).abs().max()
        ),
        "length_shortfall": float(
            (
                saved.Length_Shortfall_m
                - np.maximum(0.0, saved.Planned_Processing_Length_m - saved.Actual_Processing_Length_m)
            ).abs().max()
        ),
        "runtime": float(
            (
                saved.Run_Time_min
                - np.maximum(0.0, saved.Planned_Production_Time_min - saved.Downtime_min)
            ).abs().max()
        ),
        "availability": float(
            (saved.Availability - saved.Run_Time_min / saved.Planned_Production_Time_min).abs().max()
        ),
        "quality": float(
            (
                saved.loc[positive, "Quality"]
                - saved.loc[positive, "Good_Length_m"] / saved.loc[positive, "Actual_Processing_Length_m"]
            ).abs().max()
        ),
        "oee": float(
            (saved.OEE - saved.Availability * saved.Performance * saved.Quality).abs().max()
        ),
        "operational_efficiency": float(
            (saved.Operational_Efficiency_pct - 100.0 * saved.OEE).abs().max()
        ),
    }
    if saved.Performance.max() >= 1.0 or saved.Performance.min() <= 0:
        raise AssertionError("Performance retains an upper/lower saturation point")
    if ((saved.Order_Status == "Completed") & (saved.Good_Length_m < saved.Customer_Required_Length_m)).any():
        raise AssertionError("Completed order falls short of customer-required good output")

    baseline = json.loads((EXPERIMENT_DIR / "baseline_snapshot.json").read_text(encoding="utf-8"))
    production_hash = sha256(PRODUCTION_DATASET)
    if production_hash != baseline["dataset_sha256"]:
        raise AssertionError("Production dataset SHA-256 changed")
    changed_artifacts = []
    for relative_path, expected_hash in baseline["artifact_hashes"].items():
        if sha256(ROOT / Path(relative_path)) != expected_hash:
            changed_artifacts.append(relative_path)
    if changed_artifacts:
        raise AssertionError(f"Production/model artifacts changed: {changed_artifacts}")
    previous_snapshot = json.loads(PRESERVATION_SNAPSHOT.read_text(encoding="utf-8"))
    previous_hashes = file_hashes(PREVIOUS_DIR)
    if previous_hashes != previous_snapshot["files"]:
        raise AssertionError("Existing experiment directory changed during the revised run")

    result = {
        "validation": "passed",
        "fixed_seed": generator.SEED,
        "fixed_seed_regeneration_exact": True,
        "dataset_sha256": sha256(csv_path),
        "dataset_shape": list(saved.shape),
        "schema_matches_previous_experiment": list(saved.columns) == list(previous.columns),
        "categorical_labels_match_previous_experiment": True,
        "performance_min": float(saved.Performance.min()),
        "performance_median": float(saved.Performance.median()),
        "performance_max": float(saved.Performance.max()),
        "efficiency_min": float(saved.Operational_Efficiency_pct.min()),
        "efficiency_median": float(saved.Operational_Efficiency_pct.median()),
        "efficiency_max": float(saved.Operational_Efficiency_pct.max()),
        "completed_orders": int((saved.Order_Status == "Completed").sum()),
        "completed_orders_below_required_good_length": int(
            (
                (saved.Order_Status == "Completed")
                & (saved.Good_Length_m < saved.Customer_Required_Length_m)
            ).sum()
        ),
        "accounting_max_abs_residuals": residuals,
        "generator_validation_summary": audit,
        "production_dataset_sha256_unchanged": production_hash,
        "production_artifact_hashes_verified_unchanged": len(baseline["artifact_hashes"]),
        "production_artifacts_changed": changed_artifacts,
        "previous_experiment_files_verified_unchanged": len(previous_hashes),
    }
    (EXPERIMENT_DIR / "automated_validation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
