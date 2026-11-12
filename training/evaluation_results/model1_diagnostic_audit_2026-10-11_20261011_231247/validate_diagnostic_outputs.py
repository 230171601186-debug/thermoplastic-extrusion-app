"""Validate generated diagnostic deliverables without rerunning model fitting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TARGETS = [
    "Extruder_Temperature_C",
    "Die_Temperature_C",
    "Cooling_Tank_Temperature_C",
    "Melt_Pressure_bar",
    "Air_Pressure_bar",
    "Screw_RPM",
    "Line_Speed_m_min",
    "Puller_Speed_m_min",
    "Extrusion_Speed_kg_hr",
]
ARMS = {
    "A_saved_Model1_inference_predictions": True,
    "B_outer_train_only_selected_Model1_OOF": True,
    "C_oracle_recorded_process_non_deployable": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        "model1_diagnostic_report.md",
        "model1_metrics.csv",
        "baseline_comparison.csv",
        "fold_level_results.csv",
        "model1_fold_aggregate.csv",
        "residual_diagnostics.csv",
        "feature_relationship_diagnostics.csv",
        "model2_downstream_impact.csv",
        "model2_downstream_summary.csv",
        "model2_downstream_predictions.csv",
        "preservation_manifest.json",
        "preservation_verification.json",
        "diagnostic_manifest.json",
        "README.md",
        "run_model1_diagnostic.py",
        "verify_preservation.py",
        "validate_diagnostic_outputs.py",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"Required diagnostic outputs are missing: {missing}")

    metrics = pd.read_csv(OUT / "model1_metrics.csv")
    folds = pd.read_csv(OUT / "fold_level_results.csv")
    baselines = pd.read_csv(OUT / "baseline_comparison.csv")
    impact = pd.read_csv(OUT / "model2_downstream_impact.csv")
    summary = pd.read_csv(OUT / "model2_downstream_summary.csv")
    predictions = pd.read_csv(OUT / "model2_downstream_predictions.csv")
    residuals = pd.read_csv(OUT / "residual_diagnostics.csv")
    manifest = json.loads((OUT / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    preservation = json.loads(
        (OUT / "preservation_verification.json").read_text(encoding="utf-8")
    )
    production_data = ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
    expected_hash = manifest["production_dataset_sha256"]
    checks = {
        "required_deliverables_present": not missing,
        "all_nine_model1_targets_reported_once": (
            len(metrics) == len(TARGETS) and set(metrics["target"]) == set(TARGETS)
        ),
        "baselines_cover_every_target": all(
            set(baselines.loc[baselines["comparison"] == label, "target"]) == set(TARGETS)
            for label in ("training_mean", "training_median")
        ),
        "candidate_and_saved_folds_have_five_results_each": all(
            len(group) == 5 and set(group["fold"]) == {1, 2, 3, 4, 5}
            for _, group in folds.groupby(["target", "evaluation_arm"])
        ) and folds.groupby(["target", "evaluation_arm"]).ngroups == 54,
        "model2_arms_have_five_outer_folds_and_saved_test": (
            len(summary) == 3
            and set(summary["arm"]) == set(ARMS)
            and (summary["outer_fold_count"] == 5).all()
            and summary[["saved_test_MAE", "saved_test_RMSE", "saved_test_R2"]]
            .notna().all().all()
        ),
        "oracle_marked_non_deployable": (
            summary.set_index("arm")["deployable"].to_dict() == ARMS
        ),
        "model2_fold_results_are_18_rows": len(impact) == 18,
        "model2_prediction_outputs_nonempty_and_finite": (
            len(predictions) > 0
            and np.isfinite(predictions[["actual", "predicted", "residual_pred_minus_actual"]]).all().all()
        ),
        "residual_diagnostics_nonempty_and_finite": (
            len(residuals) > 0
            and np.isfinite(residuals[["actual", "predicted", "residual_pred_minus_actual"]]).all().all()
        ),
        "speed_plots_are_valid_pngs": all(
            (OUT / name).is_file()
            and (OUT / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            for name in ("Line_Speed_m_min_diagnostics.png", "Puller_Speed_m_min_diagnostics.png")
        ),
        "preservation_hashes_verified": (
            preservation["verified"]
            and preservation["verified_file_count"] == preservation["expected_file_count"]
            and sha256(production_data) == expected_hash
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "model1_target_rows": len(metrics),
        "model1_fold_result_rows": len(folds),
        "model2_impact_rows": len(impact),
        "model2_prediction_rows": len(predictions),
        "residual_rows": len(residuals),
        "production_dataset_sha256": sha256(production_data),
    }
    (OUT / "validation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    diagnostic_manifest = manifest
    diagnostic_manifest["automated_validation"].update(checks)
    diagnostic_manifest["automated_validation"]["output_validation_passed"] = result["passed"]
    diagnostic_manifest["outputs"] = sorted(
        set(
            [path.name for path in OUT.iterdir() if path.is_file()]
            + ["validation_results.json"]
        )
    )
    (OUT / "diagnostic_manifest.json").write_text(
        json.dumps(diagnostic_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
