from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent


def validate() -> dict:
    predictions = pd.read_csv(BASE / "downstream_predictions_residuals.csv")
    fold_metrics = pd.read_csv(BASE / "downstream_fold_metrics.csv")
    summary = pd.read_csv(BASE / "downstream_cv_summary.csv")
    upstream = pd.read_csv(BASE / "upstream_fold_audit.csv")
    process_metrics = pd.read_csv(BASE / "model1_process_metrics.csv")
    process_predictions = pd.read_csv(BASE / "model1_outer_predictions.csv")
    manifest = json.loads((BASE / "experiment_manifest.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    checks["manifest_has_no_prohibited_feature"] = not manifest["feature_contract"]["prohibited_intersection"]
    checks["downstream_prediction_schema"] = set(predictions["configuration"]) == {
        "training_mean",
        "customer_only",
        "full_stack",
    }
    checks["all_downstream_outer_rows_once"] = all(
        len(group) == 4000
        and group["row_index"].nunique() == 4000
        and group["fold"].nunique() == 5
        for _, group in predictions[predictions["split"] == "outer_cv"].groupby(
            ["target", "configuration"]
        )
    )
    checks["all_downstream_saved_test_rows_once"] = all(
        len(group) == 1000 and group["row_index"].nunique() == 1000
        for _, group in predictions[predictions["split"] == "saved_test"].groupby(
            ["target", "configuration"]
        )
    )
    checks["upstream_oof_fit_predict_disjoint"] = (
        len(upstream) == 6 * 9 * 4
        and int(upstream["fit_predict_index_overlap"].max()) == 0
        and upstream["oof_prediction_count_min"].eq(1).all()
        and upstream["oof_prediction_count_max"].eq(1).all()
    )
    outer_audit = upstream[upstream["outer_partition"] == "outer_cv"]
    test_audit = upstream[upstream["outer_partition"] == "saved_test"]
    checks["outer_training_excludes_outer_validation"] = (
        len(outer_audit) == 5 * 9 * 4
        and outer_audit["outer_fit_rows"].eq(3200).all()
        and outer_audit["outer_evaluation_rows"].eq(800).all()
        and outer_audit["outer_fit_evaluation_index_overlap"].eq(0).all()
    )
    checks["saved_test_partition_excluded_from_upstream_fitting"] = (
        len(test_audit) == 9 * 4
        and test_audit["outer_fit_rows"].eq(4000).all()
        and test_audit["outer_evaluation_rows"].eq(1000).all()
        and test_audit["outer_fit_evaluation_index_overlap"].eq(0).all()
    )
    checks["outer_folds_have_800_rows"] = (
        fold_metrics[fold_metrics["split"] == "outer_cv"]["n_rows"].eq(800).all()
    )
    checks["scrap_metrics_include_percentage_points"] = all(
        column in fold_metrics.columns
        and fold_metrics.loc[fold_metrics["target"] == "Scrap_Rate", column].notna().all()
        for column in ("MAE_percentage_points", "RMSE_percentage_points")
    )
    checks["process_metrics_have_nine_targets"] = set(process_metrics["target"]) == set(
        manifest["feature_contract"]["process_targets"]
    )
    checks["process_outer_rows_cover_training_partition"] = all(
        len(group) == 4000 and group["row_index"].nunique() == 4000
        for _, group in process_predictions.groupby("target")
    )
    checks["no_process_prediction_out_of_existing_synthetic_bounds"] = (
        process_metrics["outside_existing_synthetic_bound_count"].eq(0).all()
    )

    metric_recompute_ok = True
    for row in fold_metrics.itertuples(index=False):
        group = predictions[
            predictions["split"].eq(row.split)
            & predictions["fold"].eq(row.fold)
            & predictions["target"].eq(row.target)
            & predictions["configuration"].eq(row.configuration)
        ]
        actual = group["actual"].to_numpy()
        predicted = group["predicted"].to_numpy()
        calculated = (
            mean_absolute_error(actual, predicted),
            np.sqrt(mean_squared_error(actual, predicted)),
            r2_score(actual, predicted),
        )
        saved = (row.MAE, row.RMSE, row.R2)
        if not np.allclose(calculated, saved, rtol=0, atol=1e-12):
            metric_recompute_ok = False
            break
    checks["fold_metrics_recomputed_from_saved_predictions"] = metric_recompute_ok

    summary_recompute_ok = True
    for row in summary.itertuples(index=False):
        subset = fold_metrics[
            fold_metrics["split"].eq("outer_cv")
            & fold_metrics["target"].eq(row.target)
            & fold_metrics["configuration"].eq(row.configuration)
        ]
        for metric in ("MAE", "RMSE", "R2"):
            expected_mean = float(subset[metric].mean())
            expected_std = float(subset[metric].std(ddof=1))
            if not np.isclose(getattr(row, f"{metric}_mean"), expected_mean, atol=1e-12):
                summary_recompute_ok = False
            if not np.isclose(getattr(row, f"{metric}_std"), expected_std, atol=1e-12):
                summary_recompute_ok = False
    checks["cv_summary_recomputed_from_fold_metrics"] = summary_recompute_ok

    result = {
        "checks": {name: bool(passed) for name, passed in checks.items()},
        "passed": bool(all(checks.values())),
        "downstream_prediction_rows": len(predictions),
        "outer_cv_rows_per_target_configuration": 4000,
        "saved_test_rows_per_target_configuration": 1000,
        "upstream_oof_audit_rows": len(upstream),
        "process_outer_prediction_rows": len(process_predictions),
    }
    (BASE / "validation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    validate()
