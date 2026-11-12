"""Leakage-controlled Model 2 experiment on the isolated generator dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "training"
for path in (EXPERIMENT_DIR, ROOT, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    CUSTOMER_NUMERIC_FEATURES,
    EFFICIENCY_TARGET,
    MODEL1_FEATURES,
    PROCESS_TARGETS,
    RANDOM_STATE,
    RF_PARAMS,
    SPLIT_INDICES_PATH,
    TEST_SIZE,
)
from evaluate import regression_metrics  # noqa: E402
from preprocessing import make_regressor_pipeline, make_stacked_pipeline  # noqa: E402
from training.evaluation_results.controlled_stacking import run_controlled_stacking as reference  # noqa: E402
from experimental_generator import (  # noqa: E402
    COLUMNS,
    NOMINAL_SPEED_BOUNDS,
    OUTPUT_PATH,
    PROCESS_BOUNDS,
    generate_dataset,
    validate_dataset,
)

OUT = Path(__file__).resolve().parent
BASELINE_PATH = OUT / "baseline_snapshot.json"
BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
CONTROLLED_DIR = ROOT / "training/evaluation_results/controlled_stacking"
PREVIOUS_EXPERIMENT_DIR = ROOT / "training/evaluation_results/model2_generator_experiment_2026-10-11"
BASELINE_NESTED_DIR = PREVIOUS_EXPERIMENT_DIR / "baseline_nested_reproduction"
OUTER_FOLDS = reference.OUTER_FOLDS
INNER_DOWNSTREAM_FOLDS = 3
DOWNSTREAM_CANDIDATES = tuple(reference.ALGORITHMS)
FIXED_RF_PARAMS = deepcopy(reference.DOWNSTREAM_PARAMS)
TARGET_OUTPUT_BOUNDS = (0.0, 100.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_experimental_frame() -> pd.DataFrame:
    frame = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
    if frame.shape != (5000, 40) or list(frame.columns) != COLUMNS:
        raise RuntimeError("Experimental CSV does not match the fixed 5000 x 40 contract")
    if "None" not in set(frame["Braid_Type"].astype(str)):
        raise RuntimeError("Experimental dataset lost Braid_Type='None'")
    return frame


def validate_reused_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    split = json.loads(SPLIT_INDICES_PATH.read_text(encoding="utf-8"))
    if len(split["train_index"]) != 4000 or len(split["test_index"]) != 1000:
        raise RuntimeError("The existing saved split is not 4000/1000")
    if set(split["train_index"]) & set(split["test_index"]):
        raise RuntimeError("Saved training/test indices overlap")
    # Same saved row positions and seed are reused; generated rows are a new dataset, not paired observations.
    return frame.loc[split["train_index"]].copy(), frame.loc[split["test_index"]].copy(), split


def estimator_for(name: str):
    estimator_class, params = reference.ALGORITHMS[name]
    return estimator_class(**deepcopy(params))


def score_nested_downstream_candidates(outer_train: pd.DataFrame, outer_label: str, selection_log: list[dict]) -> str:
    """Select downstream algorithm using inner folds that each rebuild their own stack."""
    splitter = KFold(n_splits=INNER_DOWNSTREAM_FOLDS, shuffle=True, random_state=RANDOM_STATE + 17)
    fold_scores = {name: [] for name in DOWNSTREAM_CANDIDATES}
    for inner_fold, (fit_positions, validation_positions) in enumerate(splitter.split(outer_train), start=1):
        inner_train = outer_train.iloc[fit_positions].copy()
        inner_validation = outer_train.iloc[validation_positions].copy()
        upstream_selection = reference.select_upstream_algorithms(
            inner_train[MODEL1_FEATURES], inner_train,
            f"{outer_label}_inner_{inner_fold}", selection_log,
        )
        inner_train_stack, inner_final_upstream = reference.build_cross_fitted_process_features(
            inner_train, upstream_selection
        )
        inner_validation_stack = reference.predict_process_features(
            inner_validation[MODEL1_FEATURES], inner_final_upstream
        )
        for name in DOWNSTREAM_CANDIDATES:
            pipeline = make_stacked_pipeline(estimator_for(name), PROCESS_TARGETS)
            pipeline.fit(inner_train_stack, inner_train[EFFICIENCY_TARGET])
            prediction = pipeline.predict(inner_validation_stack)
            scores = regression_metrics(inner_validation[EFFICIENCY_TARGET], prediction)
            fold_scores[name].append(scores)
            selection_log.append({
                "scope": outer_label,
                "inner_fold": inner_fold,
                "downstream_algorithm": name,
                "inner_validation_MAE": scores["MAE"],
                "inner_validation_RMSE": scores["RMSE"],
                "inner_validation_R2": scores["R2"],
            })
    means = {
        name: {
            metric: float(np.mean([fold[metric] for fold in folds]))
            for metric in ("MAE", "RMSE", "R2")
        }
        for name, folds in fold_scores.items()
    }
    selected = min(means, key=lambda name: (means[name]["MAE"], means[name]["RMSE"], -means[name]["R2"]))
    for name, values in means.items():
        selection_log.append({
            "scope": outer_label,
            "inner_fold": "mean",
            "downstream_algorithm": name,
            "inner_validation_MAE": values["MAE"],
            "inner_validation_RMSE": values["RMSE"],
            "inner_validation_R2": values["R2"],
            "selected": name == selected,
        })
    return selected


def fit_configurations(train: pd.DataFrame, test: pd.DataFrame, train_stack: pd.DataFrame, test_stack: pd.DataFrame):
    y = train[EFFICIENCY_TARGET].to_numpy(dtype=float)
    model_customer = make_regressor_pipeline(reference.RandomForestRegressor(**deepcopy(FIXED_RF_PARAMS)))
    model_customer.fit(train[MODEL1_FEATURES], y)
    pred_customer = model_customer.predict(test[MODEL1_FEATURES])

    process_model = reference.make_process_only_pipeline()
    process_model.fit(train_stack[PROCESS_TARGETS], y)
    pred_process = process_model.predict(test_stack[PROCESS_TARGETS])

    fixed_full = make_stacked_pipeline(reference.RandomForestRegressor(**deepcopy(FIXED_RF_PARAMS)), PROCESS_TARGETS)
    fixed_full.fit(train_stack, y)
    pred_fixed_full = fixed_full.predict(test_stack)
    mean_pred = np.full(len(test), y.mean())
    return {
        "training_mean": (mean_pred, None),
        "customer_only_fixed_RF": (np.asarray(pred_customer), model_customer),
        "process_only_fixed_RF": (np.asarray(pred_process), process_model),
        "full_stack_fixed_RF": (np.asarray(pred_fixed_full), fixed_full),
    }


def append_metrics(rows: list[dict], predictions: list[dict], split: str, fold: str, train: pd.DataFrame, evaluate: pd.DataFrame, config_predictions: dict):
    actual = evaluate[EFFICIENCY_TARGET].to_numpy(dtype=float)
    target_ranges, _ = reference.target_range_labels(train[EFFICIENCY_TARGET], evaluate[EFFICIENCY_TARGET], EFFICIENCY_TARGET)
    for configuration, (predicted, model) in config_predictions.items():
        scores = regression_metrics(actual, predicted)
        rows.append({"dataset_version": "experiment", "split": split, "fold": fold, "configuration": configuration, **scores})
        for i, (truth, estimate) in enumerate(zip(actual, predicted)):
            predictions.append({
                "dataset_version": "experiment", "split": split, "fold": fold,
                "configuration": configuration,
                "record_id": int(evaluate.iloc[i]["Record_ID"]),
                "actual": float(truth), "predicted": float(estimate),
                "residual_pred_minus_actual": float(estimate - truth),
                "absolute_error": float(abs(estimate - truth)),
                "target_range_training_quantiles": str(target_ranges.iloc[i]),
            })


def audit_experimental_dataset(frame: pd.DataFrame) -> dict:
    validate_dataset(frame)
    positive_output = frame.Actual_Processing_Length_m > 0
    expected_shortfall = np.maximum(
        0.0, frame.Planned_Processing_Length_m - frame.Actual_Processing_Length_m
    )
    residuals = {
        "actual_length": frame.Actual_Processing_Length_m - frame.Line_Speed_m_min * frame.Run_Time_min,
        "good_plus_scrap": frame.Good_Length_m + frame.Scrap_Length_m - frame.Actual_Processing_Length_m,
        "good_length": frame.Good_Length_m - (
            frame.Actual_Processing_Length_m - frame.Scrap_Length_m
        ),
        "scrap_length": frame.Scrap_Length_m - frame.Actual_Processing_Length_m * frame.Scrap_Rate,
        "length_shortfall": frame.Length_Shortfall_m - expected_shortfall,
        "runtime": frame.Run_Time_min
        - np.maximum(0.0, frame.Planned_Production_Time_min - frame.Downtime_min),
        "quality": frame.loc[positive_output, "Quality"]
        - frame.loc[positive_output, "Good_Length_m"]
        / frame.loc[positive_output, "Actual_Processing_Length_m"],
        "availability": frame.Availability - frame.Run_Time_min / frame.Planned_Production_Time_min,
        "performance_bounds": np.maximum(0.0, frame.Performance - 1.0),
        "oee": frame.OEE - frame.Availability * frame.Performance * frame.Quality,
        "efficiency_pct": frame.Operational_Efficiency_pct - frame.OEE * 100.0,
    }
    max_residuals = {
        name: float(values.abs().max()) for name, values in residuals.items()
    }
    tolerances = {
        "actual_length": 0.011,
        "good_plus_scrap": 0.011,
        "good_length": 0.011,
        "scrap_length": 0.011,
        "length_shortfall": 0.011,
        "runtime": 0.011,
        "quality": 0.001,
        "availability": 0.001,
        "performance_bounds": 0.0,
        "oee": 0.001,
        "efficiency_pct": 0.01,
    }
    failures = {
        name: value
        for name, value in max_residuals.items()
        if value > tolerances[name]
    }
    if failures:
        raise RuntimeError(f"Experimental accounting identities failed: {failures}")
    if (frame.loc[frame.Fault_Code == "No Fault", "Downtime_min"] != 0).any():
        raise RuntimeError("No-fault records must have zero downtime")
    if (frame.loc[frame.Fault_Code != "No Fault", "Downtime_min"] <= 0).any():
        raise RuntimeError("Faulted records must have positive downtime")
    fault_counts = {
        str(name): int(count)
        for name, count in frame.Fault_Code.value_counts().items()
    }
    correlations = frame[
        [
            EFFICIENCY_TARGET,
            "Availability",
            "Performance",
            "Quality",
            "Downtime_min",
            "Scrap_Rate",
            "Line_Speed_m_min",
            "Screw_RPM",
            "Extrusion_Speed_kg_hr",
        ]
    ].corr()[EFFICIENCY_TARGET]
    validation = validate_dataset(frame)
    completed = frame.Order_Status == "Completed"
    completed_below_customer_requirement = completed & (
        frame.Good_Length_m < frame.Customer_Required_Length_m
    )
    return {
        **validation,
        "shape": list(frame.shape),
        "fault_count": int((frame.Fault_Code != "No Fault").sum()),
        "completed_rows": int(completed.sum()),
        "completed_good_length_below_customer_requirement_rows": int(
            completed_below_customer_requirement.sum()
        ),
        "completed_good_length_below_customer_requirement_fraction": float(
            completed_below_customer_requirement.sum() / max(1, completed.sum())
        ),
        "orders_meeting_good_length_requirement": int(
            (frame.Good_Length_m >= frame.Customer_Required_Length_m).sum()
        ),
        "fault_counts": fault_counts,
        "faulted_fraction": float((frame.Fault_Code != "No Fault").mean()),
        "accounting_max_abs_residuals": max_residuals,
        "process_bounds": {
            target: {
                "min": float(frame[target].min()),
                "max": float(frame[target].max()),
                "documented_synthetic_bounds": list(PROCESS_BOUNDS[target]),
            }
            for target in PROCESS_TARGETS
        },
        "nominal_speed_assumptions": {
            "nominal_speed_bounds_m_min": list(NOMINAL_SPEED_BOUNDS),
            "performance_definition": (
                "realized line rate divided by synthetic nominal recipe rate; "
                "realized rate fraction is generated below nominal"
            ),
            "bounds_are_factory_validated": False,
        },
        "target_summary": frame[
            [
                "Operational_Efficiency_pct",
                "Scrap_Rate",
                "Availability",
                "Performance",
                "Quality",
                "OEE",
            ]
        ].describe().to_dict(),
        "efficiency_correlations": {
            str(name): float(value) for name, value in correlations.items()
        },
    }


def data_relationship_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [*MODEL1_FEATURES, *PROCESS_TARGETS, "Availability", "Performance", "Quality", "Downtime_min", "Scrap_Rate"]:
        if pd.api.types.is_numeric_dtype(frame[column]):
            rows.append({
                "feature": column,
                "feature_type": "numeric",
                "target": EFFICIENCY_TARGET,
                "pearson_correlation": float(frame[column].corr(frame[EFFICIENCY_TARGET])),
                "category_level": "",
                "target_mean_by_level": np.nan,
                "rows_in_level": len(frame),
            })
        else:
            for level, group in frame.groupby(column, observed=True):
                rows.append({
                    "feature": column,
                    "feature_type": "categorical",
                    "target": EFFICIENCY_TARGET,
                    "pearson_correlation": np.nan,
                    "category_level": level,
                    "target_mean_by_level": float(group[EFFICIENCY_TARGET].mean()),
                    "rows_in_level": len(group),
                })
    return pd.DataFrame(rows)


def metric_summary(rows: pd.DataFrame) -> pd.DataFrame:
    cv = rows[rows.split == "outer_cv"]
    summaries = []
    for configuration, group in cv.groupby("configuration", sort=False):
        summaries.append({
            "dataset_version": "experiment",
            "split": "nested_outer_cv",
            "configuration": configuration,
            "fold_count": int(group.fold.nunique()),
            **{f"{metric}_mean": float(group[metric].mean()) for metric in ("MAE", "RMSE", "R2")},
            **{f"{metric}_std": float(group[metric].std(ddof=1)) for metric in ("MAE", "RMSE", "R2")},
        })
    test = rows[rows.split == "held_out_test"]
    for _, row in test.iterrows():
        summaries.append({
            "dataset_version": "experiment",
            "split": "saved_test",
            "configuration": row.configuration,
            "fold_count": 1,
            "MAE_mean": row.MAE, "MAE_std": np.nan,
            "RMSE_mean": row.RMSE, "RMSE_std": np.nan,
            "R2_mean": row.R2, "R2_std": np.nan,
        })
    return pd.DataFrame(summaries)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    values = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    rows = [headers, ["---"] * len(headers), *values]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = load_experimental_frame()
    dataset_sha = digest(OUTPUT_PATH)
    audit = audit_experimental_dataset(frame)
    (OUT / "experimental_dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    train, test, split = validate_reused_split(frame)

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_log: list[dict] = []
    importance_rows: list[dict] = []
    group_rows: list[dict] = []
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for outer_number, (fit_pos, validation_pos) in enumerate(outer.split(np.arange(len(train))), start=1):
        outer_train = train.iloc[fit_pos].copy()
        outer_validation = train.iloc[validation_pos].copy()
        label = f"outer_{outer_number}"
        selected_upstream = reference.select_upstream_algorithms(
            outer_train[MODEL1_FEATURES], outer_train, label, selection_log
        )
        training_stack, final_upstream = reference.build_cross_fitted_process_features(outer_train, selected_upstream)
        validation_stack = reference.predict_process_features(outer_validation[MODEL1_FEATURES], final_upstream)
        configurations = fit_configurations(outer_train, outer_validation, training_stack, validation_stack)
        append_metrics(fold_rows, prediction_rows, "outer_cv", label, outer_train, outer_validation, configurations)

        selected_downstream = score_nested_downstream_candidates(outer_train, label, selection_log)
        tuned = make_stacked_pipeline(estimator_for(selected_downstream), PROCESS_TARGETS)
        tuned.fit(training_stack, outer_train[EFFICIENCY_TARGET])
        tuned_pred = np.asarray(tuned.predict(validation_stack), dtype=float)
        scores = regression_metrics(outer_validation[EFFICIENCY_TARGET], tuned_pred)
        target_ranges, _ = reference.target_range_labels(
            outer_train[EFFICIENCY_TARGET],
            outer_validation[EFFICIENCY_TARGET],
            EFFICIENCY_TARGET,
        )
        fold_rows.append({"dataset_version": "experiment", "split": "outer_cv", "fold": label, "configuration": "inner_selected_full_stack", **scores})
        for local_i, (truth, estimate) in enumerate(zip(outer_validation[EFFICIENCY_TARGET].to_numpy(dtype=float), tuned_pred)):
            prediction_rows.append({"dataset_version": "experiment", "split": "outer_cv", "fold": label, "configuration": "inner_selected_full_stack", "record_id": int(outer_validation.iloc[local_i].Record_ID), "actual": float(truth), "predicted": float(estimate), "residual_pred_minus_actual": float(estimate-truth), "absolute_error": float(abs(estimate-truth)), "target_range_training_quantiles": str(target_ranges.iloc[local_i])})
        names = tuned.named_steps["preprocess"].get_feature_names_out()
        importances = tuned.named_steps["model"].feature_importances_
        for name, value in zip(names, importances):
            importance_rows.append({"training_scope": label, "algorithm": selected_downstream, "feature": str(name), "importance": float(value)})

    # Fit all choices on train only, then evaluate the reused untouched test indices once.
    full_upstream_selection = reference.select_upstream_algorithms(train[MODEL1_FEATURES], train, "full_training_partition", selection_log)
    train_stack, final_upstream = reference.build_cross_fitted_process_features(train, full_upstream_selection)
    test_stack = reference.predict_process_features(test[MODEL1_FEATURES], final_upstream)
    configurations = fit_configurations(train, test, train_stack, test_stack)
    append_metrics(fold_rows, prediction_rows, "held_out_test", "saved_test", train, test, configurations)
    selected_downstream = score_nested_downstream_candidates(train, "full_training_partition", selection_log)
    tuned = make_stacked_pipeline(estimator_for(selected_downstream), PROCESS_TARGETS)
    tuned.fit(train_stack, train[EFFICIENCY_TARGET])
    tuned_test_pred = np.asarray(tuned.predict(test_stack), dtype=float)
    test_scores = regression_metrics(test[EFFICIENCY_TARGET], tuned_test_pred)
    test_target_ranges, _ = reference.target_range_labels(
        train[EFFICIENCY_TARGET], test[EFFICIENCY_TARGET], EFFICIENCY_TARGET
    )
    fold_rows.append({"dataset_version": "experiment", "split": "held_out_test", "fold": "saved_test", "configuration": "inner_selected_full_stack", **test_scores})
    for local_i, (truth, estimate) in enumerate(zip(test[EFFICIENCY_TARGET].to_numpy(dtype=float), tuned_test_pred)):
        prediction_rows.append({"dataset_version": "experiment", "split": "held_out_test", "fold": "saved_test", "configuration": "inner_selected_full_stack", "record_id": int(test.iloc[local_i].Record_ID), "actual": float(truth), "predicted": float(estimate), "residual_pred_minus_actual": float(estimate-truth), "absolute_error": float(abs(estimate-truth)), "target_range_training_quantiles": str(test_target_ranges.iloc[local_i])})
    names = tuned.named_steps["preprocess"].get_feature_names_out()
    importances = tuned.named_steps["model"].feature_importances_
    for name, value in zip(names, importances):
        importance_rows.append({"training_scope": "full_training_partition", "algorithm": selected_downstream, "feature": str(name), "importance": float(value)})

    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    fold_frame.to_csv(OUT / "experimental_model2_metrics_by_fold.csv", index=False)
    metric_summary(fold_frame).to_csv(OUT / "experimental_model2_metric_summary.csv", index=False)
    prediction_frame.to_csv(OUT / "experimental_model2_predictions_residuals.csv", index=False)
    pd.DataFrame(selection_log).to_csv(OUT / "fold_local_algorithm_selection.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(OUT / "experimental_model2_feature_importance.csv", index=False)
    data_relationship_rows(frame).to_csv(OUT / "experimental_dataset_relationships.csv", index=False)

    residual_rows = []
    for (split_name, fold, configuration, target_range), group in prediction_frame.groupby(["split", "fold", "configuration", "target_range_training_quantiles"], sort=False):
        residual = group.residual_pred_minus_actual.to_numpy(dtype=float)
        residual_rows.append({
            "split": split_name, "fold": fold, "configuration": configuration,
            "target_range_training_quantiles": target_range, "count": len(group),
            "residual_mean": float(residual.mean()), "residual_std": float(np.std(residual, ddof=1)),
            "residual_q05": float(np.quantile(residual, .05)), "residual_median": float(np.median(residual)),
            "residual_q95": float(np.quantile(residual, .95)),
            "MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(residual**2))),
        })
    pd.DataFrame(residual_rows).to_csv(OUT / "experimental_model2_residual_diagnostics.csv", index=False)

    test_predictions = prediction_frame[(prediction_frame.split == "held_out_test") & (prediction_frame.configuration.isin(["full_stack_fixed_RF", "inner_selected_full_stack"]))]
    for group_column in ("Material_Type", "Material_Grade", "Tube_Type", "Braid_Type", "Machine_ID", "Shift", "Fault_Code"):
        merged = test_predictions.merge(test[["Record_ID", group_column]], left_on="record_id", right_on="Record_ID")
        for (value, configuration), group in merged.groupby([group_column, "configuration"], observed=True):
            residual = group.residual_pred_minus_actual.to_numpy(dtype=float)
            group_rows.append({"grouping": group_column, "value": value, "configuration": configuration, "count": len(group), "MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(residual**2))), "mean_residual": float(residual.mean())})
    pd.DataFrame(group_rows).to_csv(OUT / "test_generalization_by_group.csv", index=False)

    # Baseline metrics are read from previously saved controlled results, never refit here.
    baseline_summary = pd.read_csv(BASELINE_NESTED_DIR / "metrics_summary.csv")
    baseline_test = pd.read_csv(CONTROLLED_DIR / "model2_candidate/test_comparison.csv")
    baseline_summary.to_csv(OUT / "baseline_existing_dataset_nested_summary.csv", index=False)
    baseline_test.to_csv(OUT / "baseline_existing_dataset_test_comparison.csv", index=False)
    baseline_production = pd.DataFrame([
        {"dataset_version": "existing_dataset", "model": "saved_production_Model2", "split": "held_out_test", **BASELINE["model2_production_metrics"]["test_metrics"]},
        {"dataset_version": "existing_dataset", "model": "saved_production_Model2", "split": "saved_metadata_cv_not_fold_isolated", **BASELINE["model2_production_metrics"]["cv_metrics"]},
    ])
    baseline_production.to_csv(OUT / "baseline_existing_production_model2.csv", index=False)
    baseline_importance_rows = []
    baseline_model_paths = {
        "saved_production_model2": ROOT / "models/efficiency/efficiency_model.pkl",
        "saved_controlled_fixed_rf_candidate": CONTROLLED_DIR / "model2_candidate/candidate_artifacts/efficiency_model_candidate.pkl",
    }
    for label, path in baseline_model_paths.items():
        if not path.exists():
            continue
        fitted = joblib.load(path)
        transformed_names = fitted.named_steps["preprocess"].get_feature_names_out()
        model_importances = fitted.named_steps["model"].feature_importances_
        for feature, importance in zip(transformed_names, model_importances):
            baseline_importance_rows.append({"model": label, "feature": str(feature), "importance": float(importance)})
    pd.DataFrame(baseline_importance_rows).to_csv(OUT / "baseline_model2_feature_importance.csv", index=False)
    baseline_residual_path = CONTROLLED_DIR / "model2_candidate/test_residual_diagnostics.csv"
    if baseline_residual_path.exists():
        pd.read_csv(baseline_residual_path).to_csv(OUT / "baseline_existing_dataset_residual_diagnostics.csv", index=False)
    comparison_rows = []
    baseline_cv = baseline_summary[
        (baseline_summary.target == EFFICIENCY_TARGET)
        & (baseline_summary.split == "outer_cv")
    ]
    for _, row in baseline_cv.iterrows():
        comparison_rows.append({
            "dataset_version": "existing_synthetic_dataset",
            "evaluation": "corrected_nested_cv",
            "configuration": row.configuration,
            "fold_count": row.fold_count,
            "MAE_mean": row.MAE_mean, "MAE_std": row.MAE_std,
            "RMSE_mean": row.RMSE_mean, "RMSE_std": row.RMSE_std,
            "R2_mean": row.R2_mean, "R2_std": row.R2_std,
            "cross_dataset_scores_are_paired": False,
        })
    for _, row in baseline_test.iterrows():
        comparison_rows.append({
            "dataset_version": "existing_synthetic_dataset",
            "evaluation": "saved_test_comparison",
            "configuration": row["model"],
            "fold_count": 1,
            "MAE_mean": row["MAE_percentage_points"], "MAE_std": np.nan,
            "RMSE_mean": row["RMSE_percentage_points"], "RMSE_std": np.nan,
            "R2_mean": row["R2"], "R2_std": np.nan,
            "cross_dataset_scores_are_paired": False,
        })
    for _, row in metric_summary(fold_frame).iterrows():
        comparison_rows.append({
            "dataset_version": "experimental_synthetic_dataset",
            "evaluation": row["split"],
            "configuration": row["configuration"],
            "fold_count": row["fold_count"],
            "MAE_mean": row["MAE_mean"], "MAE_std": row["MAE_std"],
            "RMSE_mean": row["RMSE_mean"], "RMSE_std": row["RMSE_std"],
            "R2_mean": row["R2_mean"], "R2_std": row["R2_std"],
            "cross_dataset_scores_are_paired": False,
        })
    pd.DataFrame(comparison_rows).to_csv(OUT / "baseline_vs_experiment_metric_table.csv", index=False)

    run_manifest = {
        "experiment": "isolated_model2_generator_and_nested_stacking_comparison",
        "baseline_snapshot_path": BASELINE_PATH.relative_to(ROOT).as_posix(),
        "baseline_snapshot_sha256": digest(BASELINE_PATH),
        "production_dataset_unchanged_sha256": BASELINE["dataset_sha256"],
        "experimental_dataset_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "experimental_dataset_sha256": digest(OUTPUT_PATH),
        "experimental_dataset_shape": list(frame.shape),
        "seed": 42,
        "split_indices_source": SPLIT_INDICES_PATH.relative_to(ROOT).as_posix(),
        "split_configuration": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "shuffle": True, "train_rows": len(train), "test_rows": len(test)},
        "train_indices": [int(x) for x in train.index],
        "test_indices": [int(x) for x in test.index],
        "outer_cv_folds": OUTER_FOLDS,
        "upstream_selection_folds": reference.ALGORITHM_SELECTION_FOLDS,
        "upstream_oof_folds": reference.UPSTREAM_OOF_FOLDS,
        "downstream_selection_inner_folds": INNER_DOWNSTREAM_FOLDS,
        "downstream_candidates": list(DOWNSTREAM_CANDIDATES),
        "baseline_fixed_rf_parameters": FIXED_RF_PARAMS,
        "candidate_downstream_parameters": {name: params for name, (_, params) in reference.ALGORITHMS.items()},
        "final_downstream_selected_on_training_only": selected_downstream,
        "package_versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__},
        "generator_assumptions": [
            "Material/grade/tube/braid profiles and process bounds are inherited synthetic assumptions from the production generator, not verified factory values.",
            "Nominal line speed is an experiment-only reference computed from product specification and bounded to a synthetic 12-22 m/min envelope; this range is not a Polyhose limit.",
            "Realized line rate is a continuous beta-distributed fraction of nominal speed with a small synthetic machine effect; faults affect process stability, downtime, and scrap.",
            "Order_Status is Completed only when Good_Length_m meets Customer_Required_Length_m; Length_Shortfall_m remains planned process length minus actual processing length.",
            "Performance is realized line rate divided by synthetic nominal rate and is generated below one rather than clipped at one.",
            "Availability, Performance, and Quality remain separately computed and OEE=Availability*Performance*Quality; efficiency is OEE*100.",
            "Fault occurrence/severity remain stochastic and risk-dependent; unobserved machine/shift/fault effects intentionally limit prediction.",
        ],
        "split_note": "The existing saved row indices and 80/20 configuration are reused on the separately generated dataset. Same partition indices do not make dataset rows paired observations; cross-dataset scores are reported separately.",
        "no_test_selection": True,
        "no_production_writes": True,
        "output_files": [
            "experimental_dataset.csv",
            "experimental_dataset_audit.json",
            "experimental_dataset_relationships.csv",
            "experimental_model2_metrics_by_fold.csv",
            "experimental_model2_metric_summary.csv",
            "experimental_model2_predictions_residuals.csv",
            "experimental_model2_residual_diagnostics.csv",
            "experimental_model2_feature_importance.csv",
            "fold_local_algorithm_selection.csv",
            "test_generalization_by_group.csv",
            "baseline_existing_dataset_nested_summary.csv",
            "baseline_existing_dataset_test_comparison.csv",
            "baseline_existing_production_model2.csv",
            "baseline_vs_experiment_metric_table.csv",
            "baseline_model2_feature_importance.csv",
            "baseline_existing_dataset_residual_diagnostics.csv",
            "experiment_manifest.json",
            "experiment_report.md",
        ],
    }
    (OUT / "experiment_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_experiment_report(audit, fold_frame, run_manifest, selected_downstream)
    print("Experimental dataset SHA-256", run_manifest["experimental_dataset_sha256"])
    print("Selected alternative downstream algorithm on full training partition", selected_downstream)
    print(metric_summary(fold_frame).to_string(index=False))


def write_experiment_report(audit: dict, metrics: pd.DataFrame, manifest: dict, selected_downstream: str) -> None:
    summaries = metric_summary(metrics)
    baseline_nested = pd.read_csv(OUT / "baseline_existing_dataset_nested_summary.csv")
    baseline_test = pd.read_csv(OUT / "baseline_existing_dataset_test_comparison.csv")
    exp_nested = summaries[summaries.split == "nested_outer_cv"]
    exp_test = summaries[summaries.split == "saved_test"]
    predictions = pd.read_csv(OUT / "experimental_model2_predictions_residuals.csv")
    importance = pd.read_csv(OUT / "experimental_model2_feature_importance.csv")
    group_errors = pd.read_csv(OUT / "test_generalization_by_group.csv")
    full_oof = predictions[
        (predictions.split == "outer_cv")
        & (predictions.configuration == "full_stack_fixed_RF")
    ]
    full_bias = full_oof.groupby(
        "target_range_training_quantiles"
    ).residual_pred_minus_actual.mean()
    tuned_test = predictions[
        (predictions.split == "held_out_test")
        & (predictions.configuration == "inner_selected_full_stack")
    ].residual_pred_minus_actual
    top_features = importance[
        importance.training_scope == "full_training_partition"
    ].nlargest(5, "importance")
    breakdown = group_errors[
        (group_errors.grouping == "Fault_Code")
        & (group_errors.value == "Machine Breakdown")
    ]
    breakdown_mae = (
        float(breakdown.iloc[0].MAE) if not breakdown.empty else float("nan")
    )
    feature_summary = ", ".join(
        f"{row.feature} ({row.importance:.3f})"
        for row in top_features.itertuples(index=False)
    )
    bias_summary = ", ".join(
        f"{label}: {value:+.2f} pp"
        for label, value in full_bias.items()
    )
    test_residual_summary = (
        f"mean {tuned_test.mean():+.2f} pp; "
        f"5th/95th percentiles {tuned_test.quantile(0.05):+.2f}/"
        f"{tuned_test.quantile(0.95):+.2f} pp"
    )
    report = [
        "# Model 2 Generator Experiment",
        "",
        "## Scope and Preservation",
        "",
        f"Baseline dataset SHA-256: `{BASELINE['dataset_sha256']}`. Experimental dataset SHA-256: `{manifest['experimental_dataset_sha256']}`. Production dataset, model artifacts, metadata, generator, and inference code were not written.",
        "",
        "The production baseline metrics are saved results, not retrained. Existing nested-stack summaries are taken from the controlled evaluation. The experiment uses the same 80/20 split configuration and saved row-index partition on a different generated dataset; scores across dataset versions are not directly comparable as paired observations.",
        "",
        "## Generator Change",
        "",
        "Performance is now modeled as realized line output rate divided by a synthetic nominal recipe speed during actual runtime. Runtime is therefore not embedded in the Performance denominator and then counted again by Availability. Efficiency remains derived only from OEE components. Faults and downtime remain risk-dependent and stochastic.",
        "",
        "The nominal-speed formula, inherited material profiles, fault tables, and bounds are synthetic engineering-inspired assumptions, not factory-validated settings.",
        "",
        "## Experimental Dataset Checks",
        "",
        f"Shape: {audit['shape']}; missing cells: {audit['missing_cells']}; duplicate IDs: {audit['duplicate_record_ids']}; faulted rows: {audit['fault_count']}; `Braid_Type='None'`: {audit['braid_none_count']}; geometry max residual: {audit['geometry_max_abs_residual_mm']:.4f} mm.",
        "",
        f"The inherited status rule uses total actual processing length versus planned processing length, not good output versus customer-required length. {BASELINE['completed_good_length_below_customer_requirement_rows']} of {BASELINE['completed_rows']} baseline completed records and {audit['completed_good_length_below_customer_requirement_rows']} of {audit['completed_rows']} experimental completed records have good output below customer-required length. No project documentation clarifies whether this is an intended fulfillment policy; treat it as an unresolved synthetic accounting assumption.",
        "",
        "Accounting residuals and target summaries are in `experimental_dataset_audit.json`. All schema, category, geometry, bounds, and accounting assertions ran before model evaluation.",
        "",
        "## Nested CV Results",
        "",
        markdown_table(exp_nested),
        "",
        "## Held-Out Test Results",
        "",
        markdown_table(exp_test),
        "",
        "Baseline current-dataset nested CV results (separate dataset):",
        "",
        markdown_table(
            baseline_nested[
                (baseline_nested.target == EFFICIENCY_TARGET)
                & (baseline_nested.split == "outer_cv")
            ]
        ),
        "",
        "Baseline existing fixed-RF held-out comparisons (separate dataset):",
        "",
        markdown_table(baseline_test),
        "",
        "The inner-selected alternative was chosen using training-only inner folds and was not selected by test results. Group generalization by material, grade, tube, braid, machine, shift, and fault is in `test_generalization_by_group.csv`; residual summaries and feature importances are separate CSVs.",
        "",
        "## Residual and Feature-Importance Diagnostics",
        "",
        f"Across the full-stack fixed-RF outer-fold predictions, residual bias by training-derived target quartile is {bias_summary}. The inner-selected held-out residuals have {test_residual_summary}. Machine Breakdown remains a difficult small test subgroup (n=12, MAE={breakdown_mae:.2f} percentage points).",
        "",
        f"The selected full-training XGBoost feature importances are led by {feature_summary}. These are model-specific predictive importance scores, not causal effects.",
        "",
        "## Interpretation and Limitations",
        "",
        f"The fixed full-stack RF experimental nested CV is {exp_nested.loc[exp_nested.configuration == 'full_stack_fixed_RF', 'R2_mean'].iloc[0]:.3f} R²; the training-only selected alternative is {selected_downstream}. Do not interpret a cross-dataset score difference as measured improvement on the same population. Review MAE/RMSE, fold variance, fault-group errors, residuals, and model importances alongside R².",
        "",
        "No R² target was imposed. Synthetic data does not validate real Polyhose operations. In the experimental target, the median and 75th percentile of Performance are both 1.0, showing substantial ceiling saturation; this is a modeling assumption to validate rather than evidence of actual process performance. The performance definition and profile constants require manufacturing-engineer review before any generator change could be considered for production.",
        "",
        "## Recommendation",
        "",
        "Revise the generator assumptions further before any candidate training. The experiment removes a potential availability/performance double count but the nominal-speed reference remains unvalidated, Performance is heavily ceiling-saturated, and order status does not consistently reflect good output against customer requirements. Retain the current production generator and artifacts unchanged until engineering review resolves these assumptions; do not promote this dataset or train production artifacts automatically.",
        "",
    ]
    (OUT / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
