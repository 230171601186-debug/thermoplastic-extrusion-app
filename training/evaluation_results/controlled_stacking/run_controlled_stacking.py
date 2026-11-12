"""Leakage-controlled Model 2/3 stacking evaluation.

This experiment never writes to the dataset, production models, or production
metadata. Results and a reproducibility manifest are written beside this file.
"""

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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

TRAINING_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    CV_FOLDS,
    DATASET_CANDIDATES,
    EFFICIENCY_TARGET,
    LGBM_PARAMS,
    MODEL1_FEATURES,
    PROCESS_TARGETS,
    RANDOM_STATE,
    RF_PARAMS,
    SCRAP_TARGET,
    SPLIT_INDICES_PATH,
    TEST_SIZE,
    XGB_PARAMS,
)
from preprocessing import (  # noqa: E402
    build_customer_preprocessor,
    build_stacked_preprocessor,
    load_dataset,
    make_regressor_pipeline,
)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTER_FOLDS = 5
ALGORITHM_SELECTION_FOLDS = 3
UPSTREAM_OOF_FOLDS = 4
SELECTION_N_ESTIMATORS = {"Random Forest": 80, "XGBoost": 100, "LightGBM": 100}
DOWNSTREAM_PARAMS = deepcopy(RF_PARAMS)
DOWNSTREAM_PARAMS["n_estimators"] = 160

ALGORITHMS = {
    "Random Forest": (RandomForestRegressor, deepcopy(RF_PARAMS)),
    "XGBoost": (XGBRegressor, deepcopy(XGB_PARAMS)),
    "LightGBM": (LGBMRegressor, deepcopy(LGBM_PARAMS)),
}
for algorithm_name, (_, algorithm_params) in ALGORITHMS.items():
    algorithm_params["n_estimators"] = SELECTION_N_ESTIMATORS[algorithm_name]
    algorithm_params["n_jobs"] = 1

TARGETS = {
    EFFICIENCY_TARGET: {"unit": "percentage_points", "bounds": (0.0, 100.0)},
    SCRAP_TARGET: {"unit": "fraction", "bounds": (0.0, 1.0)},
}
CONFIGURATIONS = ("training_mean", "customer_only", "process_only", "full_stack")


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def selected_estimator(algorithm_name: str):
    estimator_class, params = ALGORITHMS[algorithm_name]
    return estimator_class(**deepcopy(params))


def make_process_only_pipeline():
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    [("num", "passthrough", PROCESS_TARGETS)],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("model", RandomForestRegressor(**deepcopy(DOWNSTREAM_PARAMS))),
        ]
    )


def make_downstream_pipeline(configuration: str):
    estimator = RandomForestRegressor(**deepcopy(DOWNSTREAM_PARAMS))
    if configuration == "customer_only":
        return make_regressor_pipeline(estimator)
    if configuration == "process_only":
        return make_process_only_pipeline()
    if configuration == "full_stack":
        return Pipeline(
            steps=[
                ("preprocess", build_stacked_preprocessor(PROCESS_TARGETS)),
                ("model", estimator),
            ]
        )
    raise ValueError(f"Not a fitted model configuration: {configuration}")


def select_upstream_algorithms(
    customer_inputs: pd.DataFrame,
    train_frame: pd.DataFrame,
    fold_label: str,
    selection_rows: list[dict],
) -> dict[str, str]:
    """Select each process estimator using only the supplied training frame."""
    splitter = KFold(
        n_splits=ALGORITHM_SELECTION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    selected = {}
    for target in PROCESS_TARGETS:
        scores = {}
        for algorithm_name in ALGORITHMS:
            pipeline = make_regressor_pipeline(selected_estimator(algorithm_name))
            fold_mae = -cross_val_score(
                pipeline,
                customer_inputs,
                train_frame[target],
                cv=splitter,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
            scores[algorithm_name] = float(fold_mae.mean())
            selection_rows.append(
                {
                    "scope": fold_label,
                    "target": target,
                    "algorithm": algorithm_name,
                    "selection_cv_mean_mae": scores[algorithm_name],
                }
            )
        selected[target] = min(scores, key=scores.get)
        for row in selection_rows:
            if row["scope"] == fold_label and row["target"] == target:
                row["selected"] = row["algorithm"] == selected[target]
    return selected


def build_cross_fitted_process_features(
    train_frame: pd.DataFrame,
    selected_algorithms: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create OOF process predictions, then fit final upstream models."""
    customer_inputs = train_frame[MODEL1_FEATURES].copy()
    result = customer_inputs.copy()
    final_models = {}
    splitter = KFold(
        n_splits=UPSTREAM_OOF_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + 1,
    )
    for target in PROCESS_TARGETS:
        estimator = selected_estimator(selected_algorithms[target])
        pipeline = make_regressor_pipeline(estimator)
        result[target] = cross_val_predict(
            pipeline,
            customer_inputs,
            train_frame[target],
            cv=splitter,
            n_jobs=1,
        )
        final_pipeline = make_regressor_pipeline(selected_estimator(selected_algorithms[target]))
        final_pipeline.fit(customer_inputs, train_frame[target])
        final_models[target] = final_pipeline
    if list(result.columns) != MODEL1_FEATURES + PROCESS_TARGETS:
        raise RuntimeError("Cross-fitted stack has an unexpected feature schema")
    return result, final_models


def predict_process_features(
    customer_inputs: pd.DataFrame,
    final_models: dict[str, object],
) -> pd.DataFrame:
    result = customer_inputs[MODEL1_FEATURES].copy()
    for target in PROCESS_TARGETS:
        result[target] = final_models[target].predict(customer_inputs[MODEL1_FEATURES])
    if list(result.columns) != MODEL1_FEATURES + PROCESS_TARGETS:
        raise RuntimeError("Predicted stack has an unexpected feature schema")
    return result


def target_range_labels(
    training_target: pd.Series,
    evaluation_target: pd.Series,
    target_name: str,
) -> tuple[pd.Series, dict[str, float]]:
    if target_name == SCRAP_TARGET:
        q_low, q_high = np.quantile(training_target, [1 / 3, 2 / 3])
        edges = [-np.inf, float(q_low), float(q_high), np.inf]
        labels = ["low", "medium", "high"]
        cuts = {"low_medium": float(q_low), "medium_high": float(q_high)}
    else:
        quantiles = np.unique(np.quantile(training_target, [0.0, 0.25, 0.5, 0.75, 1.0]))
        if len(quantiles) < 3:
            edges = [-np.inf, float(np.median(training_target)), np.inf]
        else:
            edges = [-np.inf, *[float(value) for value in quantiles[1:-1]], np.inf]
        labels = [f"range_{index + 1}" for index in range(len(edges) - 1)]
        cuts = {f"edge_{index}": float(value) for index, value in enumerate(edges[1:-1], start=1)}
    categories = pd.cut(evaluation_target, bins=edges, labels=labels, include_lowest=True)
    return categories.astype(str), cuts


def save_prediction_records(
    records: list[dict],
    target_name: str,
    split_name: str,
    fold_label: str,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    target_cutoffs: dict[str, float],
) -> None:
    actual = eval_frame[target_name].to_numpy(dtype=float)
    ranges, _ = target_range_labels(train_frame[target_name], eval_frame[target_name], target_name)
    for configuration, predicted in predictions.items():
        for index, (observed, estimate) in enumerate(zip(actual, predicted)):
            records.append(
                {
                    "split": split_name,
                    "fold": fold_label,
                    "target": target_name,
                    "configuration": configuration,
                    "record_id": int(eval_frame.iloc[index]["Record_ID"]),
                    "actual": float(observed),
                    "predicted": float(estimate),
                    "residual_pred_minus_actual": float(estimate - observed),
                    "absolute_error": float(abs(estimate - observed)),
                    "target_range": ranges.iloc[index],
                    "target_low_medium_cut": target_cutoffs.get("low_medium"),
                    "target_medium_high_cut": target_cutoffs.get("medium_high"),
                }
            )


def fit_predict_configurations(
    train_features: dict[str, pd.DataFrame],
    eval_features: dict[str, pd.DataFrame],
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target_name: str,
) -> dict[str, np.ndarray]:
    target_train = train_frame[target_name].to_numpy(dtype=float)
    predictions = {
        "training_mean": np.full(
            len(eval_frame), float(target_train.mean()), dtype=float
        )
    }
    for configuration in ("customer_only", "process_only", "full_stack"):
        pipeline = make_downstream_pipeline(configuration)
        pipeline.fit(train_features[configuration], target_train)
        predictions[configuration] = np.asarray(
            pipeline.predict(eval_features[configuration]), dtype=float
        )
    return predictions


def add_split_results(
    fold_rows: list[dict],
    prediction_rows: list[dict],
    diagnostic_rows: list[dict],
    split_name: str,
    fold_label: str,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    training_stack: pd.DataFrame,
    evaluation_stack: pd.DataFrame,
) -> None:
    for target_name, target_metadata in TARGETS.items():
        train_features = {
            "customer_only": train_frame[MODEL1_FEATURES].copy(),
            "process_only": training_stack[PROCESS_TARGETS].copy(),
            "full_stack": training_stack[MODEL1_FEATURES + PROCESS_TARGETS].copy(),
        }
        eval_features = {
            "customer_only": eval_frame[MODEL1_FEATURES].copy(),
            "process_only": evaluation_stack[PROCESS_TARGETS].copy(),
            "full_stack": evaluation_stack[MODEL1_FEATURES + PROCESS_TARGETS].copy(),
        }
        predictions = fit_predict_configurations(
            train_features,
            eval_features,
            train_frame,
            eval_frame,
            target_name,
        )
        truth = eval_frame[target_name].to_numpy(dtype=float)
        ranges, cutoffs = target_range_labels(
            train_frame[target_name], eval_frame[target_name], target_name
        )
        for configuration, predicted in predictions.items():
            scores = metric_values(truth, predicted)
            row = {
                "split": split_name,
                "fold": fold_label,
                "target": target_name,
                "configuration": configuration,
                **scores,
            }
            if target_metadata["unit"] == "fraction":
                row["MAE_percentage_points"] = scores["MAE"] * 100.0
                row["RMSE_percentage_points"] = scores["RMSE"] * 100.0
            fold_rows.append(row)
            save_prediction_records(
                prediction_rows,
                target_name,
                split_name,
                fold_label,
                train_frame,
                eval_frame,
                {configuration: predicted},
                cutoffs,
            )
            lower, upper = target_metadata["bounds"]
            diagnostic_rows.append(
                {
                    "split": split_name,
                    "fold": fold_label,
                    "target": target_name,
                    "configuration": configuration,
                    "unit": target_metadata["unit"],
                    "prediction_count": int(len(predicted)),
                    "below_valid_bound_count": int(np.sum(predicted < lower)),
                    "above_valid_bound_count": int(np.sum(predicted > upper)),
                    "minimum_prediction": float(np.min(predicted)),
                    "maximum_prediction": float(np.max(predicted)),
                }
            )
        if target_name == SCRAP_TARGET:
            high_threshold = float(train_frame[target_name].quantile(0.80))
            true_high = truth >= high_threshold
            for configuration, predicted in predictions.items():
                predicted_high = predicted >= high_threshold
                diagnostic_rows.append(
                    {
                        "split": split_name,
                        "fold": fold_label,
                        "target": target_name,
                        "configuration": configuration,
                        "diagnostic": "high_scrap_detection",
                        "high_scrap_threshold_training_q80": high_threshold,
                        "precision": float(precision_score(true_high, predicted_high, zero_division=0)),
                        "recall": float(recall_score(true_high, predicted_high, zero_division=0)),
                        "F1": float(f1_score(true_high, predicted_high, zero_division=0)),
                        "actual_high_count": int(true_high.sum()),
                        "predicted_high_count": int(predicted_high.sum()),
                    }
                )


def validate_saved_split(df: pd.DataFrame, split_payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    saved_train = split_payload["train_index"]
    saved_test = split_payload["test_index"]
    expected_train, expected_test = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    if list(expected_train) != saved_train or list(expected_test) != saved_test:
        raise RuntimeError("Saved split differs from the configured Model 1 split")
    if set(saved_train) & set(saved_test) or len(saved_train) + len(saved_test) != len(df):
        raise RuntimeError("Saved split indices overlap or do not cover the dataset")
    return df.loc[saved_train].copy(), df.loc[saved_test].copy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = next((candidate for candidate in DATASET_CANDIDATES if candidate.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Configured dataset file is unavailable")
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    df = load_dataset()
    if df.shape != (5000, 40):
        raise RuntimeError(f"Expected the preserved 5000x40 dataset, found {df.shape}")
    if df[MODEL1_FEATURES + PROCESS_TARGETS].isna().any().any():
        raise RuntimeError("Missing model feature/target values prevent this evaluation")
    if "None" not in set(df["Braid_Type"].astype(str)):
        raise RuntimeError("Expected Braid_Type='None' to remain a legitimate category")
    if list(MODEL1_FEATURES) != CATEGORICAL_FEATURES + CUSTOMER_NUMERIC_FEATURES:
        raise RuntimeError("Model 1 feature ordering differs from preprocessing configuration")

    split_payload = json.loads(SPLIT_INDICES_PATH.read_text(encoding="utf-8"))
    train_frame, test_frame = validate_saved_split(df, split_payload)
    if not set(train_frame.index).isdisjoint(test_frame.index):
        raise RuntimeError("Training and test rows overlap")

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    selection_rows: list[dict] = []
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_positions = np.arange(len(train_frame))
    for fold_number, (outer_train_pos, outer_valid_pos) in enumerate(
        outer.split(train_positions), start=1
    ):
        outer_train = train_frame.iloc[outer_train_pos].copy()
        outer_valid = train_frame.iloc[outer_valid_pos].copy()
        label = f"outer_{fold_number}"
        selected = select_upstream_algorithms(
            outer_train[MODEL1_FEATURES], outer_train, label, selection_rows
        )
        training_stack, final_upstream = build_cross_fitted_process_features(
            outer_train, selected
        )
        validation_stack = predict_process_features(
            outer_valid[MODEL1_FEATURES], final_upstream
        )
        add_split_results(
            fold_rows,
            prediction_rows,
            diagnostic_rows,
            "outer_cv",
            label,
            outer_train,
            outer_valid,
            training_stack,
            validation_stack,
        )

    # The existing held-out partition is evaluated once after all choices are fixed.
    test_selected = select_upstream_algorithms(
        train_frame[MODEL1_FEATURES], train_frame, "full_training_partition", selection_rows
    )
    test_training_stack, test_upstream = build_cross_fitted_process_features(
        train_frame, test_selected
    )
    test_stack = predict_process_features(test_frame[MODEL1_FEATURES], test_upstream)
    add_split_results(
        fold_rows,
        prediction_rows,
        diagnostic_rows,
        "held_out_test",
        "saved_test",
        train_frame,
        test_frame,
        test_training_stack,
        test_stack,
    )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(OUTPUT_DIR / "metrics_by_fold.csv", index=False)
    cv_metrics = fold_metrics[fold_metrics["split"] == "outer_cv"]
    metric_columns = ["MAE", "RMSE", "R2"]
    summary_rows = []
    for (target, configuration), group in cv_metrics.groupby(
        ["target", "configuration"], sort=False
    ):
        summary = {
            "split": "outer_cv",
            "target": target,
            "configuration": configuration,
            "fold_count": int(group["fold"].nunique()),
        }
        for metric in metric_columns:
            summary[f"{metric}_mean"] = float(group[metric].mean())
            summary[f"{metric}_std"] = float(group[metric].std(ddof=1))
        if target == SCRAP_TARGET:
            for metric in ("MAE_percentage_points", "RMSE_percentage_points"):
                summary[f"{metric}_mean"] = float(group[metric].mean())
                summary[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(summary)
    test_metrics = fold_metrics[fold_metrics["split"] == "held_out_test"].copy()
    for _, row in test_metrics.iterrows():
        summary = {
            "split": "held_out_test",
            "target": row["target"],
            "configuration": row["configuration"],
            "fold_count": 1,
        }
        for metric in metric_columns:
            summary[f"{metric}_mean"] = row[metric]
            summary[f"{metric}_std"] = np.nan
        if row["target"] == SCRAP_TARGET:
            summary["MAE_percentage_points_mean"] = row["MAE_percentage_points"]
            summary["MAE_percentage_points_std"] = np.nan
            summary["RMSE_percentage_points_mean"] = row["RMSE_percentage_points"]
            summary["RMSE_percentage_points_std"] = np.nan
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(OUTPUT_DIR / "predictions_and_residuals.csv", index=False)
    residual_rows = []
    for keys, group in predictions.groupby(
        ["split", "fold", "target", "configuration", "target_range"],
        observed=True,
        sort=False,
    ):
        split_name, fold_label, target_name, configuration, target_range = keys
        residual = group["residual_pred_minus_actual"]
        residual_rows.append(
            {
                "split": split_name,
                "fold": fold_label,
                "target": target_name,
                "configuration": configuration,
                "target_range": target_range,
                "count": int(len(group)),
                "residual_mean": float(residual.mean()),
                "residual_std": float(residual.std(ddof=1)),
                "residual_q05": float(residual.quantile(0.05)),
                "residual_median": float(residual.median()),
                "residual_q95": float(residual.quantile(0.95)),
                "MAE": float(group["absolute_error"].mean()),
                "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
            }
        )
    pd.DataFrame(residual_rows).to_csv(OUTPUT_DIR / "residual_diagnostics.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(OUTPUT_DIR / "prediction_and_scrap_diagnostics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUTPUT_DIR / "upstream_selection.csv", index=False)

    train_indices = [int(index) for index in train_frame.index]
    test_indices = [int(index) for index in test_frame.index]
    manifest = {
        "experiment": "controlled_nested_stacking_evaluation",
        "dataset_path": dataset_path.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": dataset_sha256,
        "dataset_shape": list(df.shape),
        "dataset_columns": list(df.columns),
        "seed": RANDOM_STATE,
        "saved_split_path": SPLIT_INDICES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "saved_split_configuration": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "shuffle": True,
            "train_rows": len(train_frame),
            "test_rows": len(test_frame),
        },
        "train_indices": train_indices,
        "test_indices": test_indices,
        "outer_folds": OUTER_FOLDS,
        "algorithm_selection_folds": ALGORITHM_SELECTION_FOLDS,
        "upstream_oof_folds": UPSTREAM_OOF_FOLDS,
        "upstream_selection_estimators": SELECTION_N_ESTIMATORS,
        "downstream_estimator": "RandomForestRegressor",
        "downstream_parameters": DOWNSTREAM_PARAMS,
        "features": {
            "customer_product": MODEL1_FEATURES,
            "process_parameters": PROCESS_TARGETS,
            "process_only_order": PROCESS_TARGETS,
            "full_stack_order": MODEL1_FEATURES + PROCESS_TARGETS,
        },
        "feature_units": {
            "Material_Type": "categorical",
            "Material_Grade": "categorical",
            "Tube_Type": "categorical",
            "Braid_Type": "categorical; 'None' is a valid category",
            "Target_Tube_OD_mm": "mm",
            "Target_Tube_Wall_mm": "mm",
            "Target_Tube_ID_mm": "mm",
            "Customer_Required_Length_m": "m",
            "Extruder_Temperature_C": "degC",
            "Die_Temperature_C": "degC",
            "Cooling_Tank_Temperature_C": "degC",
            "Melt_Pressure_bar": "bar",
            "Air_Pressure_bar": "bar",
            "Screw_RPM": "rpm",
            "Line_Speed_m_min": "m/min",
            "Puller_Speed_m_min": "m/min",
            "Extrusion_Speed_kg_hr": "kg/hr",
        },
        "units": {target: values["unit"] for target, values in TARGETS.items()},
        "package_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "method_notes": [
            "Outer-fold validation rows are excluded from upstream algorithm selection, preprocessing fits, OOF model fits, and downstream fitting.",
            "Within each outer training partition, process algorithms are selected by 3-fold mean MAE; their fixed-parameter pipelines generate 4-fold OOF process features.",
            "Each OOF prediction is produced by a pipeline whose preprocessing and estimator are fit without that row.",
            "The downstream Random Forest is fixed a priori and shared across customer-only, process-only, and full-stack configurations; no downstream model selection or tuning is performed.",
            "Estimator counts are reduced for the upstream selection candidates and downstream estimator to keep the experiment computationally bounded; this is an evaluation protocol, not a production model replacement.",
            "The saved test partition is evaluated once after the training-only process algorithm selection; no test result selects an algorithm or threshold.",
            "Scrap bands use outer-training tertiles and high-scrap classification uses the outer-training 80th percentile; held-out test thresholds use the full training partition only.",
            "No production model, metadata, dataset, or application file is loaded for writing or modified by this script.",
        ],
    }
    (OUTPUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Saved controlled stacking results to {OUTPUT_DIR}")
    print(f"Dataset SHA-256: {dataset_sha256}")
    print(f"Preserved split: train={len(train_frame)}, test={len(test_frame)}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()