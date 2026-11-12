"""Train and validate a corrected Model 3 candidate without promotion.

The production dataset, metadata, and model artifacts are read-only inputs.
Candidate pipelines and diagnostics are written beneath model3_candidate/.
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
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

TRAINING_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DATASET_CANDIDATES,
    DISCLAIMER,
    LGBM_PARAMS,
    METADATA_PATH,
    MODEL1_FEATURES,
    PLANNING_CONTEXT_EXCLUDED,
    POST_PRODUCTION_LEAKAGE,
    PROCESS_MODEL_FILENAMES,
    PROCESS_TARGETS,
    QUALITY_MODELS_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    SCRAP_TARGET,
    SPLIT_INDICES_PATH,
    TEST_SIZE,
    XGB_PARAMS,
)
from preprocessing import (  # noqa: E402
    load_dataset,
    make_regressor_pipeline,
    make_stacked_pipeline,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "model3_candidate"
ARTIFACT_DIR = OUTPUT_DIR / "candidate_artifacts"
CONTROLLED_METRICS_PATH = EXPERIMENT_DIR / "metrics_by_fold.csv"
OUTER_FOLDS = 5
SELECTION_FOLDS = 3
OOF_FOLDS = 4
SELECTION_ESTIMATORS = {"Random Forest": 80, "XGBoost": 100, "LightGBM": 100}
DOWNSTREAM_PARAMS = deepcopy(RF_PARAMS)
DOWNSTREAM_PARAMS["n_estimators"] = 160
DOWNSTREAM_PARAMS["n_jobs"] = 1

ALGORITHMS = {
    "Random Forest": (RandomForestRegressor, deepcopy(RF_PARAMS)),
    "XGBoost": (XGBRegressor, deepcopy(XGB_PARAMS)),
    "LightGBM": (LGBMRegressor, deepcopy(LGBM_PARAMS)),
}
for algorithm_name, (_, params) in ALGORITHMS.items():
    params["n_estimators"] = SELECTION_ESTIMATORS[algorithm_name]
    params["n_jobs"] = 1

PROHIBITED_FEATURES = set(POST_PRODUCTION_LEAKAGE) | set(PLANNING_CONTEXT_EXCLUDED)
CANDIDATE_ALGORITHM = "Random Forest"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def verify_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_train, expected_test = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    if list(expected_train) != split["train_index"] or list(expected_test) != split["test_index"]:
        raise RuntimeError("Saved indices do not match the configured original split")
    train = df.loc[split["train_index"]].copy()
    test = df.loc[split["test_index"]].copy()
    if len(train) != 4000 or len(test) != 1000 or not set(train.index).isdisjoint(test.index):
        raise RuntimeError("Saved training/test partition is invalid")
    return train, test


def select_process_algorithms(
    train: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    splitter = KFold(n_splits=SELECTION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    selections = {}
    rows = []
    customer_inputs = train[MODEL1_FEATURES].copy()
    for target in PROCESS_TARGETS:
        scores = {}
        for algorithm_name, (estimator_class, params) in ALGORITHMS.items():
            pipeline = make_regressor_pipeline(estimator_class(**deepcopy(params)))
            fold_mae = -cross_val_score(
                pipeline,
                customer_inputs,
                train[target],
                cv=splitter,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
            scores[algorithm_name] = float(fold_mae.mean())
            rows.append(
                {
                    "target": target,
                    "algorithm": algorithm_name,
                    "selection_cv_mean_mae": scores[algorithm_name],
                }
            )
        selections[target] = min(scores, key=scores.get)
        for row in rows:
            if row["target"] == target:
                row["selected"] = row["algorithm"] == selections[target]
    return selections, pd.DataFrame(rows)


def fit_oof_and_final_process_models(
    train: pd.DataFrame,
    selections: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Build row-excluded OOF features and final training-only pipelines."""
    customer_inputs = train[MODEL1_FEATURES].copy()
    stacked_train = customer_inputs.copy()
    final_models = {}
    audit_rows = []
    splitter = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1)

    for target in PROCESS_TARGETS:
        oof_predictions = np.full(len(train), np.nan, dtype=float)
        prediction_counts = np.zeros(len(train), dtype=np.int8)
        # A row's training feature must come from an upstream pipeline that
        # excluded that row, including its preprocessing fit.
        for fold_number, (fit_positions, held_positions) in enumerate(
            splitter.split(customer_inputs), start=1
        ):
            fit_indices = train.index.to_numpy()[fit_positions]
            held_indices = train.index.to_numpy()[held_positions]
            overlap = set(fit_indices) & set(held_indices)
            if overlap:
                raise RuntimeError(f"OOF fold contains fit/held-out overlap for {target}")

            estimator_class, params = ALGORITHMS[selections[target]]
            pipeline = make_regressor_pipeline(estimator_class(**deepcopy(params)))
            pipeline.fit(
                customer_inputs.iloc[fit_positions],
                train[target].iloc[fit_positions],
            )
            oof_predictions[held_positions] = pipeline.predict(
                customer_inputs.iloc[held_positions]
            )
            prediction_counts[held_positions] += 1
            audit_rows.append(
                {
                    "target": target,
                    "fold": fold_number,
                    "fitted_rows": int(len(fit_positions)),
                    "predicted_rows": int(len(held_positions)),
                    "fit_prediction_row_overlap": len(overlap),
                    "selected_algorithm": selections[target],
                }
            )

        if np.isnan(oof_predictions).any() or not np.all(prediction_counts == 1):
            raise RuntimeError(f"OOF predictions are missing or duplicated for {target}")
        stacked_train[target] = oof_predictions

        estimator_class, params = ALGORITHMS[selections[target]]
        final_pipeline = make_regressor_pipeline(estimator_class(**deepcopy(params)))
        final_pipeline.fit(customer_inputs, train[target])
        final_models[target] = final_pipeline

    expected_columns = MODEL1_FEATURES + PROCESS_TARGETS
    if list(stacked_train.columns) != expected_columns:
        raise RuntimeError("OOF training feature order differs from inference contract")
    if set(stacked_train.columns) & PROHIBITED_FEATURES:
        raise RuntimeError("A prohibited post-production/context feature entered Model 3")
    return stacked_train, final_models, pd.DataFrame(audit_rows)


def predict_stack(customer_features: pd.DataFrame, upstream_models: dict[str, object]) -> pd.DataFrame:
    stacked = customer_features[MODEL1_FEATURES].copy()
    for target in PROCESS_TARGETS:
        stacked[target] = upstream_models[target].predict(customer_features[MODEL1_FEATURES])
    expected_columns = MODEL1_FEATURES + PROCESS_TARGETS
    if list(stacked.columns) != expected_columns:
        raise RuntimeError("Inference feature order differs from OOF training contract")
    if set(stacked.columns) & PROHIBITED_FEATURES:
        raise RuntimeError("A prohibited post-production/context feature entered inference")
    return stacked


def range_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, float, float]:
    y_train = train[SCRAP_TARGET].to_numpy(dtype=float)
    y_test = test[SCRAP_TARGET].to_numpy(dtype=float)
    low_boundary, high_boundary = np.quantile(y_train, [1 / 3, 2 / 3])
    band = np.select(
        [y_test <= low_boundary, y_test <= high_boundary],
        ["low", "medium"],
        default="high",
    )
    rows = []
    for name, predicted in predictions.items():
        residual = predicted - y_test
        rows.append(
            {
                "model": name,
                "target_band": "all",
                "count": len(y_test),
                "MAE_fraction": float(mean_absolute_error(y_test, predicted)),
                "MAE_percentage_points": float(mean_absolute_error(y_test, predicted) * 100),
                "RMSE_fraction": float(np.sqrt(mean_squared_error(y_test, predicted))),
                "mean_residual_fraction": float(residual.mean()),
                "prediction_below_zero_count": int(np.sum(predicted < 0.0)),
                "prediction_above_one_count": int(np.sum(predicted > 1.0)),
                "prediction_min": float(predicted.min()),
                "prediction_max": float(predicted.max()),
            }
        )
        for target_band in ("low", "medium", "high"):
            mask = band == target_band
            band_actual = y_test[mask]
            band_predicted = predicted[mask]
            band_residual = band_predicted - band_actual
            rows.append(
                {
                    "model": name,
                    "target_band": target_band,
                    "count": int(mask.sum()),
                    "MAE_fraction": float(mean_absolute_error(band_actual, band_predicted)),
                    "MAE_percentage_points": float(mean_absolute_error(band_actual, band_predicted) * 100),
                    "RMSE_fraction": float(np.sqrt(mean_squared_error(band_actual, band_predicted))),
                    "mean_residual_fraction": float(band_residual.mean()),
                    "prediction_below_zero_count": int(np.sum(predicted < 0.0)),
                    "prediction_above_one_count": int(np.sum(predicted > 1.0)),
                    "prediction_min": float(predicted.min()),
                    "prediction_max": float(predicted.max()),
                }
            )
    return pd.DataFrame(rows), float(low_boundary), float(high_boundary)


def high_scrap_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict[str, object]:
    actual_high = actual >= threshold
    predicted_high = predicted >= threshold
    return {
        "model": model_name,
        "training_q80_threshold_fraction": threshold,
        "actual_high_count": int(actual_high.sum()),
        "predicted_high_count": int(predicted_high.sum()),
        "precision": float(precision_score(actual_high, predicted_high, zero_division=0)),
        "recall": float(recall_score(actual_high, predicted_high, zero_division=0)),
        "F1": float(f1_score(actual_high, predicted_high, zero_division=0)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Configured dataset file is missing")
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    df = load_dataset()
    if df.shape != (5000, 40):
        raise RuntimeError(f"Expected preserved 5000x40 dataset, got {df.shape}")
    if "None" not in set(df["Braid_Type"].astype(str)):
        raise RuntimeError("Braid_Type='None' was not preserved as a valid category")
    if df[MODEL1_FEATURES + PROCESS_TARGETS + [SCRAP_TARGET]].isna().any().any():
        raise RuntimeError("Missing required feature or target values")
    if set(MODEL1_FEATURES + PROCESS_TARGETS) & PROHIBITED_FEATURES:
        raise RuntimeError("Model 3 feature contract includes a prohibited feature")

    split = json.loads(SPLIT_INDICES_PATH.read_text(encoding="utf-8"))
    train, test = verify_split(df, split)
    upstream_selection, selection_table = select_process_algorithms(train)
    selection_table.to_csv(OUTPUT_DIR / "upstream_algorithm_selection.csv", index=False)
    train_stack, final_upstream, oof_audit = fit_oof_and_final_process_models(
        train, upstream_selection
    )
    # Test and later inference rows use final upstream pipelines fit only on
    # the complete training partition, never on held-out target rows.
    test_stack = predict_stack(test[MODEL1_FEATURES], final_upstream)
    expected_features = MODEL1_FEATURES + PROCESS_TARGETS
    if list(train_stack.columns) != list(test_stack.columns) or list(test_stack.columns) != expected_features:
        raise RuntimeError("Training and test stack feature schemas differ")
    if len(train_stack) != len(train) or len(test_stack) != len(test):
        raise RuntimeError("Stack feature row counts differ from the saved split")
    oof_audit.to_csv(OUTPUT_DIR / "upstream_oof_fold_audit.csv", index=False)

    # Candidate estimator is fixed to the controlled experiment's RF configuration;
    # the saved test set is not used to select or tune it.
    candidate_pipeline = make_stacked_pipeline(
        RandomForestRegressor(**deepcopy(DOWNSTREAM_PARAMS)),
        PROCESS_TARGETS,
    )
    candidate_pipeline.fit(train_stack, train[SCRAP_TARGET])
    candidate_test_prediction = np.asarray(candidate_pipeline.predict(test_stack), dtype=float)

    # Score the existing production artifacts read-only on the exact same test rows.
    production_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    production_model_path = PROJECT_ROOT / production_metadata["model3"]["file"]
    production_downstream = joblib.load(production_model_path)
    from preprocessing import attach_predicted_process, load_process_pipelines

    production_upstream = load_process_pipelines()
    production_test_stack = attach_predicted_process(test[MODEL1_FEATURES], production_upstream)
    if list(production_test_stack.columns) != expected_features:
        raise RuntimeError("Production artifact test features do not match the contract")
    production_test_prediction = np.asarray(
        production_downstream.predict(production_test_stack), dtype=float
    )

    customer_pipeline = make_stacked_pipeline(
        RandomForestRegressor(**deepcopy(DOWNSTREAM_PARAMS)),
        [],
    )
    customer_pipeline.fit(train[MODEL1_FEATURES], train[SCRAP_TARGET])
    customer_test_prediction = np.asarray(
        customer_pipeline.predict(test[MODEL1_FEATURES]), dtype=float
    )
    train_mean_prediction = np.full(len(test), float(train[SCRAP_TARGET].mean()))
    y_test = test[SCRAP_TARGET].to_numpy(dtype=float)

    test_predictions = {
        "training_mean": train_mean_prediction,
        "customer_only_fixed_RF": customer_test_prediction,
        "current_production_artifacts": production_test_prediction,
        "corrected_candidate_full_stack_RF": candidate_test_prediction,
    }
    comparison_rows = []
    for model_name, prediction in test_predictions.items():
        metrics = regression_metrics(y_test, prediction)
        comparison_rows.append(
            {
                "split": "saved_test_1000_rows",
                "model": model_name,
                "MAE_fraction": metrics["MAE"],
                "MAE_percentage_points": metrics["MAE"] * 100,
                "RMSE_fraction": metrics["RMSE"],
                "RMSE_percentage_points": metrics["RMSE"] * 100,
                "R2": metrics["R2"],
                "prediction_below_zero_count": int(np.sum(prediction < 0.0)),
                "prediction_above_one_count": int(np.sum(prediction > 1.0)),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "test_comparison.csv", index=False)

    # Reuse the already-completed nested outer-fold evaluation for this exact
    # fixed RF protocol; its fold scores were generated without test access.
    if not CONTROLLED_METRICS_PATH.exists():
        raise FileNotFoundError("Run the controlled nested evaluation before candidate training")
    controlled_metrics = pd.read_csv(CONTROLLED_METRICS_PATH)
    nested_cv = controlled_metrics.loc[
        (controlled_metrics["split"] == "outer_cv")
        & (controlled_metrics["target"] == SCRAP_TARGET)
        & (controlled_metrics["configuration"] == "full_stack")
    ].copy()
    if nested_cv["fold"].nunique() != OUTER_FOLDS or len(nested_cv) != OUTER_FOLDS:
        raise RuntimeError("Expected five existing leakage-controlled full-stack CV folds")
    nested_cv.to_csv(OUTPUT_DIR / "nested_cv_candidate_metrics.csv", index=False)
    summary = {
        "split": "nested_outer_cv",
        "model": "corrected_candidate_full_stack_RF",
        "fold_count": OUTER_FOLDS,
    }
    for metric in ("MAE", "RMSE", "R2"):
        summary[f"{metric}_mean"] = float(nested_cv[metric].mean())
        summary[f"{metric}_std"] = float(nested_cv[metric].std(ddof=1))
    summary["MAE_percentage_points_mean"] = summary["MAE_mean"] * 100
    summary["MAE_percentage_points_std"] = summary["MAE_std"] * 100
    summary["RMSE_percentage_points_mean"] = summary["RMSE_mean"] * 100
    summary["RMSE_percentage_points_std"] = summary["RMSE_std"] * 100
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "nested_cv_candidate_summary.csv", index=False)

    range_table, _, _ = range_diagnostics(
        train,
        test,
        {
            "current_production_artifacts": production_test_prediction,
            "corrected_candidate_full_stack_RF": candidate_test_prediction,
        },
    )
    range_table.to_csv(OUTPUT_DIR / "scrap_band_diagnostics.csv", index=False)
    high_threshold = float(train[SCRAP_TARGET].quantile(0.80))
    high_rows = [
        high_scrap_metrics(y_test, prediction, high_threshold, model_name)
        for model_name, prediction in test_predictions.items()
    ]
    pd.DataFrame(high_rows).to_csv(OUTPUT_DIR / "high_scrap_secondary_metrics.csv", index=False)

    # Candidate-only writes; production artifacts and metadata remain untouched.
    for target, filename in PROCESS_MODEL_FILENAMES.items():
        joblib.dump(final_upstream[target], ARTIFACT_DIR / filename)
    candidate_model_path = ARTIFACT_DIR / "scrap_rate_model_candidate.pkl"
    joblib.dump(candidate_pipeline, candidate_model_path)

    # Exercise categorical encoding with an unseen value; ignore handling must work.
    unseen_probe = test[MODEL1_FEATURES].head(1).copy()
    unseen_probe.loc[unseen_probe.index[0], "Braid_Type"] = "__unseen_category_probe__"
    unseen_stack = predict_stack(unseen_probe, final_upstream)
    unseen_prediction = candidate_pipeline.predict(unseen_stack)
    if unseen_prediction.shape != (1,) or not np.isfinite(unseen_prediction).all():
        raise RuntimeError("Candidate cannot handle an unseen categorical value")

    # Verify the saved candidate objects support the exact customer-input inference path.
    saved_upstream = {
        target: joblib.load(ARTIFACT_DIR / filename)
        for target, filename in PROCESS_MODEL_FILENAMES.items()
    }
    saved_downstream = joblib.load(candidate_model_path)
    saved_probe_stack = predict_stack(test[MODEL1_FEATURES].head(3), saved_upstream)
    saved_probe_prediction = saved_downstream.predict(saved_probe_stack)
    if saved_probe_prediction.shape != (3,) or not np.isfinite(saved_probe_prediction).all():
        raise RuntimeError("Serialized candidate artifacts failed inference validation")

    manifest = {
        "experiment": "model3_corrected_stacking_candidate",
        "disclaimer": DISCLAIMER,
        "dataset_path": dataset_path.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": dataset_hash,
        "dataset_shape": list(df.shape),
        "seed": RANDOM_STATE,
        "split_path": SPLIT_INDICES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_definition": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "shuffle": True,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_indices": [int(value) for value in train.index],
            "test_indices": [int(value) for value in test.index],
        },
        "model1_selection": {
            "cv_folds": SELECTION_FOLDS,
            "metric": "MAE",
            "rule": "lowest training-only mean CV MAE per process target",
            "selected_algorithms": upstream_selection,
            "candidate_estimators": SELECTION_ESTIMATORS,
        },
        "training_stacking": {
            "oof_folds": OOF_FOLDS,
            "random_state": RANDOM_STATE + 1,
            "method": "explicit KFold; each fold pipeline preprocesses/fits only on other training rows",
            "fit_exclusion_verified": True,
        },
        "test_and_inference_stacking": "final Model 1 pipelines fitted on all 4000 training rows only",
        "downstream": {
            "target": SCRAP_TARGET,
            "algorithm": CANDIDATE_ALGORITHM,
            "parameters": DOWNSTREAM_PARAMS,
            "selection_rule": "fixed by the controlled evaluation protocol; no test-based selection",
        },
        "feature_order": expected_features,
        "customer_feature_order": MODEL1_FEATURES,
        "process_feature_order": PROCESS_TARGETS,
        "prohibited_feature_intersection": sorted(set(expected_features) & PROHIBITED_FEATURES),
        "unseen_category_probe_passed": True,
        "serialized_inference_probe_passed": True,
        "candidate_artifacts": {
            "directory": ARTIFACT_DIR.relative_to(PROJECT_ROOT).as_posix(),
            "model3": candidate_model_path.relative_to(PROJECT_ROOT).as_posix(),
            "model1": {
                target: (ARTIFACT_DIR / filename).relative_to(PROJECT_ROOT).as_posix()
                for target, filename in PROCESS_MODEL_FILENAMES.items()
            },
        },
        "existing_production_model_read_only_path": production_model_path.relative_to(PROJECT_ROOT).as_posix(),
        "package_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "limitations": [
            "Nested CV values are reused from the controlled evaluation, which uses this same fixed RF and fold-local upstream procedure.",
            "The current production artifact comparison uses existing Model 1/3 artifacts and identical held-out rows; it is a comparison, not model selection.",
            "All findings concern synthetic data and do not establish factory performance.",
        ],
    }
    (OUTPUT_DIR / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Candidate Model 3 saved to {candidate_model_path}")
    print(f"Dataset SHA-256: {dataset_hash}")
    print("Test comparison:")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print("Nested CV summary:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Candidate artifacts remain unpromoted under {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
