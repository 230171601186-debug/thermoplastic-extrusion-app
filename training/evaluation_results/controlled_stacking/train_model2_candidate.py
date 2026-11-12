"""Train and validate a corrected Model 2 candidate without promotion.

The dataset, production metadata, and production model files are read-only.
All candidate artifacts and reports are written under model2_candidate/.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

TRAINING_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    DATASET_CANDIDATES,
    DISCLAIMER,
    EFFICIENCY_TARGET,
    LGBM_PARAMS,
    METADATA_PATH,
    MODEL1_FEATURES,
    PLANNING_CONTEXT_EXCLUDED,
    POST_PRODUCTION_LEAKAGE,
    PROCESS_MODEL_FILENAMES,
    PROCESS_TARGETS,
    RANDOM_STATE,
    RF_PARAMS,
    SPLIT_INDICES_PATH,
    TEST_SIZE,
    XGB_PARAMS,
)
from preprocessing import (  # noqa: E402
    attach_predicted_process,
    load_dataset,
    load_process_pipelines,
    make_regressor_pipeline,
    make_stacked_pipeline,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "model2_candidate"
ARTIFACT_DIR = OUTPUT_DIR / "candidate_artifacts"
CONTROLLED_METRICS_PATH = EXPERIMENT_DIR / "metrics_by_fold.csv"
CONTROLLED_PREDICTIONS_PATH = EXPERIMENT_DIR / "predictions_and_residuals.csv"
OUTER_FOLDS = 5
SELECTION_FOLDS = 3
OOF_FOLDS = 4
SELECTION_ESTIMATORS = {"Random Forest": 80, "XGBoost": 100, "LightGBM": 100}
DOWNSTREAM_PARAMS = deepcopy(RF_PARAMS)
DOWNSTREAM_PARAMS["n_estimators"] = 160
DOWNSTREAM_PARAMS["n_jobs"] = 1
KNOWN_DATASET_SHA256 = "cbd42684bc475ca9a7ad4f5b4512962c785034f2aae7c4944f0804e624c20396"

ALGORITHMS = {
    "Random Forest": (RandomForestRegressor, deepcopy(RF_PARAMS)),
    "XGBoost": (XGBRegressor, deepcopy(XGB_PARAMS)),
    "LightGBM": (LGBMRegressor, deepcopy(LGBM_PARAMS)),
}
for algorithm_name, (_, params) in ALGORITHMS.items():
    params["n_estimators"] = SELECTION_ESTIMATORS[algorithm_name]
    params["n_jobs"] = 1

STACKED_FEATURES = MODEL1_FEATURES + PROCESS_TARGETS
PROHIBITED_FEATURES = set(POST_PRODUCTION_LEAKAGE) | set(PLANNING_CONTEXT_EXCLUDED)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def verify_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_train, expected_test = train_test_split(
        df.index.to_numpy(), test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    if list(expected_train) != split["train_index"] or list(expected_test) != split["test_index"]:
        raise RuntimeError("Saved indices differ from the configured original split")
    train = df.loc[split["train_index"]].copy()
    test = df.loc[split["test_index"]].copy()
    if len(train) != 4000 or len(test) != 1000 or not set(train.index).isdisjoint(test.index):
        raise RuntimeError("Saved 4000/1000 partition is invalid")
    return train, test


def select_upstream_algorithms(train: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    """Select process estimators using CV confined to this training frame."""
    splitter = KFold(n_splits=SELECTION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    selected = {}
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
        selected[target] = min(scores, key=scores.get)
        for row in rows:
            if row["target"] == target:
                row["selected"] = row["algorithm"] == selected[target]
    return selected, pd.DataFrame(rows)


def make_selected_pipeline(target: str, selected: dict[str, str]):
    estimator_class, params = ALGORITHMS[selected[target]]
    return make_regressor_pipeline(estimator_class(**deepcopy(params)))


def fit_oof_and_final_upstream(
    train: pd.DataFrame,
    selections: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Create OOF process columns and separate full-training inference pipelines."""
    customer_inputs = train[MODEL1_FEATURES].copy()
    stacked = customer_inputs.copy()
    final_models = {}
    audit_rows = []
    splitter = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1)

    for target in PROCESS_TARGETS:
        predictions = np.full(len(train), np.nan, dtype=float)
        prediction_counts = np.zeros(len(train), dtype=np.int8)
        for fold_number, (fit_positions, held_positions) in enumerate(
            splitter.split(customer_inputs), start=1
        ):
            fit_indices = train.index.to_numpy()[fit_positions]
            held_indices = train.index.to_numpy()[held_positions]
            overlap = set(fit_indices) & set(held_indices)
            if overlap:
                raise RuntimeError(f"Upstream OOF fit/prediction overlap for {target}")

            # The whole pipeline is fitted within this fold, so its encoder sees
            # only the fit rows and cannot learn from the row it predicts.
            fold_pipeline = make_selected_pipeline(target, selections)
            fold_pipeline.fit(
                customer_inputs.iloc[fit_positions],
                train[target].iloc[fit_positions],
            )
            predictions[held_positions] = fold_pipeline.predict(
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

        if np.isnan(predictions).any() or not np.all(prediction_counts == 1):
            raise RuntimeError(f"OOF process features incomplete for {target}")
        stacked[target] = predictions

        final_pipeline = make_selected_pipeline(target, selections)
        final_pipeline.fit(customer_inputs, train[target])
        final_models[target] = final_pipeline

    if list(stacked.columns) != STACKED_FEATURES:
        raise RuntimeError("OOF training feature order differs from the production contract")
    if set(stacked.columns) & PROHIBITED_FEATURES:
        raise RuntimeError("A prohibited post-production feature entered Model 2 inputs")
    return stacked, final_models, pd.DataFrame(audit_rows)


def predict_stack(
    customer_features: pd.DataFrame,
    upstream_models: dict[str, object],
) -> pd.DataFrame:
    stacked = customer_features[MODEL1_FEATURES].copy()
    for target in PROCESS_TARGETS:
        stacked[target] = upstream_models[target].predict(customer_features[MODEL1_FEATURES])
    if list(stacked.columns) != STACKED_FEATURES:
        raise RuntimeError("Inference feature order differs from Model 2 training")
    if set(stacked.columns) & PROHIBITED_FEATURES:
        raise RuntimeError("A prohibited post-production feature entered inference")
    return stacked


def test_residual_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prediction_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    y = test[EFFICIENCY_TARGET].to_numpy(dtype=float)
    bins = np.unique(np.quantile(train[EFFICIENCY_TARGET], [0, 0.25, 0.5, 0.75, 1]))
    if len(bins) < 3:
        raise RuntimeError("Training target has too few unique values for range diagnostics")
    labels = [f"range_{index + 1}" for index in range(len(bins) - 1)]
    ranges = pd.cut(y, bins=[-np.inf, *bins[1:-1], np.inf], labels=labels, include_lowest=True)
    rows = []
    for name, predicted in prediction_map.items():
        residual = predicted - y
        for target_range in ["all", *labels]:
            mask = np.ones(len(y), dtype=bool) if target_range == "all" else (ranges == target_range)
            actual_group = y[mask]
            predicted_group = predicted[mask]
            residual_group = predicted_group - actual_group
            rows.append(
                {
                    "model": name,
                    "target_range": target_range,
                    "count": int(mask.sum()),
                    "MAE_percentage_points": float(mean_absolute_error(actual_group, predicted_group)),
                    "RMSE_percentage_points": float(np.sqrt(mean_squared_error(actual_group, predicted_group))),
                    "mean_residual_pred_minus_actual": float(residual_group.mean()),
                    "residual_q05": float(np.quantile(residual_group, 0.05)),
                    "residual_median": float(np.median(residual_group)),
                    "residual_q95": float(np.quantile(residual_group, 0.95)),
                    "prediction_below_zero_count": int(np.sum(predicted < 0)),
                    "prediction_above_100_count": int(np.sum(predicted > 100)),
                    "prediction_min": float(predicted.min()),
                    "prediction_max": float(predicted.max()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Configured dataset is missing")
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if dataset_hash != KNOWN_DATASET_SHA256:
        raise RuntimeError("Dataset checksum differs from the preserved experiment dataset")

    df = load_dataset()
    if df.shape != (5000, 40):
        raise RuntimeError(f"Expected unchanged 5000x40 dataset, got {df.shape}")
    if "None" not in set(df["Braid_Type"].astype(str)):
        raise RuntimeError("Braid_Type='None' was not preserved as a valid category")
    if df[MODEL1_FEATURES + PROCESS_TARGETS + [EFFICIENCY_TARGET]].isna().any().any():
        raise RuntimeError("Required input or target columns contain missing values")
    if list(MODEL1_FEATURES) != CATEGORICAL_FEATURES + CUSTOMER_NUMERIC_FEATURES:
        raise RuntimeError("Model 1 customer feature order differs from configuration")
    if set(STACKED_FEATURES) & PROHIBITED_FEATURES:
        raise RuntimeError("Configured Model 2 inputs intersect prohibited features")

    split = json.loads(SPLIT_INDICES_PATH.read_text(encoding="utf-8"))
    train, test = verify_split(df, split)
    selections, selection_rows = select_upstream_algorithms(train)
    selection_rows.to_csv(OUTPUT_DIR / "upstream_algorithm_selection.csv", index=False)
    train_stack, final_upstream, oof_audit = fit_oof_and_final_upstream(train, selections)
    # Held-out and later inference rows use upstream pipelines fitted on all
    # training rows, never on held-out target values.
    test_stack = predict_stack(test[MODEL1_FEATURES], final_upstream)
    if list(train_stack.columns) != list(test_stack.columns) or len(train_stack) != len(train) or len(test_stack) != len(test):
        raise RuntimeError("Training/test stack schema or row counts differ")
    oof_audit.to_csv(OUTPUT_DIR / "upstream_oof_fold_audit.csv", index=False)

    candidate = make_stacked_pipeline(
        RandomForestRegressor(**deepcopy(DOWNSTREAM_PARAMS)), PROCESS_TARGETS
    )
    candidate.fit(train_stack, train[EFFICIENCY_TARGET])
    candidate_prediction = np.asarray(candidate.predict(test_stack), dtype=float)

    customer_model = make_regressor_pipeline(
        RandomForestRegressor(**deepcopy(DOWNSTREAM_PARAMS))
    )
    customer_model.fit(train[MODEL1_FEATURES], train[EFFICIENCY_TARGET])
    customer_prediction = np.asarray(
        customer_model.predict(test[MODEL1_FEATURES]), dtype=float
    )

    process_only_features_train = train_stack[PROCESS_TARGETS].copy()
    process_only_features_test = test_stack[PROCESS_TARGETS].copy()
    # Process-only is diagnostic: construct an estimator with only process columns.
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    process_only_model = Pipeline(
        [
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
    process_only_model.fit(process_only_features_train, train[EFFICIENCY_TARGET])
    process_prediction = np.asarray(
        process_only_model.predict(process_only_features_test), dtype=float
    )

    production_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    production_model_path = PROJECT_ROOT / production_metadata["model2"]["file"]
    production_downstream = joblib.load(production_model_path)
    production_upstream = load_process_pipelines()
    production_test_stack = attach_predicted_process(test[MODEL1_FEATURES], production_upstream)
    if list(production_test_stack.columns) != STACKED_FEATURES:
        raise RuntimeError("Production artifact input order differs from the Model 2 contract")
    production_prediction = np.asarray(production_downstream.predict(production_test_stack), dtype=float)

    mean_prediction = np.full(len(test), float(train[EFFICIENCY_TARGET].mean()))
    prediction_map = {
        "training_mean": mean_prediction,
        "customer_only_fixed_RF": customer_prediction,
        "process_only_fixed_RF": process_prediction,
        "current_production_artifacts": production_prediction,
        "corrected_candidate_full_stack_RF": candidate_prediction,
    }
    y_test = test[EFFICIENCY_TARGET].to_numpy(dtype=float)
    test_rows = []
    residual_rows = []
    for name, prediction in prediction_map.items():
        scores = regression_metrics(y_test, prediction)
        test_rows.append(
            {
                "split": "saved_test_1000_rows",
                "model": name,
                "MAE_percentage_points": scores["MAE"],
                "RMSE_percentage_points": scores["RMSE"],
                "R2": scores["R2"],
                "prediction_below_zero_count": int(np.sum(prediction < 0)),
                "prediction_above_100_count": int(np.sum(prediction > 100)),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
            }
        )
        residual_rows.append(pd.DataFrame({
            "record_id": test["Record_ID"].astype(int).to_numpy(),
            "model": name,
            "actual": y_test,
            "predicted": prediction,
            "residual_pred_minus_actual": prediction - y_test,
            "absolute_error": np.abs(prediction - y_test),
        }))
    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / "test_comparison.csv", index=False)
    pd.concat(residual_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "test_predictions_and_residuals.csv", index=False
    )
    diagnostics = test_residual_diagnostics(train, test, prediction_map)
    diagnostics.to_csv(OUTPUT_DIR / "test_residual_diagnostics.csv", index=False)

    if not CONTROLLED_METRICS_PATH.exists() or not CONTROLLED_PREDICTIONS_PATH.exists():
        raise FileNotFoundError("Controlled nested Model 2 evaluation outputs are required")
    controlled_metrics = pd.read_csv(CONTROLLED_METRICS_PATH)
    model2_cv = controlled_metrics.loc[
        (controlled_metrics["split"] == "outer_cv")
        & (controlled_metrics["target"] == EFFICIENCY_TARGET)
    ].copy()
    if model2_cv.groupby("configuration")["fold"].nunique().min() != OUTER_FOLDS:
        raise RuntimeError("Controlled Model 2 metrics lack five isolated outer folds")
    model2_cv.to_csv(OUTPUT_DIR / "nested_cv_metrics_by_fold.csv", index=False)
    cv_rows = []
    for configuration, group in model2_cv.groupby("configuration", sort=False):
        row = {
            "configuration": configuration,
            "fold_count": int(group["fold"].nunique()),
        }
        for metric in ("MAE", "RMSE", "R2"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        cv_rows.append(row)
    pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR / "nested_cv_summary.csv", index=False)

    controlled_predictions = pd.read_csv(CONTROLLED_PREDICTIONS_PATH)
    efficiency_predictions = controlled_predictions.loc[
        (controlled_predictions["split"] == "outer_cv")
        & (controlled_predictions["target"] == EFFICIENCY_TARGET)
    ].copy()
    efficiency_predictions.to_csv(OUTPUT_DIR / "nested_cv_predictions_and_residuals.csv", index=False)
    nested_residuals = []
    for keys, group in efficiency_predictions.groupby(
        ["fold", "configuration", "target_range"], sort=False
    ):
        fold, configuration, target_range = keys
        residual = group["residual_pred_minus_actual"].to_numpy(dtype=float)
        nested_residuals.append(
            {
                "fold": fold,
                "configuration": configuration,
                "target_range": target_range,
                "count": len(group),
                "residual_mean": float(residual.mean()),
                "residual_std": float(np.std(residual, ddof=1)),
                "residual_q05": float(np.quantile(residual, 0.05)),
                "residual_median": float(np.median(residual)),
                "residual_q95": float(np.quantile(residual, 0.95)),
                "MAE_percentage_points": float(np.mean(np.abs(residual))),
                "RMSE_percentage_points": float(np.sqrt(np.mean(np.square(residual)))),
            }
        )
    pd.DataFrame(nested_residuals).to_csv(
        OUTPUT_DIR / "nested_cv_residual_diagnostics.csv", index=False
    )

    # Candidate-only artifact writes; no production path is used here.
    for target, filename in PROCESS_MODEL_FILENAMES.items():
        joblib.dump(final_upstream[target], ARTIFACT_DIR / filename)
    candidate_model_path = ARTIFACT_DIR / "efficiency_model_candidate.pkl"
    joblib.dump(candidate, candidate_model_path)

    unseen_probe = test[MODEL1_FEATURES].head(1).copy()
    unseen_probe.loc[unseen_probe.index[0], "Braid_Type"] = "__unseen_category_probe__"
    unseen_stack = predict_stack(unseen_probe, final_upstream)
    unseen_prediction = candidate.predict(unseen_stack)
    if unseen_prediction.shape != (1,) or not np.isfinite(unseen_prediction).all():
        raise RuntimeError("Unseen category did not produce a valid efficiency prediction")

    saved_upstream = {
        target: joblib.load(ARTIFACT_DIR / filename)
        for target, filename in PROCESS_MODEL_FILENAMES.items()
    }
    saved_candidate = joblib.load(candidate_model_path)
    serialized_stack = predict_stack(test[MODEL1_FEATURES].head(3), saved_upstream)
    serialized_prediction = saved_candidate.predict(serialized_stack)
    if serialized_prediction.shape != (3,) or not np.isfinite(serialized_prediction).all():
        raise RuntimeError("Serialized candidate failed end-to-end inference check")

    units = {
        "Material_Type": "categorical",
        "Material_Grade": "categorical",
        "Tube_Type": "categorical",
        "Braid_Type": "categorical; 'None' valid and unseen values ignored",
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
    }
    manifest = {
        "experiment": "model2_corrected_isolated_stacking_candidate",
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
        "outer_folds": OUTER_FOLDS,
        "upstream_selection": {
            "folds": SELECTION_FOLDS,
            "metric": "mean CV MAE per process target",
            "scope": "training rows only; repeated independently inside each outer fold for CV",
            "final_training_selection": "same CV selection using the complete 4000-row training partition",
            "candidate_estimators": SELECTION_ESTIMATORS,
            "candidate_parameters": {
                name: params for name, (_, params) in ALGORITHMS.items()
            },
            "selected_algorithms": selections,
        },
        "training_stacking": {
            "inner_oof_folds": OOF_FOLDS,
            "random_state": RANDOM_STATE + 1,
            "assertions": "each row predicted exactly once; no fit/prediction index overlap; preprocessing in fold pipeline",
        },
        "test_and_inference_stacking": "upstream pipelines fitted only on all 4000 training rows",
        "downstream": {
            "target": EFFICIENCY_TARGET,
            "target_unit": "percentage_points",
            "candidate_algorithm": "RandomForestRegressor",
            "candidate_parameters": DOWNSTREAM_PARAMS,
            "selection": "fixed to controlled evaluation protocol; test rows do not select algorithms or hyperparameters",
        },
        "features": {
            "customer_product_order": MODEL1_FEATURES,
            "process_parameter_order": PROCESS_TARGETS,
            "full_stack_order": STACKED_FEATURES,
            "units": units,
            "prohibited_features": sorted(PROHIBITED_FEATURES),
            "prohibited_intersection": sorted(set(STACKED_FEATURES) & PROHIBITED_FEATURES),
        },
        "categorical_preprocessing": "OneHotEncoder(handle_unknown='ignore'); Braid_Type='None' retained",
        "prediction_clipping": False,
        "unseen_category_probe_passed": True,
        "serialized_end_to_end_inference_passed": True,
        "candidate_artifacts": {
            "directory": ARTIFACT_DIR.relative_to(PROJECT_ROOT).as_posix(),
            "model2": candidate_model_path.relative_to(PROJECT_ROOT).as_posix(),
            "model1": {
                target: (ARTIFACT_DIR / filename).relative_to(PROJECT_ROOT).as_posix()
                for target, filename in PROCESS_MODEL_FILENAMES.items()
            },
        },
        "production_model_read_only_path": production_model_path.relative_to(PROJECT_ROOT).as_posix(),
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
            "Outer-CV comparisons reuse the previously completed fold-isolated controlled evaluation, with fixed RF downstream parameters and fold-specific upstream selection/OOF construction.",
            "Final candidate upstream selection uses the full training partition only; the held-out test is used once for comparison and never for selection.",
            "Results concern synthetic data and do not establish real factory performance.",
        ],
    }
    (OUTPUT_DIR / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Model 2 candidate saved to {candidate_model_path}")
    print(f"Dataset SHA-256: {dataset_hash}")
    print("Same-split test comparison:")
    print(pd.DataFrame(test_rows).to_string(index=False))
    print("Nested CV summary:")
    print(pd.DataFrame(cv_rows).to_string(index=False))
    print(f"Candidate artifacts remain unpromoted under {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
