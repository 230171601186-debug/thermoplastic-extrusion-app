from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "training"
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    DATASET_CANDIDATES,
    EFFICIENCY_TARGET,
    MODEL1_FEATURES,
    PROCESS_TARGETS,
    RANDOM_STATE,
    SCRAP_TARGET,
    SPLIT_INDICES_PATH,
    TEST_SIZE,
)

OUTER_FOLDS = 5
UPSTREAM_OOF_FOLDS = 4
MODEL_PARAMS = {
    "n_estimators": 60,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
}
STACKED_FEATURES = MODEL1_FEATURES + PROCESS_TARGETS
DOWNSTREAM_FEATURES = set(STACKED_FEATURES)
FORBIDDEN_FEATURES = {
    EFFICIENCY_TARGET,
    SCRAP_TARGET,
    "OEE",
    "Availability",
    "Performance",
    "Quality",
    "Good_Length_m",
    "Scrap_Length_m",
    "Actual_Processing_Length_m",
    "Length_Shortfall_m",
    "Order_Status",
    "Fault_Code",
    "Power_Consumption_kWh",
    "Downtime_min",
    "Run_Time_min",
    "Planned_Processing_Length_m",
    "Planned_Production_Time_min",
    "Machine_ID",
    "Shift",
    "Ambient_Temperature_C",
    "Humidity_percent",
    "Record_ID",
}
PROCESS_BOUNDS = {
    "Extruder_Temperature_C": (180.0, 245.0),
    "Die_Temperature_C": (180.0, 235.0),
    "Cooling_Tank_Temperature_C": (12.0, 28.0),
    "Melt_Pressure_bar": (90.0, 200.0),
    "Air_Pressure_bar": (3.0, 12.0),
    "Screw_RPM": (48.0, 100.0),
    "Line_Speed_m_min": (8.0, 24.0),
    "Puller_Speed_m_min": (7.0, 25.0),
    "Extrusion_Speed_kg_hr": (25.0, 170.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def customer_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", CUSTOMER_NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def process_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", customer_preprocessor()),
            ("model", RandomForestRegressor(**MODEL_PARAMS)),
        ]
    )


def downstream_pipeline(features: list[str]) -> Pipeline:
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    numeric = [feature for feature in features if feature not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    params = dict(MODEL_PARAMS)
    params["n_estimators"] = 100
    return Pipeline(
        [("preprocess", preprocessor), ("model", RandomForestRegressor(**params))]
    )


def make_prediction_stack(customer: pd.DataFrame, upstream: dict[str, Pipeline]) -> pd.DataFrame:
    result = customer[MODEL1_FEATURES].copy()
    for target in PROCESS_TARGETS:
        prediction = np.asarray(upstream[target].predict(customer[MODEL1_FEATURES]), dtype=float)
        if prediction.shape != (len(customer),) or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid upstream output for {target}")
        result[target] = prediction
    if list(result.columns) != STACKED_FEATURES:
        raise RuntimeError("The upstream stack has a different feature contract")
    if set(result.columns) & FORBIDDEN_FEATURES:
        raise RuntimeError("A prohibited or post-production field entered the stack")
    return result


def crossfit_training_stack(
    training: pd.DataFrame,
    fold_label: str,
    audit: list[dict],
    evaluation_indices: pd.Index,
    evaluation_partition: str,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    outer_overlap = set(training.index) & set(evaluation_indices)
    if outer_overlap:
        raise RuntimeError(
            f"Outer training/evaluation overlap for {fold_label}: {len(outer_overlap)}"
        )
    customer = training[MODEL1_FEATURES].copy()
    stacked = customer.copy()
    final_models: dict[str, Pipeline] = {}
    splitter = KFold(
        n_splits=UPSTREAM_OOF_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + 1,
    )
    for target in PROCESS_TARGETS:
        values = np.full(len(training), np.nan, dtype=float)
        counts = np.zeros(len(training), dtype=np.int8)
        for inner_fold, (fit_pos, predict_pos) in enumerate(splitter.split(customer), start=1):
            fit_ids = set(training.index.to_numpy()[fit_pos].tolist())
            predict_ids = set(training.index.to_numpy()[predict_pos].tolist())
            overlap = fit_ids & predict_ids
            if overlap:
                raise RuntimeError(f"Cross-fit overlap for {target}: {len(overlap)}")
            model = process_pipeline()
            model.fit(customer.iloc[fit_pos], training[target].iloc[fit_pos])
            values[predict_pos] = model.predict(customer.iloc[predict_pos])
            counts[predict_pos] += 1
            audit.append(
                {
                    "evaluation_fold": fold_label,
                    "target": target,
                    "upstream_oof_fold": inner_fold,
                    "fit_rows": len(fit_pos),
                    "predicted_rows": len(predict_pos),
                    "fit_predict_index_overlap": len(overlap),
                    "oof_prediction_count_min": 1,
                    "oof_prediction_count_max": 1,
                    "outer_partition": evaluation_partition,
                    "outer_fit_rows": len(training),
                    "outer_evaluation_rows": len(evaluation_indices),
                    "outer_fit_evaluation_index_overlap": len(outer_overlap),
                }
            )
        if np.isnan(values).any() or not np.all(counts == 1):
            raise RuntimeError(f"Cross-fitted values incomplete for {target}")
        stacked[target] = values
        final_model = process_pipeline()
        final_model.fit(customer, training[target])
        final_models[target] = final_model
    if list(stacked.columns) != STACKED_FEATURES:
        raise RuntimeError("Cross-fitted training stack order mismatch")
    return stacked, final_models


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def record_prediction_rows(
    rows: list[dict],
    *,
    split: str,
    fold: str,
    target: str,
    configuration: str,
    frame: pd.DataFrame,
    predictions: np.ndarray,
    train_target: pd.Series,
) -> None:
    actual = frame[target].to_numpy(dtype=float)
    cuts = np.quantile(train_target.to_numpy(dtype=float), [0.25, 0.5, 0.75])
    for position, (observed, estimate) in enumerate(zip(actual, predictions)):
        rows.append(
            {
                "split": split,
                "fold": fold,
                "row_index": int(frame.index[position]),
                "target": target,
                "configuration": configuration,
                "actual": float(observed),
                "predicted": float(estimate),
                "residual_pred_minus_actual": float(estimate - observed),
                "absolute_error": float(abs(estimate - observed)),
                "training_q25": float(cuts[0]),
                "training_median": float(cuts[1]),
                "training_q75": float(cuts[2]),
            }
        )


def evaluate_downstream_fold(
    rows: list[dict],
    *,
    split: str,
    fold: str,
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    training_stack: pd.DataFrame,
    evaluation_stack: pd.DataFrame,
) -> None:
    for target in (EFFICIENCY_TARGET, SCRAP_TARGET):
        target_train = training[target].to_numpy(dtype=float)
        actual = evaluation[target].to_numpy(dtype=float)
        feature_groups = {
            "customer_only": (training[MODEL1_FEATURES], evaluation[MODEL1_FEATURES]),
            "full_stack": (
                training_stack[STACKED_FEATURES],
                evaluation_stack[STACKED_FEATURES],
            ),
        }
        predictions = {"training_mean": np.full(len(evaluation), target_train.mean())}
        for configuration, (train_features, eval_features) in feature_groups.items():
            model = downstream_pipeline(list(train_features.columns))
            model.fit(train_features, target_train)
            predictions[configuration] = np.asarray(model.predict(eval_features), dtype=float)
        for configuration, prediction in predictions.items():
            record_prediction_rows(
                rows,
                split=split,
                fold=fold,
                target=target,
                configuration=configuration,
                frame=evaluation,
                predictions=prediction,
                train_target=training[target],
            )


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict] = []
    for (split, fold, target, configuration), group in predictions.groupby(
        ["split", "fold", "target", "configuration"], sort=False
    ):
        scores = metrics(group["actual"].to_numpy(), group["predicted"].to_numpy())
        row = {
            "split": split,
            "fold": fold,
            "target": target,
            "configuration": configuration,
            "n_rows": len(group),
            **scores,
        }
        if target == SCRAP_TARGET:
            row["MAE_percentage_points"] = scores["MAE"] * 100
            row["RMSE_percentage_points"] = scores["RMSE"] * 100
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    cv = fold_metrics[fold_metrics["split"] == "outer_cv"]
    summary_rows = []
    for (target, configuration), group in cv.groupby(
        ["target", "configuration"], sort=False
    ):
        row = {
            "target": target,
            "configuration": configuration,
            "fold_count": int(group["fold"].nunique()),
            "evaluation_rows": int(group["n_rows"].sum()),
        }
        for metric in ("MAE", "RMSE", "R2"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        if target == SCRAP_TARGET:
            row["MAE_percentage_points_mean"] = row["MAE_mean"] * 100
            row["MAE_percentage_points_std"] = row["MAE_std"] * 100
            row["RMSE_percentage_points_mean"] = row["RMSE_mean"] * 100
            row["RMSE_percentage_points_std"] = row["RMSE_std"] * 100
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    residual_rows = []
    for (split, target, configuration), group in predictions.groupby(
        ["split", "target", "configuration"], sort=False
    ):
        residual = group["residual_pred_minus_actual"]
        residual_rows.append(
            {
                "split": split,
                "target": target,
                "configuration": configuration,
                "n_rows": len(group),
                "residual_mean": float(residual.mean()),
                "residual_median": float(residual.median()),
                "residual_std": float(residual.std(ddof=1)),
                "residual_q05": float(residual.quantile(0.05)),
                "residual_q95": float(residual.quantile(0.95)),
                "absolute_error_median": float(group["absolute_error"].median()),
                "absolute_error_q90": float(group["absolute_error"].quantile(0.90)),
            }
        )
    residuals = pd.DataFrame(residual_rows)
    return fold_metrics, summary, residuals


def evaluate_process_targets(
    data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    audit: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold_number, (train_pos, valid_pos) in enumerate(outer.split(train), start=1):
        fitting = train.iloc[train_pos]
        validation = train.iloc[valid_pos]
        label = f"outer_{fold_number}"
        for target in PROCESS_TARGETS:
            model = process_pipeline()
            model.fit(fitting[MODEL1_FEATURES], fitting[target])
            prediction = np.asarray(model.predict(validation[MODEL1_FEATURES]), dtype=float)
            truth = validation[target].to_numpy(dtype=float)
            low, high = PROCESS_BOUNDS[target]
            baseline_mean = np.full(len(validation), float(fitting[target].mean()))
            baseline_median = np.full(len(validation), float(fitting[target].median()))
            row = {
                "split": "outer_cv",
                "fold": label,
                "target": target,
                "n_rows": len(validation),
                "training_mean": float(fitting[target].mean()),
                "training_median": float(fitting[target].median()),
                **metrics(truth, prediction),
                "mean_baseline_MAE": float(mean_absolute_error(truth, baseline_mean)),
                "mean_baseline_RMSE": float(np.sqrt(mean_squared_error(truth, baseline_mean))),
                "mean_baseline_R2": float(r2_score(truth, baseline_mean)),
                "median_baseline_MAE": float(mean_absolute_error(truth, baseline_median)),
                "median_baseline_RMSE": float(np.sqrt(mean_squared_error(truth, baseline_median))),
                "median_baseline_R2": float(r2_score(truth, baseline_median)),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
                "outside_existing_synthetic_bound_count": int(((prediction < low) | (prediction > high)).sum()),
            }
            rows.append(row)
            for i, value in enumerate(prediction):
                prediction_rows.append(
                    {
                        "split": "outer_cv",
                        "fold": label,
                        "row_index": int(validation.index[i]),
                        "target": target,
                        "actual": float(truth[i]),
                        "predicted": float(value),
                        "residual_pred_minus_actual": float(value - truth[i]),
                    }
                )
    for target in PROCESS_TARGETS:
        model = process_pipeline()
        model.fit(train[MODEL1_FEATURES], train[target])
        prediction = np.asarray(model.predict(test[MODEL1_FEATURES]), dtype=float)
        truth = test[target].to_numpy(dtype=float)
        low, high = PROCESS_BOUNDS[target]
        baseline_mean = np.full(len(test), float(train[target].mean()))
        baseline_median = np.full(len(test), float(train[target].median()))
        rows.append(
            {
                "split": "saved_test",
                "fold": "saved_test",
                "target": target,
                "n_rows": len(test),
                "training_mean": float(train[target].mean()),
                "training_median": float(train[target].median()),
                **metrics(truth, prediction),
                "mean_baseline_MAE": float(mean_absolute_error(truth, baseline_mean)),
                "mean_baseline_RMSE": float(np.sqrt(mean_squared_error(truth, baseline_mean))),
                "mean_baseline_R2": float(r2_score(truth, baseline_mean)),
                "median_baseline_MAE": float(mean_absolute_error(truth, baseline_median)),
                "median_baseline_RMSE": float(np.sqrt(mean_squared_error(truth, baseline_median))),
                "median_baseline_R2": float(r2_score(truth, baseline_median)),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
                "outside_existing_synthetic_bound_count": int(((prediction < low) | (prediction > high)).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def run() -> None:
    dataset_path = next((path for path in DATASET_CANDIDATES if path.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError("Configured production dataset was not found.")
    dataset_hash = sha256(dataset_path)
    frame = pd.read_csv(dataset_path, keep_default_na=False)
    split = json.loads(SPLIT_INDICES_PATH.read_text(encoding="utf-8"))
    expected_train, expected_test = train_test_split(
        frame.index.to_numpy(), test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    if list(expected_train) != split["train_index"] or list(expected_test) != split["test_index"]:
        raise RuntimeError("Saved split does not match configured fixed split.")
    training = frame.loc[split["train_index"]].copy()
    test = frame.loc[split["test_index"]].copy()
    if set(training.index) & set(test.index) or len(training) != 4000 or len(test) != 1000:
        raise RuntimeError("Saved train/test split is invalid.")

    required = MODEL1_FEATURES + PROCESS_TARGETS + [EFFICIENCY_TARGET, SCRAP_TARGET]
    missing = [feature for feature in required if feature not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required fields: {missing}")
    if set(STACKED_FEATURES) & FORBIDDEN_FEATURES:
        raise RuntimeError("The declared inference stack contains prohibited outcomes.")

    upstream_audit: list[dict] = []
    model2_rows: list[dict] = []
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold_number, (train_pos, valid_pos) in enumerate(outer.split(training), start=1):
        fitting = training.iloc[train_pos].copy()
        validation = training.iloc[valid_pos].copy()
        label = f"outer_{fold_number}"
        training_stack, final_models = crossfit_training_stack(
            fitting,
            label,
            upstream_audit,
            validation.index,
            "outer_cv",
        )
        validation_stack = make_prediction_stack(validation[MODEL1_FEATURES], final_models)
        evaluate_downstream_fold(
            model2_rows,
            split="outer_cv",
            fold=label,
            training=fitting,
            evaluation=validation,
            training_stack=training_stack,
            evaluation_stack=validation_stack,
        )

    test_train_stack, test_upstream = crossfit_training_stack(
        training,
        "saved_test_training_partition",
        upstream_audit,
        test.index,
        "saved_test",
    )
    test_stack = make_prediction_stack(test[MODEL1_FEATURES], test_upstream)
    evaluate_downstream_fold(
        model2_rows,
        split="saved_test",
        fold="saved_test",
        training=training,
        evaluation=test,
        training_stack=test_train_stack,
        evaluation_stack=test_stack,
    )

    fold_metrics, summary, residuals = summarize_predictions(pd.DataFrame(model2_rows))
    fold_metrics.to_csv(EXPERIMENT_DIR / "downstream_fold_metrics.csv", index=False)
    summary.to_csv(EXPERIMENT_DIR / "downstream_cv_summary.csv", index=False)
    pd.DataFrame(model2_rows).to_csv(EXPERIMENT_DIR / "downstream_predictions_residuals.csv", index=False)
    residuals.to_csv(EXPERIMENT_DIR / "downstream_residual_diagnostics.csv", index=False)
    pd.DataFrame(upstream_audit).to_csv(EXPERIMENT_DIR / "upstream_fold_audit.csv", index=False)

    process_metrics, process_predictions = evaluate_process_targets(
        frame, training, test, upstream_audit
    )
    process_metrics.to_csv(EXPERIMENT_DIR / "model1_process_metrics.csv", index=False)
    process_predictions.to_csv(EXPERIMENT_DIR / "model1_outer_predictions.csv", index=False)

    candidate_manifest_paths = [
        ROOT / "training/evaluation_results/controlled_stacking/model2_candidate/candidate_manifest.json",
        ROOT / "training/evaluation_results/controlled_stacking/model3_candidate/candidate_manifest.json",
    ]
    manifests = {path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in candidate_manifest_paths}
    artifact_files = sorted(
        {
            ROOT / relative
            for manifest in manifests.values()
            for relative in [
                *manifest["candidate_artifacts"]["model1"].values(),
                manifest["candidate_artifacts"].get("model2", ""),
                manifest["candidate_artifacts"].get("model3", ""),
            ]
            if relative
        }
    )
    artifact_hashes = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
        for path in artifact_files
    ]

    manifest = {
        "experiment": "fixed_estimator_nested_cross_fitted_stacking_remediation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": dataset_hash,
        "dataset_shape": list(frame.shape),
        "saved_split_path": SPLIT_INDICES_PATH.relative_to(ROOT).as_posix(),
        "seed": RANDOM_STATE,
        "train_rows": len(training),
        "held_out_rows": len(test),
        "outer_folds": OUTER_FOLDS,
        "upstream_crossfit_folds": UPSTREAM_OOF_FOLDS,
        "fixed_process_model": {"class": "RandomForestRegressor", **MODEL_PARAMS},
        "fixed_downstream_model": {
            "class": "RandomForestRegressor",
            **{**MODEL_PARAMS, "n_estimators": 100},
        },
        "selection": "No model or hyperparameter selection in this run. Fixed estimator classes and parameters were declared before outer evaluation.",
        "feature_contract": {
            "customer_features": MODEL1_FEATURES,
            "process_targets": PROCESS_TARGETS,
            "downstream_features": STACKED_FEATURES,
            "prohibited_intersection": sorted(set(STACKED_FEATURES) & FORBIDDEN_FEATURES),
        },
        "config_baselines": ["training-target mean", "customer-input-only RF", "cross-fitted 17-feature RF"],
        "package_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "loaded_app_artifact_hashes": artifact_hashes,
        "notes": [
            "All data are synthetic; metric evidence is not factory validation.",
            "No actual process or post-run outcome value enters downstream predictor features.",
            "Every outer validation row is excluded from upstream process fitting and preprocessing fitting.",
            "Model 1 training-stack process features are row-excluded OOF predictions within each outer training fold.",
            "Saved held-out test partition is evaluated once after outer-fold reporting and is not used for selection.",
            "Existing candidate model artifacts are hashed and identified but not reused for this fixed-estimator evaluation.",
            "Numeric prediction bounds are copied only as existing synthetic assertions; they are not plant operating limits.",
        ],
    }
    (EXPERIMENT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run()
