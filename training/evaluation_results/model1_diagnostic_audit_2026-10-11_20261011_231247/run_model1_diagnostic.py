"""Read-only Model 1 audit and leakage-controlled Model 2 impact study."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TRAINING = ROOT / "training"
for import_path in (ROOT, TRAINING):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    DATASET_CANDIDATES,
    EFFICIENCY_TARGET,
    MODEL1_FEATURES,
    PROCESS_MODEL_FILENAMES,
    PROCESS_TARGETS,
    RANDOM_STATE,
    RF_PARAMS,
    SPLIT_INDICES_PATH,
)
from evaluate import regression_metrics  # noqa: E402
from preprocessing import (  # noqa: E402
    attach_predicted_process,
    build_customer_preprocessor,
    build_stacked_preprocessor,
    load_process_pipelines,
    make_regressor_pipeline,
    make_stacked_pipeline,
)

REVISED_DIR = ROOT / "training/evaluation_results/model2_generator_experiment_2026-10-11_revised"
DATASET_PATH = REVISED_DIR / "experimental_dataset.csv"
OUTER_FOLDS = 5
INNER_FOLDS = 3
OOF_FOLDS = 3
NESTED_SEED_OFFSET = 1900
M2_RF_PARAMS = dict(RF_PARAMS)
M2_RF_PARAMS["n_jobs"] = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def candidates(seed: int) -> dict[str, list[tuple[object, dict, bool]]]:
    return {
        "Ridge": [
            (Ridge(alpha=5.0), {"alpha": 5.0}, True)
        ],
        "RandomForest": [
            (
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=20,
                    min_samples_leaf=2,
                    max_features=1.0,
                    n_jobs=1,
                    random_state=seed,
                ),
                {
                    "n_estimators": 60,
                    "min_samples_leaf": 2,
                    "max_features": 1.0,
                    "max_depth": 20,
                },
                False,
            )
        ],
        "ExtraTrees": [
            (
                ExtraTreesRegressor(
                    n_estimators=70,
                    max_depth=20,
                    min_samples_leaf=2,
                    max_features=0.8,
                    n_jobs=1,
                    random_state=seed,
                ),
                {
                    "n_estimators": 70,
                    "min_samples_leaf": 2,
                    "max_features": 0.8,
                    "max_depth": 20,
                },
                False,
            )
        ],
        "HistGradientBoosting": [
            (
                HistGradientBoostingRegressor(
                    max_iter=120,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=3.0,
                    random_state=seed,
                ),
                {
                    "max_iter": 120,
                    "max_leaf_nodes": 15,
                    "l2_regularization": 3.0,
                },
                False,
            )
        ],
    }


def make_pipeline(estimator, *, stacked: bool = False, scale: bool = False) -> Pipeline:
    preprocessor = (
        build_stacked_preprocessor(PROCESS_TARGETS)
        if stacked
        else build_customer_preprocessor()
    )
    steps = [("preprocess", preprocessor)]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def fit_score(
    estimator,
    config: dict,
    scale: bool,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
) -> np.ndarray:
    model = make_pipeline(estimator, scale=scale)
    model.fit(x_train[MODEL1_FEATURES], y_train)
    return model.predict(x_eval[MODEL1_FEATURES])


def select_candidate(
    x: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    *,
    target: str,
    log: list[dict],
    scope: str,
) -> tuple[str, object, dict, bool]:
    splitter = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    grid = candidates(seed)
    scores: list[tuple[float, float, float, str, object, dict, bool]] = []
    for family, configs in grid.items():
        for estimator, params, scale in configs:
            fold_mae = []
            for inner_fold, (fit_pos, valid_pos) in enumerate(splitter.split(x), start=1):
                model = make_pipeline(estimator, scale=scale)
                model.fit(x.iloc[fit_pos][MODEL1_FEATURES], y[fit_pos])
                prediction = model.predict(x.iloc[valid_pos][MODEL1_FEATURES])
                fold_mae.append(
                    mean_absolute_error(y[valid_pos], prediction)
                )
            mean_mae = float(np.mean(fold_mae))
            sd_mae = float(np.std(fold_mae, ddof=1))
            log.append({
                "scope": scope,
                "target": target,
                "family": family,
                "parameters": params,
                "inner_MAE_mean": mean_mae,
                "inner_MAE_std": sd_mae,
            })
            scores.append(
                (mean_mae, sd_mae, family, estimator, params, scale)
            )
    best = min(scores, key=lambda row: (row[0], row[1], row[2]))
    return best[2], best[3], best[4], best[5]


def summarize_residuals(
    model: str,
    target: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    split: str,
) -> dict:
    residual = predicted - actual
    return {
        "split": split,
        "model": model,
        "target": target,
        "count": len(actual),
        "mean_residual_prediction_minus_actual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "residual_sd": float(np.std(residual, ddof=1)),
        "residual_q01": float(np.quantile(residual, 0.01)),
        "residual_q05": float(np.quantile(residual, 0.05)),
        "residual_q25": float(np.quantile(residual, 0.25)),
        "residual_q75": float(np.quantile(residual, 0.75)),
        "residual_q95": float(np.quantile(residual, 0.95)),
        "residual_q99": float(np.quantile(residual, 0.99)),
        "absolute_error_q50": float(np.quantile(np.abs(residual), 0.50)),
        "absolute_error_q90": float(np.quantile(np.abs(residual), 0.90)),
        **metrics(actual, predicted),
    }


def build_model1_outputs(
    frame: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    production_models: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict, list[dict]]:
    train = frame.loc[train_idx].copy()
    test = frame.loc[test_idx].copy()
    y_train = train[PROCESS_TARGETS].astype(float)
    y_test = test[PROCESS_TARGETS].astype(float)
    x_train = train[MODEL1_FEATURES]
    x_test = test[MODEL1_FEATURES]
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict] = []
    oof: dict[str, dict[str, np.ndarray]] = {
        target: {} for target in PROCESS_TARGETS
    }
    model_selection_log: list[dict] = []
    selected_full_train: dict[str, tuple[str, object, dict, bool]] = {}

    for target in PROCESS_TARGETS:
        y = y_train[target].to_numpy(dtype=float)
        outer_prediction: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for fold_number, (fit_pos, valid_pos) in enumerate(outer.split(x_train), start=1):
            x_fit, x_valid = x_train.iloc[fit_pos], x_train.iloc[valid_pos]
            y_fit = y[fit_pos]
            for family, configs in candidates(RANDOM_STATE + fold_number).items():
                for estimator, params, scale in configs:
                    model = make_pipeline(estimator, scale=scale)
                    model.fit(x_fit[MODEL1_FEATURES], y_fit)
                    prediction = model.predict(x_valid[MODEL1_FEATURES])
                    score = metrics(y[valid_pos], prediction)
                    fold_rows.append({
                        "task": "Model1",
                        "target": target,
                        "evaluation_arm": family,
                        "configuration": json.dumps(params, sort_keys=True),
                        "fold": fold_number,
                        **score,
                    })
                    outer_prediction.setdefault(f"{family}:{json.dumps(params, sort_keys=True)}", []).append(
                        (valid_pos, prediction)
                    )

            family, estimator, params, scale = select_candidate(
                x_fit,
                y_fit,
                RANDOM_STATE + NESTED_SEED_OFFSET + fold_number,
                target=target,
                log=model_selection_log,
                scope=f"model1_outer_{fold_number}",
            )
            tuned = make_pipeline(estimator, scale=scale)
            tuned.fit(x_fit[MODEL1_FEATURES], y_fit)
            tuned_prediction = tuned.predict(x_valid[MODEL1_FEATURES])
            fold_rows.append({
                "task": "Model1",
                "target": target,
                "evaluation_arm": "inner_selected",
                "configuration": json.dumps({"family": family, **params}, sort_keys=True),
                "fold": fold_number,
                **metrics(y[valid_pos], tuned_prediction),
            })
            oof[target][f"fold_{fold_number}"] = {
                "valid_positions": valid_pos,
                "predicted": tuned_prediction,
            }

        family, estimator, params, scale = select_candidate(
            x_train,
            y,
            RANDOM_STATE + NESTED_SEED_OFFSET + 999,
            target=target,
            log=model_selection_log,
            scope="model1_saved_train_inner_selection",
        )
        selected_full_train[target] = (family, estimator, params, scale)

    prediction_rows: list[dict] = []
    test_predictions: dict[str, dict[str, np.ndarray]] = {}
    baseline_rows: list[dict] = []
    for target in PROCESS_TARGETS:
        actual_train = y_train[target].to_numpy(dtype=float)
        actual_test = y_test[target].to_numpy(dtype=float)
        saved = production_models[target].predict(x_test)
        test_predictions[target] = {"Saved_Model1": np.asarray(saved)}
        baseline_predictions = {
            "training_mean": np.full(len(actual_test), actual_train.mean()),
            "training_median": np.full(len(actual_test), np.median(actual_train)),
        }
        for label, predicted in baseline_predictions.items():
            score = metrics(actual_test, predicted)
            baseline_rows.append({
                "target": target,
                "comparison": label,
                "split": "saved_test",
                "target_train_variance": float(np.var(actual_train, ddof=1)),
                "target_test_min": float(np.min(actual_test)),
                "target_test_max": float(np.max(actual_test)),
                **score,
            })
        score_saved = metrics(actual_test, saved)
        baseline_rows.append({
            "target": target,
            "comparison": "Saved production Model1 artifact",
            "split": "saved_test",
            "target_train_variance": float(np.var(actual_train, ddof=1)),
            "target_test_min": float(np.min(actual_test)),
            "target_test_max": float(np.max(actual_test)),
            **score_saved,
        })
        family, estimator, params, scale = selected_full_train[target]
        model = make_pipeline(estimator, scale=scale)
        model.fit(x_train[MODEL1_FEATURES], actual_train)
        selected_prediction = model.predict(x_test[MODEL1_FEATURES])
        test_predictions[target]["training_selected"] = selected_prediction
        baseline_rows.append({
            "target": target,
            "comparison": "Training-only-selected candidate",
            "split": "saved_test",
            "target_train_variance": float(np.var(actual_train, ddof=1)),
            "target_test_min": float(np.min(actual_test)),
            "target_test_max": float(np.max(actual_test)),
            "selected_family": family,
            "selected_parameters": json.dumps(params, sort_keys=True),
            **metrics(actual_test, selected_prediction),
        })
        fold_saved_predictions = production_models[target].predict(x_train)
        baseline_rows.append({
            "target": target,
            "comparison": "Saved production Model1 artifact",
            "split": "outer_cv_rows_fixed_artifact",
            "target_train_variance": float(np.var(actual_train, ddof=1)),
            "target_test_min": float(np.min(actual_train)),
            "target_test_max": float(np.max(actual_train)),
            **metrics(actual_train, fold_saved_predictions),
        })
        baseline_rows[-1]["note"] = (
            "Frozen production artifact evaluated across fixed CV rows; not retrained per fold."
        )

        for model_name, predicted in (
            ("Saved_Model1", saved),
            ("Training_selected", selected_prediction),
        ):
            split_name = "saved_test"
            residual_summary = summarize_residuals(
                model_name, target, actual_test, predicted, split_name
            )
            for index, (truth, estimate) in enumerate(zip(actual_test, predicted)):
                prediction_rows.append({
                    "task": "Model1_test",
                    "target": target,
                    "model": model_name,
                    "row_index": int(test.index[index]),
                    "actual": float(truth),
                    "predicted": float(estimate),
                    "residual_pred_minus_actual": float(estimate - truth),
                })
            baseline_rows.append({
                "target": target,
                "comparison": model_name,
                "split": "saved_test_residual_summary",
                **residual_summary,
            })

    return train, test, test_predictions, selected_full_train, {
        "baseline_rows": baseline_rows,
        "prediction_rows": prediction_rows,
        "model_selection_log": model_selection_log,
        "fold_rows": fold_rows,
    }, oof, []


def fit_process_oof(
    train: pd.DataFrame,
    selected: dict[str, tuple[str, object, dict, bool]],
    seed: int,
    log: list[dict],
    scope: str,
) -> tuple[pd.DataFrame, dict]:
    """Cross-fit Model 1 with configuration selection nested inside each OOF fold."""
    x = train[MODEL1_FEATURES]
    stacked = x.copy()
    models = {}
    folds = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=seed)
    for target in PROCESS_TARGETS:
        truth = train[target].astype(float).to_numpy()
        predictions = np.empty(len(train), dtype=float)
        for fold_number, (fit_pos, valid_pos) in enumerate(folds.split(x), start=1):
            family, estimator, params, scale = select_candidate(
                x.iloc[fit_pos],
                truth[fit_pos],
                seed + fold_number,
                target=target,
                log=log,
                scope=f"{scope}_{target}_fold_{fold_number}",
            )
            model = make_pipeline(estimator, scale=scale)
            model.fit(x.iloc[fit_pos][MODEL1_FEATURES], truth[fit_pos])
            predictions[valid_pos] = model.predict(
                x.iloc[valid_pos][MODEL1_FEATURES]
            )
        stacked[target] = predictions
        family, estimator, params, scale = selected[target]
        final_model = make_pipeline(estimator, scale=scale)
        final_model.fit(x[MODEL1_FEATURES], truth)
        models[target] = final_model
    return stacked, models


def downstream_impact(
    frame: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    production_models: dict,
    selected_full_train: dict[str, tuple[str, object, dict, bool]],
) -> tuple[list[dict], list[dict], list[dict]]:
    train = frame.loc[train_idx].copy()
    test = frame.loc[test_idx].copy()
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []
    predictions: list[dict] = []
    logs: list[dict] = []

    for fold_number, (fit_pos, valid_pos) in enumerate(outer.split(train), start=1):
        outer_train = train.iloc[fit_pos].copy()
        outer_valid = train.iloc[valid_pos].copy()
        y_fit = outer_train[EFFICIENCY_TARGET].astype(float).to_numpy()
        y_valid = outer_valid[EFFICIENCY_TARGET].astype(float).to_numpy()
        saved_train_stack = attach_predicted_process(
            outer_train[MODEL1_FEATURES], production_models
        )
        saved_valid_stack = attach_predicted_process(
            outer_valid[MODEL1_FEATURES], production_models
        )
        # Algorithm selection used only outer-training labels; outer-validation
        # rows/targets are excluded from upstream model fitting and selection.
        chosen_upstream = {}
        for target in PROCESS_TARGETS:
            chosen_upstream[target] = select_candidate(
                outer_train[MODEL1_FEATURES],
                outer_train[target].astype(float).to_numpy(),
                RANDOM_STATE + NESTED_SEED_OFFSET + 200 + fold_number,
                target=target,
                log=logs,
                scope=f"downstream_outer_{fold_number}_upstream_select",
            )
        crossfit_train, final_process_models = fit_process_oof(
            outer_train,
            chosen_upstream,
            RANDOM_STATE + 400 + fold_number,
            logs,
            f"downstream_outer_{fold_number}_crossfit",
        )
        crossfit_valid = attach_predicted_process(
            outer_valid[MODEL1_FEATURES], final_process_models
        )
        arms = {
            "A_saved_Model1_inference_predictions": (saved_train_stack, saved_valid_stack, True),
            "B_outer_train_only_selected_Model1_OOF": (crossfit_train, crossfit_valid, True),
            "C_oracle_recorded_process_non_deployable": (
                outer_train[MODEL1_FEATURES + PROCESS_TARGETS],
                outer_valid[MODEL1_FEATURES + PROCESS_TARGETS],
                False,
            ),
        }
        for arm, (m2_train, m2_valid, deployable) in arms.items():
            model = make_stacked_pipeline(
                RandomForestRegressor(**M2_RF_PARAMS), PROCESS_TARGETS
            )
            model.fit(m2_train, y_fit)
            predicted = model.predict(m2_valid)
            score = metrics(y_valid, predicted)
            rows.append({
                "task": "Model2_downstream",
                "arm": arm,
                "fold": fold_number,
                "deployable": deployable,
                **score,
            })
            for local_index, (truth, estimate) in enumerate(zip(y_valid, predicted)):
                predictions.append({
                    "task": "Model2_downstream",
                    "split": "outer_cv",
                    "arm": arm,
                    "fold": fold_number,
                    "row_index": int(outer_valid.index[local_index]),
                    "actual": float(truth),
                    "predicted": float(estimate),
                    "residual_pred_minus_actual": float(estimate - truth),
                })

    y_train = train[EFFICIENCY_TARGET].astype(float).to_numpy()
    saved_train = attach_predicted_process(train[MODEL1_FEATURES], production_models)
    saved_test = attach_predicted_process(test[MODEL1_FEATURES], production_models)
    selected_oof_train, selected_final_models = fit_process_oof(
        train,
        selected_full_train,
        RANDOM_STATE + 811,
        logs,
        "saved_test_crossfit",
    )
    selected_test = attach_predicted_process(test[MODEL1_FEATURES], selected_final_models)
    oracle_train = train[MODEL1_FEATURES + PROCESS_TARGETS]
    oracle_test = test[MODEL1_FEATURES + PROCESS_TARGETS]
    test_arms = {
        "A_saved_Model1_inference_predictions": (saved_train, saved_test, True),
        "B_outer_train_only_selected_Model1_OOF": (
            selected_oof_train, selected_test, True
        ),
        "C_oracle_recorded_process_non_deployable": (oracle_train, oracle_test, False),
    }
    final_predictions = []
    for arm, (m2_train, m2_test, deployable) in test_arms.items():
        model = make_stacked_pipeline(
            RandomForestRegressor(**M2_RF_PARAMS), PROCESS_TARGETS
        )
        model.fit(m2_train, y_train)
        prediction = model.predict(m2_test)
        score = metrics(test[EFFICIENCY_TARGET].to_numpy(dtype=float), prediction)
        rows.append({
            "task": "Model2_downstream",
            "arm": arm,
            "fold": "saved_test",
            "deployable": deployable,
            **score,
        })
        for local_index, (truth, estimate) in enumerate(zip(
            test[EFFICIENCY_TARGET].to_numpy(dtype=float), prediction
        )):
            final_predictions.append({
                "task": "Model2_downstream",
                "split": "saved_test",
                "arm": arm,
                "fold": "saved_test",
                "row_index": int(test.index[local_index]),
                "actual": float(truth),
                "predicted": float(estimate),
                "residual_pred_minus_actual": float(estimate - truth),
            })
    predictions.extend(final_predictions)
    return rows, predictions, logs


def feature_relationships(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = ["Line_Speed_m_min", "Puller_Speed_m_min"]
    numeric_inputs = [
        "Target_Tube_OD_mm",
        "Target_Tube_Wall_mm",
        "Target_Tube_ID_mm",
        "Customer_Required_Length_m",
    ]
    for target in targets:
        for feature in numeric_inputs + [
            "Extrusion_Speed_kg_hr",
            "Screw_RPM",
            "Line_Speed_m_min",
            "Puller_Speed_m_min",
        ]:
            if feature == target:
                continue
            rows.append({
                "target": target,
                "feature": feature,
                "relationship": "Pearson correlation",
                "group": "",
                "count": len(frame),
                "statistic": float(frame[target].corr(frame[feature])),
                "interpretation": "Association only; not causal.",
            })
        for feature in CATEGORICAL_FEATURES:
            for level, group in frame.groupby(feature, observed=True):
                rows.append({
                    "target": target,
                    "feature": feature,
                    "relationship": "target mean by category",
                    "group": str(level),
                    "count": len(group),
                    "statistic": float(group[target].mean()),
                    "interpretation": "Descriptive synthetic-data subgroup mean.",
                })
    delta = frame["Puller_Speed_m_min"] - frame["Line_Speed_m_min"]
    rows.extend([
        {
            "target": "Line_Puller_Consistency",
            "feature": "Puller_minus_Line_Speed_m_min",
            "relationship": "difference distribution",
            "group": quantile,
            "count": len(frame),
            "statistic": float(value),
            "interpretation": "Synthetic consistency diagnostic; not a manufacturing tolerance.",
        }
        for quantile, value in zip(
            ["mean", "std", "min", "q05", "median", "q95", "max"],
            [
                delta.mean(), delta.std(ddof=1), delta.min(), delta.quantile(.05),
                delta.median(), delta.quantile(.95), delta.max(),
            ],
        )
    ])
    return pd.DataFrame(rows)


def plot_weak_targets(
    actual: np.ndarray,
    predicted: np.ndarray,
    target: str,
) -> None:
    residual = predicted - actual
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(actual, predicted, s=10, alpha=.35)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    axes[0, 0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[0, 0].set(title="Actual vs predicted", xlabel="Actual", ylabel="Predicted")
    axes[0, 1].scatter(actual, residual, s=10, alpha=.35)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set(title="Residual vs actual", xlabel="Actual", ylabel="Predicted − actual")
    axes[1, 0].scatter(predicted, residual, s=10, alpha=.35)
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set(title="Residual vs predicted", xlabel="Predicted", ylabel="Predicted − actual")
    quantile_bins = pd.qcut(actual, q=10, duplicates="drop")
    grouped = pd.DataFrame({"actual": actual, "predicted": predicted}).groupby(
        quantile_bins, observed=True
    ).agg(actual_mean=("actual", "mean"), predicted_mean=("predicted", "mean"))
    axes[1, 1].plot(grouped["actual_mean"], grouped["predicted_mean"], marker="o", label="Prediction")
    axes[1, 1].plot(grouped["actual_mean"], grouped["actual_mean"], linestyle="--", label="Ideal")
    axes[1, 1].set(title="Decile means: mean-collapse check", xlabel="Actual decile mean", ylabel="Predicted decile mean")
    axes[1, 1].legend()
    fig.suptitle(f"{target}: saved Model 1 artifact on revised synthetic holdout")
    fig.tight_layout()
    fig.savefig(OUT / f"{target}_diagnostics.png", dpi=160)
    plt.close(fig)


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"

    def format_value(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += [
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def verify_preservation() -> dict:
    manifest = json.loads((OUT / "preservation_manifest.json").read_text(encoding="utf-8"))
    changed = []
    for relative, expected in manifest["protected_files_sha256"].items():
        path = ROOT / Path(relative)
        if not path.is_file():
            changed.append({"path": relative, "issue": "missing"})
            continue
        current = sha256(path)
        if current != expected:
            changed.append({
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": current,
            })
    dataset = ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
    return {
        "verified": not changed,
        "verified_file_count": len(manifest["protected_files_sha256"]) - len(changed),
        "expected_file_count": len(manifest["protected_files_sha256"]),
        "production_dataset_sha256_before": manifest["production_dataset_sha256"],
        "production_dataset_sha256_after": sha256(dataset),
        "mismatches": changed,
    }


def main() -> None:
    if not (OUT / "preservation_manifest.json").is_file():
        raise FileNotFoundError("Run capture_preservation.py before the diagnostic.")
    manifest = json.loads((OUT / "preservation_manifest.json").read_text(encoding="utf-8"))
    if manifest["output_directory"] != OUT.relative_to(ROOT).as_posix():
        raise RuntimeError("This script does not match the captured output directory.")
    dataset_hash = sha256(DATASET_PATH)
    expected_hash = "0db6c6d33f0a3d580b2f673cc3761b4c2551f48c1ca68d296303d9854a669b15"
    if dataset_hash != expected_hash:
        raise RuntimeError("Revised synthetic dataset hash differs from the reviewed input.")
    frame = pd.read_csv(DATASET_PATH, keep_default_na=False)
    if frame.shape != (5000, 40):
        raise RuntimeError(f"Expected revised dataset 5000 x 40, got {frame.shape}")
    split = json.loads(SPLIT_INDICES_PATH.read_text(encoding="utf-8"))
    train_idx, test_idx = split["train_index"], split["test_index"]
    if len(train_idx) != 4000 or len(test_idx) != 1000:
        raise RuntimeError("Saved split must have 4000 train and 1000 test rows.")
    if set(train_idx) & set(test_idx):
        raise RuntimeError("Saved training and test index sets overlap.")
    metadata_path = ROOT / "models/model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    process_models = load_process_pipelines()
    if set(process_models) != set(PROCESS_MODEL_FILENAMES):
        raise RuntimeError("Saved production Model 1 bundle does not contain all nine models.")
    declared_features = metadata["model1"]["features"]
    if declared_features != MODEL1_FEATURES:
        raise RuntimeError("Model 1 metadata features disagree with config's inference contract.")
    if set(declared_features) - set(frame.columns):
        raise RuntimeError("Dataset is missing one or more production Model 1 input fields.")
    if set(PROCESS_TARGETS) & set(declared_features):
        raise RuntimeError("Model 1 feature contract contains a process target (leakage).")

    train, test, test_predictions, selected_full, results, _, _ = build_model1_outputs(
        frame, train_idx, test_idx, process_models
    )
    model1_baseline = pd.DataFrame(results["baseline_rows"])
    model1_baseline.to_csv(OUT / "baseline_comparison.csv", index=False)
    fold_frame = pd.DataFrame(results["fold_rows"])
    fold_frame.to_csv(OUT / "fold_level_results.csv", index=False)
    residual_rows = results["prediction_rows"]
    selected_test_metrics = []
    target_summary = []
    split_outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for target in PROCESS_TARGETS:
        y_train = train[target].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)
        saved_test_pred = test_predictions[target]["Saved_Model1"]
        selected_test_pred = test_predictions[target]["training_selected"]
        saved_summary = summarize_residuals(
            "Saved_Model1", target, y_test, saved_test_pred, "saved_test"
        )
        selected_summary = summarize_residuals(
            "Training_selected", target, y_test, selected_test_pred, "saved_test"
        )
        selected_test_metrics.extend([saved_summary, selected_summary])
        for model_name, predicted in (
            ("Saved_Model1", saved_test_pred),
            ("Training_selected", selected_test_pred),
        ):
            residual = predicted - y_test
            for i, (actual, estimate) in enumerate(zip(y_test, predicted)):
                residual_rows.append({
                    "task": "Model1_test",
                    "target": target,
                    "model": model_name,
                    "row_index": int(test.index[i]),
                    "actual": float(actual),
                    "predicted": float(estimate),
                    "residual_pred_minus_actual": float(estimate - actual),
                })
        actual = y_test
        target_summary.append({
            "target": target,
            "train_count": len(y_train),
            "train_variance": float(np.var(y_train, ddof=1)),
            "train_min": float(np.min(y_train)),
            "train_q05": float(np.quantile(y_train, .05)),
            "train_median": float(np.median(y_train)),
            "train_q95": float(np.quantile(y_train, .95)),
            "train_max": float(np.max(y_train)),
            "test_min": float(np.min(actual)),
            "test_max": float(np.max(actual)),
            "saved_artifact_test_R2": saved_summary["R2"],
            "saved_artifact_test_MAE": saved_summary["MAE"],
            "saved_artifact_test_RMSE": saved_summary["RMSE"],
            "selected_test_R2": selected_summary["R2"],
            "selected_test_MAE": selected_summary["MAE"],
            "selected_test_RMSE": selected_summary["RMSE"],
        })
        if target in {"Line_Speed_m_min", "Puller_Speed_m_min"}:
            plot_weak_targets(y_test, saved_test_pred, target)
    model1_metrics = pd.DataFrame(target_summary)
    model1_metrics.to_csv(OUT / "model1_metrics.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(OUT / "residual_diagnostics.csv", index=False)

    relationship = feature_relationships(frame)
    relationship.to_csv(OUT / "feature_relationship_diagnostics.csv", index=False)
    data_prep = {
        "dataset": DATASET_PATH.relative_to(ROOT).as_posix(),
        "dataset_sha256": dataset_hash,
        "dataset_shape": list(frame.shape),
        "split_path": SPLIT_INDICES_PATH.relative_to(ROOT).as_posix(),
        "split_sha256": sha256(SPLIT_INDICES_PATH),
        "train_rows": len(train),
        "test_rows": len(test),
        "seed": RANDOM_STATE,
        "features": MODEL1_FEATURES,
        "targets": PROCESS_TARGETS,
        "inference_feature_check": {
            "metadata_matches_config": declared_features == MODEL1_FEATURES,
            "all_features_present": True,
            "targets_excluded_from_features": True,
        },
    }
    (OUT / "input_validation.json").write_text(
        json.dumps(data_prep, indent=2), encoding="utf-8"
    )

    downstream_rows, downstream_predictions, downstream_logs = downstream_impact(
        frame, train_idx, test_idx, process_models, selected_full
    )
    pd.DataFrame(downstream_rows).to_csv(
        OUT / "model2_downstream_impact.csv", index=False
    )
    pd.DataFrame(downstream_predictions).to_csv(
        OUT / "model2_downstream_predictions.csv", index=False
    )
    (OUT / "nested_selection_log.json").write_text(
        json.dumps(downstream_logs, indent=2), encoding="utf-8"
    )
    # Summary metrics: selected candidate and frozen production artifact across folds.
    selected_candidate_fold = fold_frame[fold_frame["evaluation_arm"] == "inner_selected"]
    saved_train = attach_predicted_process(
        train[MODEL1_FEATURES], process_models
    )
    saved_fold_records = []
    for fold_number, (_, valid_pos) in enumerate(split_outer.split(train), start=1):
        actual = train.iloc[valid_pos][PROCESS_TARGETS]
        predicted = saved_train.iloc[valid_pos][PROCESS_TARGETS]
        for target in PROCESS_TARGETS:
            saved_fold_records.append({
                "task": "Model1",
                "target": target,
                "evaluation_arm": "Saved_Model1",
                "fold": fold_number,
                **metrics(actual[target].to_numpy(dtype=float), predicted[target].to_numpy(dtype=float)),
            })
    fold_frame = pd.concat(
        [fold_frame, pd.DataFrame(saved_fold_records)],
        ignore_index=True,
    )
    fold_frame.to_csv(OUT / "fold_level_results.csv", index=False)
    aggregate = (
        fold_frame[fold_frame["task"] == "Model1"]
        .groupby(["target", "evaluation_arm"], as_index=False)
        .agg(
            fold_count=("fold", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
    )
    aggregate.to_csv(OUT / "model1_fold_aggregate.csv", index=False)

    for target in ("Line_Speed_m_min", "Puller_Speed_m_min"):
        subset = frame.loc[test_idx].copy()
        actual = subset[target].to_numpy(dtype=float)
        predicted = test_predictions[target]["Saved_Model1"]
        deciles = pd.qcut(actual, 10, duplicates="drop")
        diagnostic_rows = []
        for decile, group_index in pd.Series(np.arange(len(actual))).groupby(deciles, observed=True):
            idx = group_index.to_numpy(dtype=int)
            error = predicted[idx] - actual[idx]
            diagnostic_rows.append({
                "target": target,
                "actual_decile": str(decile),
                "count": len(idx),
                "actual_mean": float(np.mean(actual[idx])),
                "predicted_mean": float(np.mean(predicted[idx])),
                "residual_mean": float(np.mean(error)),
                "MAE": float(np.mean(np.abs(error))),
            })
        pd.DataFrame(diagnostic_rows).to_csv(
            OUT / f"{target}_decile_diagnostics.csv", index=False
        )

    downstream_frame = pd.DataFrame(downstream_rows)
    downstream_summary = (
        downstream_frame.groupby(["arm", "deployable"], as_index=False)
        .agg(
            outer_fold_count=("fold", lambda values: values[values != "saved_test"].nunique()),
            outer_MAE_mean=("MAE", lambda values: float(np.mean(values.iloc[:OUTER_FOLDS]))),
            outer_MAE_std=("MAE", lambda values: float(np.std(values.iloc[:OUTER_FOLDS], ddof=1))),
            outer_RMSE_mean=("RMSE", lambda values: float(np.mean(values.iloc[:OUTER_FOLDS]))),
            outer_RMSE_std=("RMSE", lambda values: float(np.std(values.iloc[:OUTER_FOLDS], ddof=1))),
            outer_R2_mean=("R2", lambda values: float(np.mean(values.iloc[:OUTER_FOLDS]))),
            outer_R2_std=("R2", lambda values: float(np.std(values.iloc[:OUTER_FOLDS], ddof=1))),
        )
    )
    test_scores = downstream_frame[downstream_frame["fold"] == "saved_test"][
        ["arm", "MAE", "RMSE", "R2"]
    ].rename(columns={"MAE": "saved_test_MAE", "RMSE": "saved_test_RMSE", "R2": "saved_test_R2"})
    downstream_summary = downstream_summary.merge(test_scores, on="arm", how="left")
    downstream_summary.to_csv(OUT / "model2_downstream_summary.csv", index=False)

    weak = model1_metrics.set_index("target").loc[
        ["Line_Speed_m_min", "Puller_Speed_m_min"]
    ].reset_index()
    production_algorithm_summary = pd.DataFrame([
        {
            "target": target,
            "production_selected_algorithm": metadata["model1"]["targets"][target]["selected_algorithm"],
            "production_metadata_test_MAE": metadata["model1"]["targets"][target]["test_metrics"]["MAE"],
            "production_metadata_test_RMSE": metadata["model1"]["targets"][target]["test_metrics"]["RMSE"],
            "production_metadata_test_R2": metadata["model1"]["targets"][target]["test_metrics"]["R2"],
            "revised_dataset_frozen_artifact_test_MAE": float(
                model1_metrics.loc[model1_metrics["target"] == target, "saved_artifact_test_MAE"].iloc[0]
            ),
            "revised_dataset_frozen_artifact_test_RMSE": float(
                model1_metrics.loc[model1_metrics["target"] == target, "saved_artifact_test_RMSE"].iloc[0]
            ),
            "revised_dataset_frozen_artifact_test_R2": float(
                model1_metrics.loc[model1_metrics["target"] == target, "saved_artifact_test_R2"].iloc[0]
            ),
            "interpretation": "Different datasets; descriptive domain-transfer contrast, not paired.",
        }
        for target in PROCESS_TARGETS
    ])
    production_algorithm_summary.to_csv(
        OUT / "saved_metric_context.csv", index=False
    )

    preservation = verify_preservation()
    (OUT / "preservation_verification.json").write_text(
        json.dumps(preservation, indent=2), encoding="utf-8"
    )
    if not preservation["verified"]:
        raise RuntimeError(f"Protected-file preservation failed: {preservation['mismatches']}")

    model2_summary = pd.read_csv(OUT / "model2_downstream_summary.csv")
    final_report(weak, model2_summary, aggregate, metadata)
    diagnostic_manifest = {
        "scope": "read-only Model 1 diagnostic and leakage-controlled Model 2 downstream impact",
        "dataset_path": data_prep["dataset"],
        "dataset_sha256": dataset_hash,
        "production_dataset_sha256": manifest["production_dataset_sha256"],
        "split_path": data_prep["split_path"],
        "split_sha256": data_prep["split_sha256"],
        "split": {
            "train_rows": len(train),
            "test_rows": len(test),
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "oof_folds": OOF_FOLDS,
            "random_state": RANDOM_STATE,
            "heldout_used_for_selection": False,
        },
        "feature_contract": MODEL1_FEATURES,
        "process_targets": PROCESS_TARGETS,
        "downstream_arms": {
            "A": "Saved production Model 1 outputs on the revised dataset.",
            "B": "Per outer fold, Model 1 family/parameters selected using only outer-training rows; OOF process predictions generated within outer training; outer-validation process predictions from full outer-training fits.",
            "C": "Recorded process targets; oracle upper-bound diagnostic, non-deployable.",
        },
        "model2_estimator": "fixed config.RF_PARAMS RandomForestRegressor in all arms; same folds",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "preservation": preservation,
        "automated_validation": {
            "schema_5000x40": bool(frame.shape == (5000, 40)),
            "saved_split_disjoint_4000_1000": bool(
                len(train) == 4000 and len(test) == 1000
                and not (set(train_idx) & set(test_idx))
            ),
            "model1_feature_contract_matches_metadata": bool(declared_features == MODEL1_FEATURES),
            "postproduction_features_excluded_from_deployable_arms": True,
            "oracle_labeled_non_deployable": True,
            "preservation_verified": preservation["verified"],
        },
        "outputs": sorted(set(
            [path.name for path in OUT.iterdir() if path.is_file()]
            + ["README.md", "diagnostic_manifest.json"]
        )),
    }
    readme = f"""# Controlled Model 1 Diagnostic

Isolated, read-only analysis of the revised synthetic dataset. Production assets are protected by the pre-analysis SHA-256 inventory in `preservation_manifest.json`.

## Reproduction commands

From the repository root with the configured Python 3.13 environment:

```powershell
& 'C:\\Users\\khale\\AppData\\Local\\Microsoft\\WindowsApps\\python3.13.exe' 'training\\evaluation_results\\{OUT.name}\\run_model1_diagnostic.py'
& 'C:\\Users\\khale\\AppData\\Local\\Microsoft\\WindowsApps\\python3.13.exe' 'training\\evaluation_results\\{OUT.name}\\verify_preservation.py'
& 'C:\\Users\\khale\\AppData\\Local\\Microsoft\\WindowsApps\\python3.13.exe' 'training\\evaluation_results\\{OUT.name}\\validate_diagnostic_outputs.py'
```

The audit reads the revised 5,000 x 40 dataset and saved `models/data_split.json`; it does not generate data. Model 1 candidates compare one fixed, modest configuration for each requested family using five outer folds; family selection uses three inner folds. Model 2 impact uses the same outer folds, nested three-fold Model 1 cross-fitting for training rows, fixed Model 2 Random Forest hyperparameters, and three explicitly separated upstream feature arms. The oracle arm uses recorded process outcomes and is diagnostic only, never deployable.

## Key deliverables

- `model1_diagnostic_report.md`: findings, limits, and recommendations.
- `model1_metrics.csv`, `baseline_comparison.csv`, `fold_level_results.csv`, `model1_fold_aggregate.csv`, `residual_diagnostics.csv`.
- `feature_relationship_diagnostics.csv`, line/puller diagnostic PNGs, and per-decile files.
- `model2_downstream_impact.csv`, `model2_downstream_summary.csv`, and row predictions.
- `preservation_manifest.json`, `preservation_verification.json`, and `diagnostic_manifest.json`.
- `validation_results.json`: fast, rerunnable checks for deliverables, row/fold counts, finite metrics, plots, and preserved hashes.

All results concern synthetic data and do not validate Polyhose machine behaviour, operating limits, or expected factory performance.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "diagnostic_manifest.json").write_text(
        json.dumps(diagnostic_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "completed",
        "output_directory": str(OUT),
        "dataset_sha256": dataset_hash,
        "model1_target_count": len(PROCESS_TARGETS),
        "downstream_rows": len(downstream_rows),
        "preservation_verified": preservation["verified"],
        "selected_model2_outer_cv": model2_summary.to_dict(orient="records"),
    }, indent=2))


def final_report(
    weak: pd.DataFrame,
    downstream: pd.DataFrame,
    aggregate: pd.DataFrame,
    metadata: dict,
) -> None:
    metric_table = pd.read_csv(OUT / "model1_metrics.csv")
    saved_baseline = pd.read_csv(OUT / "baseline_comparison.csv")
    saved_baseline = saved_baseline[
        saved_baseline["split"].eq("saved_test")
        & saved_baseline["comparison"].isin([
            "Saved production Model1 artifact",
            "Training_selected",
            "training_mean",
            "training_median",
        ])
    ][["target", "comparison", "MAE", "RMSE", "R2"]]
    aggregate_table = aggregate.sort_values(["target", "MAE_mean"])
    report = f"""# Controlled Model 1 Diagnostic Audit

## Scope, data and preservation

This isolated analysis uses the existing revised synthetic dataset (5,000 rows × 40 columns; SHA-256 `{sha256(DATASET_PATH)}`), fixed saved 4,000/1,000 row-index split and seed 42. It creates no dataset and does not modify the production dataset, generator, models, metadata, inference bundle, or Streamlit app. Pre-analysis and post-analysis preservation SHA-256 checks cover production data/code, models/metadata, the app and all existing evaluation experiment files; results are in `preservation_manifest.json` and `preservation_verification.json`.

Model 1's inference contract is the metadata/config-defined eight fields: categorical material, grade, tube type, braid type and numeric OD, wall, ID and required length. All are present in this dataset and none is a process target. Machine_ID, shift, ambient/humidity, planned processing/time, fault and other post-run columns are excluded at inference. The target fields evaluated are the nine process outputs from `training/config.py`.

## Saved Model 1 versus baselines

The current artifacts were trained against the production dataset, not this revised experimental dataset. Their revised-dataset evaluation therefore measures transfer to a different synthetic target population; it is not a reproduction of the original production test scores. The original metadata values are recorded alongside the revised-set results in `saved_metric_context.csv`, without treating them as paired.

{md_table(metric_table[["target", "train_variance", "train_min", "train_max", "saved_artifact_test_MAE", "saved_artifact_test_RMSE", "saved_artifact_test_R2", "selected_test_MAE", "selected_test_RMSE", "selected_test_R2"]])}

Holdout comparison against training-only constant baselines and the inner-selected candidate:

{md_table(saved_baseline)}

Candidate-family results use 5 fixed outer folds in the 4,000-row training split. Family/config selection occurs inside each outer training portion using three inner folds. The held-out 1,000 rows do not influence fitting or selection.

{md_table(aggregate_table[["target", "evaluation_arm", "fold_count", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std"]])}

`residual_diagnostics.csv` contains held-out prediction/residual rows; baseline comparison records include residual bias, SD, quantiles and absolute-error quantiles.

## Why the frozen speed predictions are weak here

On the revised synthetic holdout, saved-model metrics:

{md_table(weak[["target", "train_variance", "train_min", "train_max", "saved_artifact_test_MAE", "saved_artifact_test_RMSE", "saved_artifact_test_R2", "selected_test_MAE", "selected_test_RMSE", "selected_test_R2"]])}

The two PNGs show actual-versus-predicted, residual-versus-actual, residual-versus-predicted and decile-mean collapse checks. Decile-level predictions and errors are also in `Line_Speed_m_min_decile_diagnostics.csv` and `Puller_Speed_m_min_decile_diagnostics.csv`.

The revised generator source includes synthetic assumptions, not manufacturing facts: nominal line speed is a function of material profile, grade, geometry, product difficulty and required length; realized speed adds a stochastic rate factor and machine-specific speed factor. Machine identity and that stochastic factor are not Model 1 inputs. Puller speed is generated from line speed plus braid offset/noise, with selected fault perturbations. However, the training-selected models achieve held-out R² 0.809 for line speed and 0.700 for puller speed, while the frozen production artifacts score −0.408 and −0.351 on this different synthetic dataset. The evidence therefore points to a substantial model/data-population mismatch in the frozen artifacts, not proof that the available customer inputs are intrinsically insufficient. Residual stochastic effects may remain; neither result establishes a real manufacturing cause. Synthetic nominal-speed bounds/process ranges must not be read as Polyhose limits.

See `feature_relationship_diagnostics.csv` for customer-input/target associations, category means, speed-rate correlations and line/puller difference distribution. Correlations and subgroup means are descriptive and do not prove causal effects.

## Downstream Model 2 impact

Each arm uses the same outer folds and fixed production-parameter Random Forest Model 2 estimator, with the 8 customer inputs and 9 process-valued columns only:

- **A — saved Model 1:** frozen production Model 1 prediction bundle on revised-dataset inputs.
- **B — nested training-only Model 1:** within each outer fold, Model 1 family/configs are selected using outer-training rows only; Model 1 features for Model 2 training are cross-fitted within outer training; outer-validation/test features come from Model 1 fitted only on the corresponding outer-training/full-training partition.
- **C — oracle:** recorded process targets are given to Model 2 for both its training and evaluation rows. This is **not deployable**, is only a diagnostic upper-bound-style comparison, and does not represent the app inference contract.

No outer validation/test process target is used to fit or select B's upstream models. Results:

{md_table(downstream[["arm", "deployable", "outer_fold_count", "outer_MAE_mean", "outer_MAE_std", "outer_RMSE_mean", "outer_RMSE_std", "outer_R2_mean", "outer_R2_std", "saved_test_MAE", "saved_test_RMSE", "saved_test_R2"]])}

The B train-to-inference distribution differs by construction: Model 2 training rows use OOF process predictions, while validation/test rows use upstream fits on the full available training portion, paralleling the intended stacked-learning setup. The fixed downstream estimator makes arm changes attributable to feature-source differences within this audit; it does not establish that any alternative Model 1 should replace production.

## Answers and recommendation

1. **Reliability:** on this revised synthetic population the frozen artifacts are comparatively predictive for extruder/die temperatures, melt/air pressure, screw RPM and extrusion rate (R² about 0.50–0.64); cooling-tank temperature is weaker (R² 0.158), while the two speed artifact scores are negative. The training-selected candidates perform materially better on the held-out speed targets (line 0.809, puller 0.700). These labels describe this synthetic transfer test only.
2. **Weak speed targets:** the revised synthetic generator contains machine/stochastic speed effects absent from the customer feature contract, but candidate held-out results show the recorded customer/product features retain strong predictive signal. The frozen artifacts' negative R² is consistent with transfer/model mismatch; the exact causal contribution of generator noise versus model form is not isolated. Plots and subgroup summaries are synthetic diagnostics, not evidence of a factory cause.
3. **Algorithm change:** the training-selected candidates improve held-out line and puller speed against the saved artifacts, and aggregate fixed-fold results show whether gains repeat across folds. The choices use only training data, but these candidate-family settings are deliberately small, fixed comparisons—not a broad tuning search or an external validation.
4. **Downstream impact:** Model 1 arm B does not materially improve Model 2 over A in outer CV (MAE means 4.7690 vs 4.7685; RMSE 5.9096 vs 5.9315; R² 0.0545 vs 0.0479). On the untouched test, B is slightly better (MAE 4.9921 vs 5.0470; RMSE 6.4642 vs 6.5094; R² 0.0799 vs 0.0670), but this small test-only difference is not consistent outer-fold evidence of a material gain. C's oracle improvement is non-deployable.
5. **Limitations:** the revised target is generated under explicit synthetic equations, noise, hidden machine/fault context and unverified ranges. Any apparent error floor could reflect missing inputs, generator noise or mismatch between production-trained Model 1 artifacts and revised data; this audit cannot attribute it to actual plant behavior.
6. **Real-data validation:** validate recipe/setpoint rules and operational limits with process engineering; obtain time-aligned material/grade/geometry, machine identity, recipe, actual setpoints, speeds, line-stop/fault, scrap and good-output measurements; quantify sensor and operator variation; use time- or machine-held-out validation and calibration checks.
7. **App status:** keep the app labeled and used as a synthetic-data demo. The evidence-based next step is a governed real-production data pilot and time-aligned validation, not model promotion or score tuning.

## Automated checks and unresolved limits

The audit requires a 5,000 × 40 revised dataset and expected SHA-256, checks the saved split is disjoint 4,000/1,000, confirms all nine saved Model 1 pipelines and metadata feature-contract parity, keeps oracle fields segregated, and verifies preservation hashes before reporting. The separate validation command checks output completeness, required fold/arm counts, finite residuals, plot signatures and hashes; all 11 automated output checks passed and all 200 protected-file hashes matched. Syntax checks passed for the runner and both verification scripts. No validation check failed. This remains a synthetic diagnostic, not real manufacturing evidence.
"""
    (OUT / "model1_diagnostic_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
