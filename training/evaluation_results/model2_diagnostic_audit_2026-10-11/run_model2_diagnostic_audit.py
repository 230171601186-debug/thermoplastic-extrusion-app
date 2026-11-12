"""Read-only diagnostic audit of Model 2 on the revised synthetic dataset.

All generated files stay in this directory. Production data, models, metadata,
training code, and the two prior experiment directories are read-only inputs.
"""

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    METADATA_PATH,
    MODEL1_FEATURES,
    PROCESS_MODEL_FILENAMES,
    PROCESS_TARGETS,
    RANDOM_STATE,
    SPLIT_INDICES_PATH,
)
from preprocessing import (  # noqa: E402
    attach_predicted_process,
    build_customer_preprocessor,
    build_stacked_preprocessor,
    load_process_pipelines,
    make_regressor_pipeline,
    make_stacked_pipeline,
)

REVISED_DIR = ROOT / "training/evaluation_results/model2_generator_experiment_2026-10-11_revised"
REVISED_DATASET = REVISED_DIR / "experimental_dataset.csv"
EXPECTED_REVISED_SHA256 = "0db6c6d33f0a3d580b2f673cc3761b4c2551f48c1ca68d296303d9854a669b15"
OUTER_SPLITS = 5
INNER_SPLITS = 3
MIN_SUBGROUP_COUNT = 20
CUSTOMER_FIELDS = list(MODEL1_FEATURES)
PROCESS_FIELDS = list(PROCESS_TARGETS)
FULL_FIELDS = CUSTOMER_FIELDS + PROCESS_FIELDS
TARGET_COMPONENTS = [
    "Availability", "Performance", "Quality", "OEE",
    "Downtime_min", "Run_Time_min", "Actual_Processing_Length_m",
    "Good_Length_m", "Scrap_Length_m", "Scrap_Rate",
]

OUTPUT_CSVS = [
    "target_distribution_summary.csv",
    "feature_role_and_leakage_audit.csv",
    "feature_group_ablation_results.csv",
    "fold_level_model_comparison.csv",
    "aggregate_model_comparison.csv",
    "residual_diagnostics.csv",
    "subgroup_error_analysis.csv",
    "feature_relationship_diagnostics.csv",
    "model1_process_prediction_diagnostics.csv",
    "customer_input_sensitivity.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preservation() -> dict:
    manifest_path = OUT / "production_preservation_manifest.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    mismatches = []
    for relative, expected in before["file_sha256"].items():
        current_path = ROOT / Path(relative)
        if not current_path.exists():
            mismatches.append({"path": relative, "issue": "missing"})
        else:
            actual = sha256(current_path)
            if actual != expected:
                mismatches.append(
                    {"path": relative, "expected": expected, "actual": actual}
                )
    return {
        "verified": not mismatches,
        "verified_file_count": len(before["file_sha256"]) - len(mismatches),
        "expected_file_count": len(before["file_sha256"]),
        "dataset_sha256_before": before["dataset_sha256"],
        "dataset_sha256_after": sha256(
            ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
        ),
        "existing_experiment_directory_file_counts": before[
            "experiment_directory_file_counts"
        ],
        "mismatches": mismatches,
    }


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "R2": float(r2_score(y_true, prediction)),
    }


def pipeline_for(
    estimator,
    fields: list[str],
    *,
    scale: bool = False,
) -> Pipeline:
    categoricals = [column for column in CATEGORICAL_FEATURES if column in fields]
    numeric = [column for column in fields if column not in categoricals]
    if fields == CUSTOMER_FIELDS:
        preprocessor = build_customer_preprocessor()
    elif categoricals:
        process_numeric = [column for column in numeric if column not in CUSTOMER_NUMERIC_FEATURES]
        preprocessor = build_stacked_preprocessor(process_numeric)
    else:
        preprocessor = ColumnTransformer(
            [("num", "passthrough", numeric)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    steps = [("preprocess", preprocessor)]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def rf_estimator(seed_offset: int = 0, **params):
    options = {
        "n_estimators": 160,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "n_jobs": 1,
        "random_state": RANDOM_STATE + seed_offset,
    }
    options.update(params)
    return RandomForestRegressor(**options)


def model_grids() -> dict[str, list[tuple[object, dict, bool]]]:
    """Small, fixed search grids; each choice is scored only within inner folds."""
    grids: dict[str, list[tuple[object, dict, bool]]] = {
        "RandomForest": [
            (rf_estimator(max_features=value), {}, False)
            for value in (0.7, 1.0)
        ],
        "ExtraTrees": [
            (
                ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=leaf,
                    max_features=0.8,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
                {"min_samples_leaf": leaf},
                False,
            )
            for leaf in (2, 5)
        ],
        "HistGradientBoosting": [
            (
                HistGradientBoostingRegressor(
                    max_iter=iterations,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=regularization,
                    random_state=RANDOM_STATE,
                ),
                {"max_iter": iterations, "l2_regularization": regularization},
                False,
            )
            for iterations, regularization in ((120, 3.0), (220, 5.0))
        ],
        "Ridge": [
            (Ridge(alpha=alpha), {"alpha": alpha}, True)
            for alpha in (1.0, 10.0, 50.0)
        ],
    }
    try:
        from lightgbm import LGBMRegressor

        grids["LightGBM"] = [
            (
                LGBMRegressor(
                    n_estimators=trees,
                    max_depth=5,
                    num_leaves=leaves,
                    learning_rate=0.04,
                    min_child_samples=20,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    reg_lambda=3.0,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                ),
                {"n_estimators": trees, "num_leaves": leaves},
                False,
            )
            for trees, leaves in ((150, 15), (200, 31))
        ]
    except ImportError:
        pass
    return grids


def feature_data(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    return frame.loc[:, fields]


def select_process_subset(
    inner_train: pd.DataFrame,
    y: pd.Series,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[str], dict[str, float]]:
    """Retain only process predictions with positive inner-CV marginal value."""
    baseline_scores: list[float] = []
    process_scores = {feature: [] for feature in PROCESS_FIELDS}
    for fit_positions, validation_positions in inner_splits:
        train_fold = inner_train.iloc[fit_positions]
        validation_fold = inner_train.iloc[validation_positions]
        y_fit = y.iloc[fit_positions].to_numpy(dtype=float)
        y_validation = y.iloc[validation_positions].to_numpy(dtype=float)
        customer_model = pipeline_for(rf_estimator(), CUSTOMER_FIELDS)
        customer_model.fit(feature_data(train_fold, CUSTOMER_FIELDS), y_fit)
        baseline_prediction = customer_model.predict(
            feature_data(validation_fold, CUSTOMER_FIELDS)
        )
        baseline_scores.append(mean_absolute_error(y_validation, baseline_prediction))
        for feature in PROCESS_FIELDS:
            fields = CUSTOMER_FIELDS + [feature]
            candidate = pipeline_for(rf_estimator(), fields)
            candidate.fit(feature_data(train_fold, fields), y_fit)
            prediction = candidate.predict(feature_data(validation_fold, fields))
            process_scores[feature].append(mean_absolute_error(y_validation, prediction))
    marginal_improvement = {
        feature: float(
            np.mean(baseline_scores) - np.mean(scores)
        )
        for feature, scores in process_scores.items()
    }
    selected = [
        feature
        for feature in PROCESS_FIELDS
        if marginal_improvement[feature] > 0
    ]
    if not selected:
        selected = [
            max(marginal_improvement, key=marginal_improvement.get)
        ]
    return selected, marginal_improvement


def select_parameters(
    name: str,
    candidates: list[tuple[object, dict, bool]],
    train: pd.DataFrame,
    y: pd.Series,
    fields: list[str],
    splitter: KFold,
) -> tuple[object, dict, bool, float]:
    best: tuple[object, dict, bool, float] | None = None
    for estimator, params, scale in candidates:
        fold_mae = []
        for fit_positions, validation_positions in splitter.split(train):
            fit_frame = train.iloc[fit_positions]
            validation_frame = train.iloc[validation_positions]
            fitted = pipeline_for(estimator, fields, scale=scale)
            fitted.fit(
                feature_data(fit_frame, fields),
                y.iloc[fit_positions].to_numpy(dtype=float),
            )
            prediction = fitted.predict(feature_data(validation_frame, fields))
            fold_mae.append(
                mean_absolute_error(
                    y.iloc[validation_positions].to_numpy(dtype=float), prediction
                )
            )
        score = float(np.mean(fold_mae))
        if best is None or score < best[3]:
            best = (estimator, params, scale, score)
    if best is None:
        raise RuntimeError(f"No candidate configurations supplied for {name}")
    return best


def target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    numeric_columns = [EFFICIENCY_TARGET, *TARGET_COMPONENTS]
    for column in numeric_columns:
        series = frame[column].astype(float)
        rows.append({
            "kind": "numeric",
            "variable": column,
            "level": "",
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "unique_values": int(series.nunique()),
            "frequency": np.nan,
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "variance": float(series.var(ddof=1)),
            "min": float(series.min()),
            "q05": float(series.quantile(.05)),
            "q25": float(series.quantile(.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(.75)),
            "q95": float(series.quantile(.95)),
            "max": float(series.max()),
        })
    for level, group in frame.groupby("Fault_Code", observed=True):
        rows.append({
            "kind": "fault_category",
            "variable": "Fault_Code",
            "level": str(level),
            "count": len(group),
            "missing": 0,
            "unique_values": 1,
            "frequency": float(len(group) / len(frame)),
            "mean": float(group[EFFICIENCY_TARGET].mean()),
            "std": float(group[EFFICIENCY_TARGET].std()),
            "variance": np.nan,
            "min": float(group[EFFICIENCY_TARGET].min()),
            "q05": np.nan,
            "q25": np.nan,
            "median": float(group[EFFICIENCY_TARGET].median()),
            "q75": np.nan,
            "q95": np.nan,
            "max": float(group[EFFICIENCY_TARGET].max()),
        })
    return pd.DataFrame(rows)


def feature_roles(frame: pd.DataFrame) -> pd.DataFrame:
    reasons: dict[str, tuple[str, str, str, str]] = {}
    for feature in CUSTOMER_FIELDS:
        reasons[feature] = (
            "Customer/product input",
            "Yes, per planned inference contract",
            "Low",
            "Included in the customer-facing Model 2 feature contract.",
        )
    for feature in PROCESS_FIELDS:
        reasons[feature] = (
            "Model 1 process prediction",
            "Yes, as a Model 1 prediction",
            "Low when predicted; high if recorded outcome is substituted",
            "The deployed stack must use Model 1 outputs, not post-run recorded process values.",
        )
    reasons.update({
        "Record_ID": ("Identifier", "No", "High", "Identifier has no causal/predictive production meaning and may encode row order."),
        "Machine_ID": ("Plant context", "Potentially; not in specified UI", "Medium", "Could be known after scheduling, but availability and supported use are unverified."),
        "Shift": ("Plant context", "Potentially; not in specified UI", "Medium", "May be known at scheduling; not part of current customer input contract."),
        "Ambient_Temperature_C": ("Plant context", "Potentially; measure at planning", "Medium", "Timing and sensor availability at inference require confirmation."),
        "Humidity_percent": ("Plant context", "Potentially; measure at planning", "Medium", "Timing and sensor availability at inference require confirmation."),
        "Planned_Scrap_Allowance_pct": ("Planning context", "Potentially, but not current UI", "Medium", "Plan field may be known but the current stack intentionally excludes it."),
        "Planned_Processing_Length_m": ("Planning context", "Potentially, but derived/ambiguous", "Medium", "Can be planned pre-run but was not part of the documented customer-facing input set."),
        "Planned_Production_Time_min": ("Planning context", "Potentially, but derived/ambiguous", "Medium", "May be set pre-run; generation depends on synthetic speed/time assumptions."),
        "Operational_Efficiency_pct": ("Target", "No; target", "Direct target", "This is the prediction target."),
    })
    post = {
        "OEE": "Direct target component; mathematically multiplied to create efficiency.",
        "Availability": "Direct target component; runtime/planned time.",
        "Performance": "Direct target component; output rate divided by synthetic nominal speed.",
        "Quality": "Direct target component; good output / actual processing length.",
        "Good_Length_m": "Post-run output directly informs Quality and completion.",
        "Scrap_Length_m": "Post-run quantity; directly determines Good_Length_m and Quality.",
        "Scrap_Rate": "Post-run outcome directly determines Quality.",
        "Actual_Processing_Length_m": "Post-run output; directly used in performance/quality/accounting.",
        "Length_Shortfall_m": "Post-run derivation from planned and actual length.",
        "Order_Status": "Post-run fulfillment status derived from good output.",
        "Fault_Code": "Fault realization occurs during/after operation; affects downtime and scrap.",
        "Power_Consumption_kWh": "Post-run measurement; may proxy runtime/process conditions.",
        "Downtime_min": "Post-run operational outcome; directly changes Availability.",
        "Run_Time_min": "Post-run operational outcome; directly changes Availability and output.",
    }
    for feature, reason in post.items():
        reasons[feature] = (
            "Post-production outcome/target component",
            "No",
            "High",
            reason,
        )
    rows = []
    target = frame[EFFICIENCY_TARGET]
    for feature in frame.columns:
        role, available, risk, reason = reasons.get(
            feature,
            ("Unclassified", "Investigate", "Investigate", "No role mapping was assigned."),
        )
        exact_copy = False
        if feature != EFFICIENCY_TARGET and pd.api.types.is_numeric_dtype(frame[feature]):
            exact_copy = bool(
                np.allclose(
                    frame[feature].to_numpy(dtype=float),
                    target.to_numpy(dtype=float),
                    rtol=0,
                    atol=1e-12,
                )
            )
        if exact_copy:
            risk = "Critical"
            reason += " Exact duplicate of target values detected."
        rows.append({
            "Feature": feature,
            "Role": role,
            "Available at prediction time?": available,
            "Target leakage risk": risk,
            "Reason": reason,
            "Recommendation": (
                "Exclude from model input; use only for target/accounting audit."
                if risk in {"High", "Critical", "Direct target"}
                else "Use only in the role stated; verify timing for ambiguous context."
            ),
        })
    return pd.DataFrame(rows)


def relationship_diagnostics(frame: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    joined = frame.reset_index(drop=True).copy()
    for process in PROCESS_FIELDS:
        joined[f"Predicted_{process}"] = predicted[process].to_numpy()
    rows = []
    candidates = CUSTOMER_FIELDS + PROCESS_FIELDS
    for feature in candidates:
        if pd.api.types.is_numeric_dtype(joined[feature]):
            rows.append({
                "feature": feature,
                "feature_role": "customer_input" if feature in CUSTOMER_FIELDS else "recorded_actual_process_target",
                "statistic": "pearson",
                "level": "",
                "count": len(joined),
                "value": float(joined[feature].corr(joined[EFFICIENCY_TARGET], method="pearson")),
                "mean_efficiency": np.nan,
                "prediction_error_MAE": np.nan,
            })
            if feature in PROCESS_FIELDS:
                p = joined[f"Predicted_{feature}"]
                rows.append({
                    "feature": f"Predicted_{feature}",
                    "feature_role": "inference_time_Model1_prediction",
                    "statistic": "pearson",
                    "level": "",
                    "count": len(joined),
                    "value": float(p.corr(joined[EFFICIENCY_TARGET])),
                    "mean_efficiency": np.nan,
                    "prediction_error_MAE": float(
                        mean_absolute_error(joined[feature], p)
                    ),
                })
        else:
            for level, group in joined.groupby(feature, observed=True):
                rows.append({
                    "feature": feature,
                    "feature_role": "customer_input",
                    "statistic": "category_mean",
                    "level": str(level),
                    "count": len(group),
                    "value": np.nan,
                    "mean_efficiency": float(group[EFFICIENCY_TARGET].mean()),
                    "prediction_error_MAE": np.nan,
                })
    for component in TARGET_COMPONENTS:
        rows.append({
            "feature": component,
            "feature_role": "post_outcome_component_diagnostic_only",
            "statistic": "pearson",
            "level": "",
            "count": len(joined),
            "value": float(joined[component].corr(joined[EFFICIENCY_TARGET])),
            "mean_efficiency": np.nan,
            "prediction_error_MAE": np.nan,
        })
    return pd.DataFrame(rows)


def summarize_model1_process(
    frame: pd.DataFrame,
    predicted: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for process in PROCESS_FIELDS:
        truth = frame[process].to_numpy(dtype=float)
        estimate = predicted[process].to_numpy(dtype=float)
        rows.append({
            "process_target": process,
            "diagnostic_scope": "revised synthetic dataset; frozen production inference artifact",
            **metrics(truth, estimate),
            "actual_mean": float(np.mean(truth)),
            "actual_std": float(np.std(truth, ddof=1)),
            "prediction_mean": float(np.mean(estimate)),
            "prediction_std": float(np.std(estimate, ddof=1)),
        })
    return pd.DataFrame(rows)


def evaluate_ablation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fold_number: int,
) -> tuple[list[dict], list[dict], list[str], dict[str, float]]:
    rows = []
    predictions = []
    inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE + 17)
    inner_splits = list(inner.split(train))
    selected_process, marginal = select_process_subset(train, y_train, inner_splits)
    groups = {
        "customer_inputs_only_RF": CUSTOMER_FIELDS,
        "predicted_process_only_RF": PROCESS_FIELDS,
        "all_17_inference_inputs_RF": FULL_FIELDS,
        "customer_plus_inner_selected_process_RF": CUSTOMER_FIELDS + selected_process,
    }
    for name, fields in groups.items():
        fitted = pipeline_for(rf_estimator(), fields)
        fitted.fit(feature_data(train, fields), y_train.to_numpy(dtype=float))
        predicted = fitted.predict(feature_data(test, fields))
        rows.append({
            "fold": fold_number,
            "configuration": name,
            "feature_count": len(fields),
            "selected_process_features": "|".join(selected_process) if "selected_process" in name else "",
            **metrics(y_test.to_numpy(dtype=float), predicted),
        })
        predictions.append({
            "fold": fold_number,
            "configuration": name,
            "y_true": y_test.to_numpy(dtype=float),
            "y_pred": predicted,
        })
    for baseline in ("training_fold_mean", "training_fold_median"):
        value = (
            float(y_train.mean())
            if baseline == "training_fold_mean"
            else float(y_train.median())
        )
        predicted = np.full(len(test), value, dtype=float)
        rows.append({
            "fold": fold_number,
            "configuration": baseline,
            "feature_count": 0,
            "selected_process_features": "",
            **metrics(y_test.to_numpy(dtype=float), predicted),
        })
        predictions.append({
            "fold": fold_number,
            "configuration": baseline,
            "y_true": y_test.to_numpy(dtype=float),
            "y_pred": predicted,
        })
    marginal_rows = [
        {"fold": fold_number, "process_feature": feature, "inner_cv_MAE_improvement": improvement}
        for feature, improvement in marginal.items()
    ]
    return rows, predictions, selected_process, {
        feature: value for feature, value in marginal.items()
    }


def model_family_search(
    train: pd.DataFrame,
    y_train: pd.Series,
    fields: list[str],
    grids: dict[str, list[tuple[object, dict, bool]]],
    seed_offset: int,
) -> tuple[dict, dict]:
    splitter = KFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + seed_offset,
    )
    chosen = {}
    logs = {}
    for name, candidates in grids.items():
        estimator, params, scale, score = select_parameters(
            name, candidates, train, y_train, fields, splitter
        )
        chosen[name] = (estimator, params, scale)
        logs[name] = {"inner_MAE": score, "parameters": params}
    return chosen, logs


def append_residual_rows(
    rows: list[dict],
    split: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold: str,
) -> None:
    residual = y_pred - y_true
    for i, (actual, prediction, error) in enumerate(zip(y_true, y_pred, residual)):
        rows.append({
            "split": split,
            "fold": fold,
            "model": model,
            "row_order_within_evaluation": i,
            "actual": float(actual),
            "predicted": float(prediction),
            "residual_pred_minus_actual": float(error),
            "absolute_error": float(abs(error)),
        })


def subgroup_errors(
    residual_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    training_target: pd.Series,
    model_names: list[str],
) -> pd.DataFrame:
    test_context = test_frame[
        ["Record_ID", "Material_Type", "Tube_Type", "Braid_Type", "Machine_ID", "Fault_Code"]
    ].reset_index(drop=True)
    test_context["row_order_within_evaluation"] = np.arange(len(test_context))
    merged = residual_frame[residual_frame["split"] == "held_out_test"].merge(
        test_context,
        on="row_order_within_evaluation",
        how="left",
    )
    quantiles = np.unique(
        np.quantile(training_target.to_numpy(dtype=float), [0, .25, .5, .75, 1])
    )
    if len(quantiles) >= 3:
        merged["target_quantile"] = pd.cut(
            merged["actual"],
            bins=np.unique(np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))),
            labels=False,
            include_lowest=True,
        )
    groups = [
        "Material_Type", "Tube_Type", "Braid_Type", "Machine_ID", "Fault_Code",
        "target_quantile",
    ]
    rows = []
    for model in model_names:
        subset = merged[merged["model"] == model]
        for group_field in groups:
            for value, group in subset.groupby(group_field, observed=True, dropna=False):
                residual = group["residual_pred_minus_actual"].to_numpy(dtype=float)
                rows.append({
                    "model": model,
                    "grouping": group_field,
                    "group": str(value),
                    "count": len(group),
                    "MAE": float(np.mean(np.abs(residual))),
                    "RMSE": float(np.sqrt(np.mean(residual**2))),
                    "mean_residual": float(np.mean(residual)),
                    "reliable_at_minimum_n": len(group) >= MIN_SUBGROUP_COUNT,
                    "minimum_n_threshold": MIN_SUBGROUP_COUNT,
                })
    return pd.DataFrame(rows)


def sensitivity_analysis(
    model: Pipeline,
    model_name: str,
    fields: list[str],
    test: pd.DataFrame,
    training_customer: pd.DataFrame,
    process_models: dict,
    max_rows: int = 250,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE + 2026)
    selected = rng.choice(len(test), size=min(max_rows, len(test)), replace=False)
    base_customer = test.iloc[selected][CUSTOMER_FIELDS].copy().reset_index(drop=True)

    def predict_customer(customer: pd.DataFrame) -> np.ndarray:
        stacked = attach_predicted_process(customer, process_models)
        return model.predict(feature_data(stacked, fields))

    base = predict_customer(base_customer)
    rows = []
    numeric_steps = {
        "Target_Tube_OD_mm": 0.10,
        "Target_Tube_Wall_mm": 0.05,
        "Target_Tube_ID_mm": 0.10,
        "Customer_Required_Length_m": None,
    }
    for feature, step in numeric_steps.items():
        for direction in (-1, 1):
            changed = base_customer.copy()
            delta = (
                max(1.0, float(changed[feature].median()) * 0.01)
                if step is None
                else step
            )
            changed[feature] = changed[feature].astype(float) + direction * delta
            if feature == "Target_Tube_OD_mm":
                changed["Target_Tube_ID_mm"] = (
                    changed["Target_Tube_OD_mm"] - 2 * changed["Target_Tube_Wall_mm"]
                )
            elif feature == "Target_Tube_Wall_mm":
                changed["Target_Tube_ID_mm"] = (
                    changed["Target_Tube_OD_mm"] - 2 * changed["Target_Tube_Wall_mm"]
                )
            else:
                changed["Target_Tube_Wall_mm"] = (
                    changed["Target_Tube_OD_mm"] - changed["Target_Tube_ID_mm"]
                ) / 2
            prediction = predict_customer(changed)
            difference = prediction - base
            rows.append({
                "model": model_name,
                "feature": feature,
                "perturbation": f"{direction:+g} x {delta:.4g}",
                "perturbation_type": "local_numeric; geometry kept valid",
                "rows_evaluated": len(base_customer),
                "mean_prediction_change": float(np.mean(difference)),
                "median_prediction_change": float(np.median(difference)),
                "q10_prediction_change": float(np.quantile(difference, .1)),
                "q90_prediction_change": float(np.quantile(difference, .9)),
                "mean_absolute_prediction_change": float(np.mean(np.abs(difference))),
                "fraction_positive_change": float(np.mean(difference > 0)),
            })
    for feature in CATEGORICAL_FEATURES:
        levels = training_customer[feature].value_counts().index.tolist()
        if len(levels) < 2:
            continue
        alternative = str(levels[1])
        changed = base_customer.copy()
        changed[feature] = alternative
        difference = predict_customer(changed) - base
        rows.append({
            "model": model_name,
            "feature": feature,
            "perturbation": f"set to common alternate level {alternative}",
            "perturbation_type": "categorical contrast; not a local derivative",
            "rows_evaluated": len(base_customer),
            "mean_prediction_change": float(np.mean(difference)),
            "median_prediction_change": float(np.median(difference)),
            "q10_prediction_change": float(np.quantile(difference, .1)),
            "q90_prediction_change": float(np.quantile(difference, .9)),
            "mean_absolute_prediction_change": float(np.mean(np.abs(difference))),
            "fraction_positive_change": float(np.mean(difference > 0)),
        })
    return pd.DataFrame(rows)


def create_plots(
    frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    cv_predictions: pd.DataFrame,
    test_residuals: pd.DataFrame,
    selected_model: str,
) -> None:
    target = frame[EFFICIENCY_TARGET].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(target, bins=35, color="#3178a8", alpha=.85)
    ax.set(title="Revised synthetic efficiency target distribution", xlabel="Efficiency (%)", ylabel="Rows")
    fig.tight_layout()
    fig.savefig(OUT / "efficiency_target_histogram.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, component in zip(axes.ravel(), ["Availability", "Performance", "Quality", "OEE"]):
        ax.scatter(frame[component], frame[EFFICIENCY_TARGET], s=7, alpha=.25)
        ax.set(title=f"Efficiency vs {component}", xlabel=component, ylabel="Efficiency (%)")
    fig.tight_layout()
    fig.savefig(OUT / "efficiency_vs_oee_components.png", dpi=150)
    plt.close(fig)

    data = cv_predictions[
        cv_predictions["model"] == "all_17_inference_inputs_RF"
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data["actual"], data["predicted"], s=9, alpha=.3)
    limits = [
        min(data["actual"].min(), data["predicted"].min()),
        max(data["actual"].max(), data["predicted"].max()),
    ]
    ax.plot(limits, limits, color="black", linewidth=1)
    ax.set(title="Outer-CV actual vs predicted (full-stack RF)", xlabel="Actual efficiency (%)", ylabel="Predicted efficiency (%)")
    fig.tight_layout()
    fig.savefig(OUT / "outer_cv_actual_vs_predicted_full_stack_rf.png", dpi=150)
    plt.close(fig)

    residual = test_residuals[test_residuals["model"] == selected_model]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(residual["predicted"], residual["residual_pred_minus_actual"], s=10, alpha=.35)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Held-out residual vs prediction", xlabel="Predicted efficiency (%)", ylabel="Predicted − actual (pp)")
    axes[1].scatter(residual["actual"], residual["residual_pred_minus_actual"], s=10, alpha=.35)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(title="Held-out residual vs actual", xlabel="Actual efficiency (%)", ylabel="Predicted − actual (pp)")
    fig.tight_layout()
    fig.savefig(OUT / "heldout_residual_diagnostics.png", dpi=150)
    plt.close(fig)

    fault_summary = (
        test_residuals[test_residuals["model"] == selected_model]
        .merge(
            test_frame[["Fault_Code"]]
            .reset_index(drop=True)
            .assign(row_order_within_evaluation=np.arange(len(test_frame))),
            on="row_order_within_evaluation",
        )
        .groupby("Fault_Code", observed=True)
        .residual_pred_minus_actual.agg(["count", "mean"])
        .sort_values("mean")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fault_summary.index.astype(str), fault_summary["mean"], color="#8b5e83")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(title="Held-out mean residual by fault category (selected model)", xlabel="Predicted − actual (pp)")
    fig.tight_layout()
    fig.savefig(OUT / "heldout_residual_by_fault.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not OUT.exists():
        raise FileNotFoundError(f"Output directory must exist and be uniquely named: {OUT}")
    initial_preservation = verify_preservation()
    if not initial_preservation["verified"]:
        raise RuntimeError(f"Pre-analysis preservation checks failed: {initial_preservation['mismatches']}")
    if sha256(REVISED_DATASET) != EXPECTED_REVISED_SHA256:
        raise RuntimeError("Revised experimental dataset hash does not match the reviewed report.")
    frame = pd.read_csv(REVISED_DATASET, keep_default_na=False)
    if frame.shape != (5000, 40):
        raise RuntimeError(f"Unexpected revised experiment schema/shape: {frame.shape}")
    if EFFICIENCY_TARGET not in frame:
        raise RuntimeError(f"Missing target {EFFICIENCY_TARGET}")
    split = json.loads(SPLIT_INDICES_PATH.read_text(encoding="utf-8"))
    if len(split["train_index"]) != 4000 or len(split["test_index"]) != 1000:
        raise RuntimeError("Expected saved 4000/1000 split.")
    if set(split["train_index"]) & set(split["test_index"]):
        raise RuntimeError("Saved train/test split indices overlap.")
    process_models = load_process_pipelines()
    if set(process_models) != set(PROCESS_MODEL_FILENAMES):
        raise RuntimeError("Saved inference bundle does not provide exactly nine Model 1 pipelines.")
    process_features = attach_predicted_process(frame[MODEL1_FEATURES], process_models)
    if list(process_features.columns) != FULL_FIELDS:
        raise RuntimeError("Inference-time stacked input order is not the documented 8+9 contract.")
    model_frame = frame.copy()
    model_frame.loc[:, PROCESS_FIELDS] = process_features.loc[:, PROCESS_FIELDS].to_numpy()
    train = model_frame.loc[split["train_index"]].copy()
    test = model_frame.loc[split["test_index"]].copy()
    y_train = train[EFFICIENCY_TARGET].astype(float)
    y_test = test[EFFICIENCY_TARGET].astype(float)

    target_summary(frame).to_csv(OUT / "target_distribution_summary.csv", index=False)
    feature_roles(frame).to_csv(OUT / "feature_role_and_leakage_audit.csv", index=False)
    relationship_diagnostics(frame, process_features).to_csv(
        OUT / "feature_relationship_diagnostics.csv", index=False
    )
    process_diag = summarize_model1_process(frame, process_features)
    process_diag.to_csv(OUT / "model1_process_prediction_diagnostics.csv", index=False)
    groups = model_grids()

    outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    ablation_fold_rows = []
    ablation_selected = []
    model_fold_rows = []
    nested_predictions: list[dict] = []
    ablation_oof: list[dict] = []
    model_selection_log: list[dict] = []
    selected_feature_by_fold: dict[str, list[str]] = {}

    for fold_number, (fit_positions, validation_positions) in enumerate(outer.split(train), start=1):
        outer_train = train.iloc[fit_positions].copy()
        outer_validation = train.iloc[validation_positions].copy()
        y_fit = outer_train[EFFICIENCY_TARGET].astype(float)
        y_validation = outer_validation[EFFICIENCY_TARGET].astype(float)

        ablation_rows, prediction_sets, selected_process, marginal = evaluate_ablation(
            outer_train,
            outer_validation,
            y_fit,
            y_validation,
            fold_number,
        )
        ablation_fold_rows.extend(ablation_rows)
        selected_feature_by_fold[str(fold_number)] = selected_process
        for feature, value in marginal.items():
            ablation_selected.append({
                "fold": fold_number,
                "process_feature": feature,
                "inner_cv_MAE_improvement": value,
                "selected": feature in selected_process,
            })
        for values in prediction_sets:
            for order, (actual, prediction) in enumerate(zip(values["y_true"], values["y_pred"])):
                ablation_oof.append({
                    "fold": fold_number,
                    "configuration": values["configuration"],
                    "row_index": int(outer_validation.index[order]),
                    "actual": float(actual),
                    "predicted": float(prediction),
                    "residual_pred_minus_actual": float(prediction - actual),
                })

        for family, candidates in groups.items():
            best_by_family, inner_log = model_family_search(
                outer_train,
                y_fit,
                FULL_FIELDS,
                {family: candidates},
                seed_offset=fold_number * 13,
            )
            estimator, params, scale = best_by_family[family]
            fitted = pipeline_for(estimator, FULL_FIELDS, scale=scale)
            fitted.fit(feature_data(outer_train, FULL_FIELDS), y_fit.to_numpy())
            predicted = fitted.predict(feature_data(outer_validation, FULL_FIELDS))
            score = metrics(y_validation.to_numpy(), predicted)
            model_fold_rows.append({
                "fold": fold_number,
                "model": family,
                "features": "all_17_inference_inputs",
                "inner_selected_parameters": json.dumps(params, sort_keys=True),
                **score,
            })
            model_selection_log.append({
                "scope": f"outer_fold_{fold_number}",
                "model": family,
                "inner_cv_MAE": inner_log[family]["inner_MAE"],
                "parameters": params,
            })
            for order, (actual, prediction) in enumerate(zip(y_validation, predicted)):
                nested_predictions.append({
                    "split": "outer_cv",
                    "fold": fold_number,
                    "model": family,
                    "row_index": int(outer_validation.index[order]),
                    "actual": float(actual),
                    "predicted": float(prediction),
                    "residual_pred_minus_actual": float(prediction - actual),
                    "absolute_error": float(abs(prediction - actual)),
                })

    ablation_frame = pd.DataFrame(ablation_fold_rows)
    ablation_frame.to_csv(OUT / "feature_group_ablation_results.csv", index=False)
    pd.DataFrame(ablation_selected).to_csv(
        OUT / "selected_process_features_by_outer_fold.csv", index=False
    )
    ablation_predictions = pd.DataFrame(ablation_oof)

    # Tune each family on the full training partition only; test is evaluated once.
    full_inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE + 900)
    final_search, final_search_logs = model_family_search(
        train,
        y_train,
        FULL_FIELDS,
        groups,
        seed_offset=900,
    )
    final_family = min(
        final_search_logs,
        key=lambda family: final_search_logs[family]["inner_MAE"],
    )
    final_estimator, final_params, final_scale = final_search[final_family]
    final_model = pipeline_for(final_estimator, FULL_FIELDS, scale=final_scale)
    final_model.fit(feature_data(train, FULL_FIELDS), y_train.to_numpy())
    best_test_prediction = final_model.predict(feature_data(test, FULL_FIELDS))

    test_ablation_models = {}
    for name, fields in {
        "customer_inputs_only_RF": CUSTOMER_FIELDS,
        "predicted_process_only_RF": PROCESS_FIELDS,
        "all_17_inference_inputs_RF": FULL_FIELDS,
    }.items():
        fitted = pipeline_for(rf_estimator(), fields)
        fitted.fit(feature_data(train, fields), y_train.to_numpy())
        test_ablation_models[name] = (fitted, fields)
    full_subset, full_marginal = select_process_subset(
        train,
        y_train,
        list(full_inner.split(train)),
    )
    subset_fields = CUSTOMER_FIELDS + full_subset
    subset_model = pipeline_for(rf_estimator(), subset_fields)
    subset_model.fit(feature_data(train, subset_fields), y_train.to_numpy())
    test_ablation_models["customer_plus_inner_selected_process_RF"] = (
        subset_model,
        subset_fields,
    )

    for family, (estimator, params, scale) in final_search.items():
        model = pipeline_for(estimator, FULL_FIELDS, scale=scale)
        model.fit(feature_data(train, FULL_FIELDS), y_train.to_numpy())
        prediction = model.predict(feature_data(test, FULL_FIELDS))
        test_score = metrics(y_test.to_numpy(), prediction)
        for order, (actual, estimate) in enumerate(zip(y_test, prediction)):
            nested_predictions.append({
                "split": "held_out_test",
                "fold": 0,
                "model": family,
                "row_index": int(test.index[order]),
                "actual": float(actual),
                "predicted": float(estimate),
                "residual_pred_minus_actual": float(estimate - actual),
                "absolute_error": float(abs(estimate - actual)),
            })
        model_fold_rows.append({
            "fold": "held_out_test",
            "model": family,
            "features": "all_17_inference_inputs",
            "inner_selected_parameters": json.dumps(params, sort_keys=True),
            **test_score,
        })

    test_ablation_rows = []
    for name, (fitted, fields) in test_ablation_models.items():
        prediction = fitted.predict(feature_data(test, fields))
        test_ablation_rows.append({
            "fold": "held_out_test",
            "configuration": name,
            "feature_count": len(fields),
            "selected_process_features": "|".join(full_subset) if "selected_process" in name else "",
            **metrics(y_test.to_numpy(), prediction),
        })
    for baseline in ("training_fold_mean", "training_fold_median"):
        center = float(y_train.mean() if baseline.endswith("mean") else y_train.median())
        test_ablation_rows.append({
            "fold": "held_out_test",
            "configuration": baseline,
            "feature_count": 0,
            "selected_process_features": "",
            **metrics(y_test.to_numpy(), np.full(len(test), center)),
        })
    pd.concat(
        [ablation_frame, pd.DataFrame(test_ablation_rows)],
        ignore_index=True,
    ).to_csv(OUT / "feature_group_ablation_results.csv", index=False)

    model_fold = pd.DataFrame(model_fold_rows)
    model_fold.to_csv(OUT / "fold_level_model_comparison.csv", index=False)
    candidate_cv = model_fold[model_fold["fold"] != "held_out_test"]
    candidate_aggregate = (
        candidate_cv.groupby("model", as_index=False)
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
    candidate_aggregate["full_training_inner_MAE"] = candidate_aggregate["model"].map(
        {key: value["inner_MAE"] for key, value in final_search_logs.items()}
    )
    candidate_aggregate["selected_on_outer_scores"] = False
    candidate_aggregate["family_selected_for_test"] = candidate_aggregate["model"].eq(final_family)

    ablation_aggregate = (
        ablation_frame.groupby("configuration", as_index=False)
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
    test_ablation_aggregate = pd.DataFrame(test_ablation_rows)
    ablation_aggregate.to_csv(OUT / "feature_group_ablation_aggregate.csv", index=False)
    candidate_aggregate.to_csv(OUT / "aggregate_model_comparison.csv", index=False)

    residual_records = []
    for values in ablation_oof:
        residual_records.append({
            "split": "outer_cv",
            "fold": values["fold"],
            "model": values["configuration"],
            "row_order_within_evaluation": int(values["row_index"]),
            "actual": values["actual"],
            "predicted": values["predicted"],
            "residual_pred_minus_actual": values["residual_pred_minus_actual"],
            "absolute_error": abs(values["residual_pred_minus_actual"]),
        })
    for family in groups:
        fitted = pipeline_for(
            final_search[family][0],
            FULL_FIELDS,
            scale=final_search[family][2],
        )
        fitted.fit(feature_data(train, FULL_FIELDS), y_train.to_numpy())
        pred = fitted.predict(feature_data(test, FULL_FIELDS))
        append_residual_rows(
            residual_records,
            "held_out_test",
            family,
            y_test.to_numpy(),
            pred,
            "saved_test",
        )
    for name, (fitted, fields) in test_ablation_models.items():
        pred = fitted.predict(feature_data(test, fields))
        append_residual_rows(
            residual_records,
            "held_out_test",
            name,
            y_test.to_numpy(),
            pred,
            "saved_test",
        )
    residual_frame = pd.DataFrame(residual_records)
    residual_frame.to_csv(OUT / "residual_diagnostics.csv", index=False)
    test_residual_frame = residual_frame[residual_frame["split"] == "held_out_test"].copy()
    selected_model_name = final_family
    selected_test = test_residual_frame[test_residual_frame["model"] == selected_model_name]
    residual_stats = {
        "model": selected_model_name,
        "count": len(selected_test),
        "mean": float(selected_test["residual_pred_minus_actual"].mean()),
        "median": float(selected_test["residual_pred_minus_actual"].median()),
        "std": float(selected_test["residual_pred_minus_actual"].std(ddof=1)),
        "q05": float(selected_test["residual_pred_minus_actual"].quantile(.05)),
        "q95": float(selected_test["residual_pred_minus_actual"].quantile(.95)),
        "absolute_error_q50": float(selected_test["absolute_error"].quantile(.5)),
        "absolute_error_q90": float(selected_test["absolute_error"].quantile(.9)),
        **metrics(y_test.to_numpy(), best_test_prediction),
    }
    subgroup = subgroup_errors(
        residual_frame,
        test,
        y_train,
        [selected_model_name, "all_17_inference_inputs_RF", "customer_plus_inner_selected_process_RF"],
    )
    subgroup.to_csv(OUT / "subgroup_error_analysis.csv", index=False)

    sensitivity = sensitivity_analysis(
        final_model,
        selected_model_name,
        FULL_FIELDS,
        test,
        train[CUSTOMER_FIELDS],
        process_models,
    )
    sensitivity.to_csv(OUT / "customer_input_sensitivity.csv", index=False)
    create_plots(
        frame,
        test,
        residual_frame[residual_frame["split"] == "outer_cv"],
        residual_frame,
        selected_model_name,
    )

    summary = target_summary(frame)
    component_correlation = (
        frame[[EFFICIENCY_TARGET, *TARGET_COMPONENTS]].corr()[EFFICIENCY_TARGET].to_dict()
    )
    exact_duplicates = []
    for column in frame.columns:
        if column == EFFICIENCY_TARGET or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if np.allclose(
            frame[column].to_numpy(dtype=float),
            frame[EFFICIENCY_TARGET].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ):
            exact_duplicates.append(column)
    target_series = frame[EFFICIENCY_TARGET].astype(float)
    target_diagnostics = {
        "rows": len(frame),
        "missing_target": int(target_series.isna().sum()),
        "unique_target_values": int(target_series.nunique()),
        "target_mean": float(target_series.mean()),
        "target_std": float(target_series.std(ddof=1)),
        "target_variance": float(target_series.var(ddof=1)),
        "target_median": float(target_series.median()),
        "target_min": float(target_series.min()),
        "target_max": float(target_series.max()),
        "target_q05": float(target_series.quantile(.05)),
        "target_q25": float(target_series.quantile(.25)),
        "target_q75": float(target_series.quantile(.75)),
        "target_q95": float(target_series.quantile(.95)),
        "performance_at_or_above_one": int((frame.Performance >= 1.0).sum()),
        "efficiency_at_or_above_100": int((target_series >= 100).sum()),
        "exact_target_duplicate_columns": exact_duplicates,
        "component_correlations_with_efficiency": component_correlation,
        "mean_baseline_RMSE": float(
            np.sqrt(np.mean((y_test.to_numpy() - y_train.mean()) ** 2))
        ),
        "median_baseline_RMSE": float(
            np.sqrt(np.mean((y_test.to_numpy() - y_train.median()) ** 2))
        ),
    }
    (OUT / "target_construction_diagnostics.json").write_text(
        json.dumps(target_diagnostics, indent=2), encoding="utf-8"
    )

    source_report = {
        "production_model2_feature_order": MODEL1_FEATURES + PROCESS_TARGETS,
        "customer_fields": CUSTOMER_FIELDS,
        "process_fields": PROCESS_FIELDS,
        "production_training_source": "training/model2_efficiency.py make_oof_process_features()",
        "production_inference_source": "training/preprocessing.py attach_predicted_process() and models/process/*.pkl",
        "candidate_bundle_source": "training/evaluation_results/controlled_stacking/train_model2_candidate.py and bundle_audit/candidate_inference.py",
        "revised_experiment_source": "run_controlled_experiment.py rebuilds upstream predictors on revised experiment data and uses cross-fitted process predictions; it does not directly use the saved production Model 1 inference bundle.",
        "primary_diagnostic_pipeline": "Frozen production Model 1 inference pipelines predict all nine process fields from customer features. These predicted outputs are then evaluated on identical saved 5-fold outer partitions and the saved held-out test partition.",
        "train_inference_assessment": [
            "Production training constructs 5-fold OOF Model 1 process predictions on the production training split; production inference appends predictions from Model 1 pipelines fit on that full training split.",
            "This OOF-to-full-fit change is an expected stack deployment distinction, but can cause distribution shift; this audit measures process prediction errors on the revised synthetic distribution.",
            "Production Model 2 ordinary CV applies downstream CV to process OOF columns built once over the full training set. For each downstream CV validation fold, upstream process models can have been trained on other downstream validation rows. Treat ordinary saved CV as potentially optimistic; use the controlled nested outer-CV evidence instead.",
            "The prior revised-dataset controlled runner uses in-experiment cross-fitted process predictors, not the exact frozen production inference bundle. It uses predictions rather than actual process targets, but is a different upstream model population. This audit's main results use actual saved inference-time Model 1 artifacts.",
            "The primary full-stack and ablation features contain no recorded process targets or post-production target components.",
        ],
    }
    (OUT / "data_lineage_and_target_diagnostics.json").write_text(
        json.dumps(source_report, indent=2), encoding="utf-8"
    )
    (OUT / "model_selection_log.json").write_text(
        json.dumps(
            {
                "seed": RANDOM_STATE,
                "outer_folds": OUTER_SPLITS,
                "inner_folds": INNER_SPLITS,
                "subset_selection_by_outer_fold": selected_feature_by_fold,
                "subset_inner_cv_marginal_improvement_final_train": full_marginal,
                "final_full_train_inner_cv_scores_and_parameters": final_search_logs,
                "final_family_selected_using_training_only_inner_cv": final_family,
                "final_parameters": final_params,
                "candidate_comparison_uses_outer_fold_scores_for_reporting_only": True,
                "outer_fold_scores_used_to_select_final_family": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    final_preservation = verify_preservation()
    if not final_preservation["verified"]:
        raise RuntimeError(f"Post-analysis preservation checks failed: {final_preservation['mismatches']}")
    (OUT / "production_preservation_verification.json").write_text(
        json.dumps(final_preservation, indent=2), encoding="utf-8"
    )
    manifest = {
        "experiment": "read_only_model2_diagnostic_audit_revised_synthetic_data",
        "scope": "diagnostics only; no production modifications or candidate promotion",
        "dataset_path": REVISED_DATASET.relative_to(ROOT).as_posix(),
        "dataset_sha256": sha256(REVISED_DATASET),
        "production_dataset_sha256": final_preservation["dataset_sha256_after"],
        "saved_split_path": SPLIT_INDICES_PATH.relative_to(ROOT).as_posix(),
        "split": {
            "train_rows": len(train),
            "test_rows": len(test),
            "outer_folds": OUTER_SPLITS,
            "outer_shuffle": True,
            "outer_random_state": RANDOM_STATE,
            "inner_folds": INNER_SPLITS,
            "test_touched_for_model_selection": False,
        },
        "feature_groups": {
            "customer_only": CUSTOMER_FIELDS,
            "predicted_process_only": PROCESS_FIELDS,
            "full_stack": FULL_FIELDS,
            "customer_plus_process_subset": "process columns selected by inner-fold positive marginal MAE improvement; details in selected_process_features_by_outer_fold.csv",
        },
        "candidate_models": {
            name: [params for _, params, _ in candidates]
            for name, candidates in groups.items()
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "final_training_only_selected_model": final_family,
        "final_training_only_selected_parameters": final_params,
        "heldout_metrics_and_residual_summary": residual_stats,
        "production_preservation_verified": final_preservation["verified"],
        "preservation_verified_file_count": final_preservation["verified_file_count"],
        "outputs": sorted(
            [*OUTPUT_CSVS, "feature_group_ablation_aggregate.csv",
             "selected_process_features_by_outer_fold.csv",
             "target_construction_diagnostics.json",
             "data_lineage_and_target_diagnostics.json",
             "model1_process_prediction_diagnostics.csv",
             "model_selection_log.json",
             "production_preservation_manifest.json",
             "production_preservation_verification.json",
             "model2_diagnostic_report.md",
             "README.md",
             "run_model2_diagnostic_audit.py",
             "verify_preservation.py",
             "efficiency_target_histogram.png",
             "efficiency_vs_oee_components.png",
             "outer_cv_actual_vs_predicted_full_stack_rf.png",
             "heldout_residual_diagnostics.png",
             "heldout_residual_by_fault.png"]
        ),
    }
    (OUT / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "completed",
        "output_directory": OUT.as_posix(),
        "dataset_sha256": manifest["dataset_sha256"],
        "selected_model": final_family,
        "heldout_metrics": {key: residual_stats[key] for key in ("MAE", "RMSE", "R2")},
        "preservation_verified": final_preservation["verified"],
    }, indent=2))


if __name__ == "__main__":
    main()
