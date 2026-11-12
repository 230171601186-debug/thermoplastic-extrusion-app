"""MODEL 3 — Quality / scrap-rate prediction.

Predicts Scrap_Rate from customer/product requirements plus Model 1 predicted
process parameters (conditions known before the quality outcome).

Quality is not trained as a second target; at application time it can be
derived as Quality ≈ 1 - Scrap_Rate when both are stored as fractions.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (  # noqa: E402
    CV_FOLDS,
    DISCLAIMER,
    LGBM_PARAMS,
    METADATA_PATH,
    MODEL1_FEATURES,
    PLANNING_CONTEXT_EXCLUDED,
    PROCESS_TARGETS,
    PROJECT_ROOT,
    QUALITY_MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_PARAMS,
    SCRAP_TARGET,
    XGB_PARAMS,
)
from evaluate import (  # noqa: E402
    cv_metrics,
    plot_actual_vs_predicted,
    plot_algorithm_comparison,
    plot_feature_importance,
    plot_residuals,
    print_comparison_tables,
    print_metric_row,
    regression_metrics,
    select_best_algorithm,
)
from preprocessing import (  # noqa: E402
    apply_saved_split,
    attach_predicted_process,
    load_dataset,
    load_process_pipelines,
    make_regressor_pipeline,
    make_stacked_pipeline,
    merge_metadata,
    require_columns,
    transformed_feature_names,
)

ALGORITHMS = {
    "Random Forest": (RandomForestRegressor, RF_PARAMS),
    "XGBoost": (XGBRegressor, XGB_PARAMS),
    "LightGBM": (LGBMRegressor, LGBM_PARAMS),
}

STACKED_FEATURES = MODEL1_FEATURES + PROCESS_TARGETS

MODEL3_EXCLUDED = [
    "Scrap_Rate",
    "Quality",
    "Good_Length_m",
    "Scrap_Length_m",
    "Operational_Efficiency_pct",
    "OEE",
    "Availability",
    "Performance",
    "Actual_Processing_Length_m",
    "Length_Shortfall_m",
    "Order_Status",
    "Fault_Code",
    "Power_Consumption_kWh",
    "Downtime_min",
    "Run_Time_min",
]


def make_oof_process_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe Model 1 features for Model 3 training rows.

    Each training row receives process predictions from a Model 1 pipeline
    that did not fit on that row. The held-out test rows continue to use
    Model 1 pipelines fitted on the complete training split.
    """
    customer_features = train_df[MODEL1_FEATURES].copy()
    oof_features = customer_features.copy()
    model1_metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )["model1"]
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for target in PROCESS_TARGETS:
        algorithm = model1_metadata["targets"][target]["selected_algorithm"]
        estimator_class, params = ALGORITHMS[algorithm]
        process_pipeline = make_regressor_pipeline(estimator_class(**deepcopy(params)))
        oof_features[target] = cross_val_predict(
            process_pipeline,
            customer_features,
            train_df[target],
            cv=cv,
            n_jobs=-1,
        )
    return oof_features


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def print_leakage_report() -> None:
    print("\n" + "=" * 78)
    print("LEAKAGE CHECK — MODEL 3  (Quality / Scrap Rate)")
    print("=" * 78)
    print(f"\nTarget:\n  - {SCRAP_TARGET}")
    print("  Quality is not a training target. If scrap rate is a fraction,")
    print("  the app may report Quality ≈ 1 - Scrap_Rate.")

    print("\nFeatures used:")
    print("  Customer / product requirements:")
    for f in MODEL1_FEATURES:
        print(f"    - {f}")
    print("  Model 1 predicted process parameters (before the quality outcome):")
    for f in PROCESS_TARGETS:
        print(f"    - {f}  [predicted]")

    print("\nFeatures excluded because they leak scrap / quality / efficiency:")
    for col in MODEL3_EXCLUDED:
        mark = "  <-- TARGET" if col == SCRAP_TARGET else ""
        print(f"    - {col}{mark}")

    print("\nAlso excluded (identifiers / plant context not in the planned UI):")
    for col in PLANNING_CONTEXT_EXCLUDED:
        print(f"    - {col}")

    print("\nRecorded process setpoints are not used. The deployed path is")
    print("customer inputs -> Model 1 predictions -> scrap-rate model.")
    print("\n" + DISCLAIMER)


def train_compare(X_train, y_train, X_test, y_test):
    cv_scores = {}
    test_scores = {}
    fitted = {}
    for name, (cls, params) in ALGORITHMS.items():
        pipeline = make_stacked_pipeline(cls(**deepcopy(params)), PROCESS_TARGETS)
        print(f"\n  Cross-validating {name} on training data (5-fold)...")
        metrics_cv, _ = cv_metrics(pipeline, X_train, y_train)
        cv_scores[name] = metrics_cv
        print_metric_row(name, SCRAP_TARGET, metrics_cv)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        pred_test = pipeline.predict(X_test)
        test_scores[name] = regression_metrics(y_test, pred_test)

    print_comparison_tables(SCRAP_TARGET, cv_scores, "training CV")
    print_comparison_tables(SCRAP_TARGET, test_scores, "held-out test (not used for selection)")

    best_name = select_best_algorithm(cv_scores)
    best_pipe = fitted[best_name]
    print(f"\nBEST MODEL: {best_name}")
    print("  Selection rule: lowest training-CV MAE, then RMSE, then R2.")
    print(
        f"  CV    MAE={cv_scores[best_name]['MAE']:.4f}  "
        f"RMSE={cv_scores[best_name]['RMSE']:.4f}  R2={cv_scores[best_name]['R2']:.4f}"
    )
    print(
        f"  TEST  MAE={test_scores[best_name]['MAE']:.4f}  "
        f"RMSE={test_scores[best_name]['RMSE']:.4f}  R2={test_scores[best_name]['R2']:.4f}"
    )
    return best_name, best_pipe, cv_scores, test_scores


def main() -> None:
    _configure_stdio()
    print(DISCLAIMER)
    print("=" * 78)
    print("MODEL 3 — QUALITY / SCRAP RATE")
    print("=" * 78)

    df = load_dataset()
    require_columns(df, MODEL1_FEATURES + [SCRAP_TARGET], "Model 3")
    print_leakage_report()

    print(f"\nTarget summary ({SCRAP_TARGET}):")
    print(df[SCRAP_TARGET].describe()[["min", "max", "mean", "std"]].round(4).to_string())
    if "Quality" in df.columns:
        approx = (1.0 - df[SCRAP_TARGET]) - df["Quality"]
        print(
            f"\nQuality vs 1 - Scrap_Rate: mean abs difference = {approx.abs().mean():.6f}"
        )
        print("If this is near zero, the app can derive Quality ≈ 1 - predicted Scrap_Rate.")

    pipelines = load_process_pipelines()
    train_df, test_df = apply_saved_split(df)
    print(f"[split] Reused Model 1 split: train={len(train_df)}  test={len(test_df)}")

    X_train = make_oof_process_features(train_df)
    X_test = attach_predicted_process(test_df, pipelines)
    y_train = train_df[SCRAP_TARGET]
    y_test = test_df[SCRAP_TARGET]

    best_name, best_pipe, cv_scores, test_scores = train_compare(X_train, y_train, X_test, y_test)

    if cv_scores[best_name]["R2"] < 0:
        print("\nSTOP: Model 3 training-CV R2 is negative.")
        print("Customer inputs + Model 1 process predictions do not explain scrap rate")
        print("in this table. The artifact is saved, but it is not a useful quality model.")

    pred_test = best_pipe.predict(X_test)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for algo, m in cv_scores.items():
        rows.append({"algorithm": algo, "split": "cv", **m})
    for algo, m in test_scores.items():
        rows.append({"algorithm": algo, "split": "test", **m})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "model3_metrics.csv", index=False)

    plot_algorithm_comparison(cv_scores, "Model 3 — training CV comparison", "model3_cv_comparison.png")
    plot_algorithm_comparison(test_scores, "Model 3 — test comparison", "model3_test_comparison.png")
    plot_actual_vs_predicted(
        y_test,
        pred_test,
        f"Model 3 scrap rate actual vs predicted ({best_name})",
        "model3_actual_vs_predicted.png",
    )
    plot_residuals(
        y_test,
        pred_test,
        f"Model 3 residuals ({best_name})",
        "model3_residuals.png",
    )
    names = transformed_feature_names(best_pipe)
    importances = best_pipe.named_steps["model"].feature_importances_
    plot_feature_importance(
        names,
        importances,
        f"Model 3 feature importance ({best_name})",
        "model3_feature_importance.png",
    )
    print("\nWhich factors most influence predicted scrap rate?")
    order = np.argsort(importances)[::-1]
    for i in order[:12]:
        print(f"  {names[i]:40} {importances[i]:.4f}")

    QUALITY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = QUALITY_MODELS_DIR / "scrap_rate_model.pkl"
    joblib.dump(best_pipe, model_path)
    print(f"\nSaved pipeline -> {_rel(model_path)}")

    merge_metadata(
        "model3",
        {
            "name": "Quality / scrap-rate prediction",
            "target": SCRAP_TARGET,
            "derived_quality_note": "If Scrap_Rate is a fraction, Quality ≈ 1 - Scrap_Rate",
            "features": STACKED_FEATURES,
            "process_features_are_model1_predictions": True,
            "excluded": MODEL3_EXCLUDED,
            "excluded_planning_context": PLANNING_CONTEXT_EXCLUDED,
            "selected_algorithm": best_name,
            "file": _rel(model_path),
            "cv_metrics": cv_scores[best_name],
            "test_metrics": test_scores[best_name],
            "all_algorithms_cv": cv_scores,
            "all_algorithms_test": test_scores,
        },
    )

    print("\n" + "=" * 78)
    print("MODEL 3 SUMMARY")
    print("=" * 78)
    print(f"BEST MODEL: {best_name}")
    print(f"Test MAE={test_scores[best_name]['MAE']:.4f}  "
          f"RMSE={test_scores[best_name]['RMSE']:.4f}  "
          f"R2={test_scores[best_name]['R2']:.4f}")
    print(DISCLAIMER)
    print("MODEL 3 COMPLETE")


if __name__ == "__main__":
    main()
