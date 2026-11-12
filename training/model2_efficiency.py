"""MODEL 2 — Operational efficiency prediction.

Predicts Operational_Efficiency_pct from customer/product requirements plus
Model 1 *predicted* process parameters. Post-production OEE components are
never used as features.

Requires Model 1 pipelines in models/process/.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (  # noqa: E402
    DISCLAIMER,
    EFFICIENCY_MODELS_DIR,
    EFFICIENCY_TARGET,
    CV_FOLDS,
    LGBM_PARAMS,
    METADATA_PATH,
    MODEL1_FEATURES,
    PLANNING_CONTEXT_EXCLUDED,
    POST_PRODUCTION_LEAKAGE,
    PROCESS_TARGETS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_PARAMS,
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
    make_stacked_pipeline,
    make_regressor_pipeline,
    merge_metadata,
    require_columns,
    transformed_feature_names,
)

ALGORITHMS = {
    "Random Forest": (RandomForestRegressor, RF_PARAMS),
    "XGBoost": (XGBRegressor, XGB_PARAMS),
    "LightGBM": (LGBMRegressor, LGBM_PARAMS),
}

# Customer inputs plus Model 1 process-parameter predictions.
STACKED_FEATURES = MODEL1_FEATURES + PROCESS_TARGETS


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
    print("LEAKAGE CHECK — MODEL 2  (Operational Efficiency)")
    print("=" * 78)
    print(f"\nTarget:\n  - {EFFICIENCY_TARGET}")

    print("\nFeatures used:")
    print("  Customer / product requirements:")
    for f in MODEL1_FEATURES:
        print(f"    - {f}")
    print("  Model 1 predicted process parameters (not the recorded setpoints):")
    for f in PROCESS_TARGETS:
        print(f"    - {f}  [predicted]")

    print("\nFeatures excluded because they leak the efficiency / quality outcome")
    print("or are only known after the run:")
    for col in POST_PRODUCTION_LEAKAGE:
        mark = "  <-- TARGET" if col == EFFICIENCY_TARGET else ""
        print(f"    - {col}{mark}")

    print("\nFeatures excluded as identifiers / plant context not in the planned UI:")
    for col in PLANNING_CONTEXT_EXCLUDED:
        print(f"    - {col}")

    print("\nWhy predicted process parameters instead of actual recorded ones:")
    print("  The application will only have Model 1 outputs at planning time.")
    print("  Using recorded setpoints would overstate accuracy relative to that pipeline.")
    print("  Model 2 training uses five-fold out-of-fold Model 1 predictions;")
    print("  test-set process predictions come from Model 1 fit on the full train split.")
    print("  Process predictions have target-specific uncertainty; consult Model 1")
    print("  held-out metrics and do not interpret them as validated machine setpoints.")
    print("\n" + DISCLAIMER)


def make_oof_process_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build stage-one process features without predicting rows seen in fitting."""
    customer_features = train_df[MODEL1_FEATURES].copy()
    oof_features = customer_features.copy()
    model1_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["model1"]
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


def train_compare(X_train, y_train, X_test, y_test):
    cv_scores = {}
    test_scores = {}
    fitted = {}
    for name, (cls, params) in ALGORITHMS.items():
        pipeline = make_stacked_pipeline(cls(**deepcopy(params)), PROCESS_TARGETS)
        print(f"\n  Cross-validating {name} on training data (5-fold)...")
        metrics_cv, _ = cv_metrics(pipeline, X_train, y_train)
        cv_scores[name] = metrics_cv
        print_metric_row(name, EFFICIENCY_TARGET, metrics_cv)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        pred_test = pipeline.predict(X_test)
        test_scores[name] = regression_metrics(y_test, pred_test)

    print_comparison_tables(EFFICIENCY_TARGET, cv_scores, "training CV")
    print_comparison_tables(EFFICIENCY_TARGET, test_scores, "held-out test (not used for selection)")

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
    print("MODEL 2 — OPERATIONAL EFFICIENCY")
    print("=" * 78)

    df = load_dataset()
    require_columns(df, MODEL1_FEATURES + [EFFICIENCY_TARGET], "Model 2")
    forbidden_features = set(POST_PRODUCTION_LEAKAGE)
    if set(STACKED_FEATURES) & forbidden_features:
        raise ValueError("Model 2 feature list contains a forbidden post-production field")
    print_leakage_report()

    print(f"\nTarget summary ({EFFICIENCY_TARGET}):")
    print(df[EFFICIENCY_TARGET].describe()[["min", "max", "mean", "std"]].round(3).to_string())

    pipelines = load_process_pipelines()
    train_df, test_df = apply_saved_split(df)
    print(f"[split] Reused Model 1 split: train={len(train_df)}  test={len(test_df)}")

    X_train = make_oof_process_features(train_df)
    X_test = attach_predicted_process(test_df, pipelines)
    y_train = train_df[EFFICIENCY_TARGET]
    y_test = test_df[EFFICIENCY_TARGET]

    print("\nStacked feature columns used for Model 2:")
    for c in STACKED_FEATURES:
        print(f"  - {c}")

    best_name, best_pipe, cv_scores, test_scores = train_compare(X_train, y_train, X_test, y_test)

    if max(metrics["R2"] for metrics in test_scores.values()) > 0.90:
        raise RuntimeError(
            "A Model 2 test R2 exceeded 0.90. Audit for synthetic leakage before saving artifacts."
        )

    if cv_scores[best_name]["R2"] < 0:
        print("\nSTOP: Model 2 training-CV R2 is negative.")
        print("Customer inputs + Model 1 process predictions do not explain efficiency")
        print("in this table. The model will still be saved as an artifact, but it is")
        print("not a scientifically useful efficiency predictor.")

    pred_test = best_pipe.predict(X_test)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for algo, m in cv_scores.items():
        rows.append({"algorithm": algo, "split": "cv", **m})
    for algo, m in test_scores.items():
        rows.append({"algorithm": algo, "split": "test", **m})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "model2_metrics.csv", index=False)

    plot_algorithm_comparison(cv_scores, "Model 2 — training CV comparison", "model2_cv_comparison.png")
    plot_algorithm_comparison(test_scores, "Model 2 — test comparison", "model2_test_comparison.png")
    plot_actual_vs_predicted(
        y_test,
        pred_test,
        f"Model 2 actual vs predicted ({best_name})",
        "model2_actual_vs_predicted.png",
    )
    plot_residuals(
        y_test,
        pred_test,
        f"Model 2 residuals ({best_name})",
        "model2_residuals.png",
    )
    names = transformed_feature_names(best_pipe)
    importances = best_pipe.named_steps["model"].feature_importances_
    plot_feature_importance(
        names,
        importances,
        f"Model 2 feature importance ({best_name})",
        "model2_feature_importance.png",
    )
    print("\nWhich factors most influence predicted operational efficiency?")
    order = np.argsort(importances)[::-1]
    for i in order[:12]:
        print(f"  {names[i]:40} {importances[i]:.4f}")

    EFFICIENCY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = EFFICIENCY_MODELS_DIR / "efficiency_model.pkl"
    joblib.dump(best_pipe, model_path)
    print(f"\nSaved pipeline -> {_rel(model_path)}")

    merge_metadata(
        "model2",
        {
            "name": "Operational efficiency prediction",
            "target": EFFICIENCY_TARGET,
            "features": STACKED_FEATURES,
            "process_features_are_model1_predictions": True,
            "training_process_features": "5-fold out-of-fold Model 1 predictions",
            "test_process_features": "Model 1 pipelines fit on the full 80% training split",
            "excluded_post_production": POST_PRODUCTION_LEAKAGE,
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
    print("MODEL 2 SUMMARY")
    print("=" * 78)
    print(f"BEST MODEL: {best_name}")
    print(f"Test MAE={test_scores[best_name]['MAE']:.4f}  "
          f"RMSE={test_scores[best_name]['RMSE']:.4f}  "
          f"R2={test_scores[best_name]['R2']:.4f}")
    print(DISCLAIMER)
    print("MODEL 2 COMPLETE")


if __name__ == "__main__":
    main()
