"""MODEL 1 — Process parameter prediction.

Predicts internal extrusion setpoints from customer/product requirements.

This script trains Random Forest, XGBoost, and LightGBM independently for
each process-parameter target, compares them with cross-validation on the
training split, then evaluates the held-out test set once.

Models are synthetic-data proofs of concept, not representations of real
Polyhose equipment.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    DISCLAIMER,
    LGBM_PARAMS,
    METADATA_PATH,
    MODEL1_FEATURES,
    MODELS_DIR,
    PLANNING_CONTEXT_EXCLUDED,
    POST_PRODUCTION_LEAKAGE,
    PROCESS_MODEL_FILENAMES,
    PROCESS_MODELS_DIR,
    PROCESS_TARGETS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_PARAMS,
    XGB_PARAMS,
)
from evaluate import (  # noqa: E402
    cv_metrics,
    plot_actual_vs_predicted_grid,
    plot_feature_importance,
    plot_metric_bars,
    plot_residuals,
    print_comparison_tables,
    print_metric_row,
    print_wide_r2_table,
    regression_metrics,
    select_best_algorithm,
)
from preprocessing import (  # noqa: E402
    customer_xy,
    load_dataset,
    make_regressor_pipeline,
    require_columns,
    split_frame,
    transformed_feature_names,
)

ALGORITHMS = {
    "Random Forest": (RandomForestRegressor, RF_PARAMS),
    "XGBoost": (XGBRegressor, XGB_PARAMS),
    "LightGBM": (LGBMRegressor, LGBM_PARAMS),
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def audit_dataset(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("DATASET AUDIT (Model 1)")
    print("=" * 78)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print("\nColumn list:")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:2}. {col}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    if missing.empty:
        print("  None.")
    else:
        print(missing.to_string())
        raise ValueError("Missing values present. Training stopped; values will not be invented.")

    print(f"\nDuplicate rows: {int(df.duplicated().sum())}")

    print("\nCategorical value counts (customer inputs):")
    for col in CATEGORICAL_FEATURES:
        print(f"\n  {col} ({df[col].nunique()} levels)")
        print(df[col].value_counts().to_string().replace("\n", "\n    "))

    print("\nCustomer numeric summary:")
    print(df[CUSTOMER_NUMERIC_FEATURES].describe().T[["min", "max", "mean", "std"]].round(3).to_string())

    print("\nProcess-target summary:")
    print(df[PROCESS_TARGETS].describe().T[["min", "max", "mean", "std"]].round(3).to_string())


def print_leakage_report() -> None:
    print("\n" + "=" * 78)
    print("LEAKAGE CHECK — MODEL 1  (Process Parameter Prediction)")
    print("=" * 78)
    print("\nTarget(s):")
    for t in PROCESS_TARGETS:
        print(f"  - {t}")

    print("\nFeatures used (customer / product requirements only):")
    for f in MODEL1_FEATURES:
        print(f"  - {f}")

    print("\nFeatures excluded because they ARE the Model 1 targets")
    print("(using them as inputs would leak the quantity being predicted,")
    print("and the planned UI will not ask operators for these setpoints):")
    for t in PROCESS_TARGETS:
        print(f"  - {t}")

    print("\nFeatures excluded because they are post-production outcomes")
    print("(unknown when process setpoints are chosen; would leak quality/OEE):")
    for col in POST_PRODUCTION_LEAKAGE:
        print(f"  - {col}")

    print("\nFeatures excluded because they are identifiers or plant-context")
    print("fields not in the planned customer-requirement UI:")
    for col in PLANNING_CONTEXT_EXCLUDED:
        print(f"  - {col}")

    print("\nNote on Target_Tube_OD / Wall / ID:")
    print("  These are geometrically related (ID + 2*Wall ≈ OD). All three are")
    print("  kept because the specification lists them as customer inputs.")
    print("  This is redundancy, not target leakage.")
    print("\n" + DISCLAIMER)


def new_estimators() -> dict:
    return {
        name: cls(**deepcopy(params))
        for name, (cls, params) in ALGORITHMS.items()
    }


def train_one_target(target: str, X_train, y_train, X_test, y_test) -> dict:
    print("\n" + "-" * 78)
    print(f"TARGET: {target}")
    print("-" * 78)

    cv_scores = {}
    test_scores = {}
    fitted = {}

    for name, estimator in new_estimators().items():
        pipeline = make_regressor_pipeline(estimator)
        print(f"\n  Cross-validating {name} on training data ({5}-fold)...")
        metrics_cv, _ = cv_metrics(pipeline, X_train, y_train[target])
        cv_scores[name] = metrics_cv
        print_metric_row(name, target, metrics_cv)

        # Fit on the full training split after CV. Test data is not used here.
        pipeline.fit(X_train, y_train[target])
        fitted[name] = pipeline
        pred_test = pipeline.predict(X_test)
        metrics_test = regression_metrics(y_test[target], pred_test)
        test_scores[name] = metrics_test

    print_comparison_tables(target, cv_scores, "training CV")
    print_comparison_tables(target, test_scores, "held-out test (reported, not used for selection)")

    best_name = select_best_algorithm(cv_scores)
    best_pipe = fitted[best_name]
    print(
        f"\n  SELECTED for {target}: {best_name}  "
        f"(lowest training-CV MAE, then RMSE, then R²)"
    )
    print(f"  CV    {format_inline(cv_scores[best_name])}")
    print(f"  TEST  {format_inline(test_scores[best_name])}")

    pred_test = best_pipe.predict(X_test)
    names = transformed_feature_names(best_pipe)
    importances = best_pipe.named_steps["model"].feature_importances_

    return {
        "target": target,
        "best_algorithm": best_name,
        "pipeline": best_pipe,
        "cv": cv_scores,
        "test": test_scores,
        "y_test": np.asarray(y_test[target]),
        "pred_test": np.asarray(pred_test),
        "feature_names": names,
        "importances": importances,
    }


def format_inline(m: dict) -> str:
    return f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}"


def save_selected(result: dict) -> str:
    PROCESS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = PROCESS_MODEL_FILENAMES[result["target"]]
    path = PROCESS_MODELS_DIR / filename
    joblib.dump(result["pipeline"], path)
    print(f"  Saved pipeline -> {_rel(path)}")
    return _rel(path)


def write_metadata(df: pd.DataFrame, results: list[dict], saved_paths: dict[str, str]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "disclaimer": DISCLAIMER,
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": list(df.columns),
        "random_state": RANDOM_STATE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model1": {
            "name": "Process parameter prediction",
            "features": MODEL1_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": CUSTOMER_NUMERIC_FEATURES,
            "excluded_post_production": POST_PRODUCTION_LEAKAGE,
            "excluded_planning_context": PLANNING_CONTEXT_EXCLUDED,
            "targets": {},
        },
    }
    for result in results:
        target = result["target"]
        payload["model1"]["targets"][target] = {
            "selected_algorithm": result["best_algorithm"],
            "file": saved_paths[target],
            "cv_metrics": result["cv"][result["best_algorithm"]],
            "test_metrics": result["test"][result["best_algorithm"]],
            "all_algorithms_cv": result["cv"],
            "all_algorithms_test": result["test"],
        }
    notes = []
    for result in results:
        best = result["best_algorithm"]
        r2 = result["cv"][best]["R2"]
        if r2 < 0:
            notes.append(
                f"{result['target']}: training-CV R2={r2:.4f} (negative). "
                "Customer/product features do not explain this target better than "
                "predicting the mean. The saved pipeline is retained for a complete "
                "artifact set, but it is not a scientifically useful mapping and "
                "must not be presented as a validated process model."
            )
        elif r2 < 0.5:
            notes.append(
                f"{result['target']}: training-CV R2={r2:.4f} is weak. "
                "Treat predictions as uncertain."
            )
    payload["model1"]["scientific_notes"] = notes or [
        "No target failed the R2 sanity check."
    ]
    METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[meta] Wrote {_rel(METADATA_PATH)}")
    if notes:
        print("\nSCIENTIFIC CAVEATS")
        for note in notes:
            print(f"  - {note}")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main() -> None:
    _configure_stdio()
    print(DISCLAIMER)
    df = load_dataset()
    require_columns(df, MODEL1_FEATURES + PROCESS_TARGETS, "Model 1")
    audit_dataset(df)
    print_leakage_report()

    X, y = customer_xy(df, PROCESS_TARGETS)
    record_ids = df["Record_ID"] if "Record_ID" in df.columns else None
    X_train, X_test, y_train, y_test = split_frame(X, y, record_ids=record_ids)

    cv_by_target = {}
    test_by_target = {}
    results = []
    saved_paths = {}

    for target in PROCESS_TARGETS:
        result = train_one_target(target, X_train, y_train, X_test, y_test)
        saved_paths[target] = save_selected(result)
        cv_by_target[target] = result["cv"]
        test_by_target[target] = result["test"]
        results.append(result)

    print_wide_r2_table(cv_by_target, "R2", "training CV")
    print_wide_r2_table(cv_by_target, "MAE", "training CV")
    print_wide_r2_table(cv_by_target, "RMSE", "training CV")
    print_wide_r2_table(test_by_target, "R2", "held-out test")
    print_wide_r2_table(test_by_target, "MAE", "held-out test")
    print_wide_r2_table(test_by_target, "RMSE", "held-out test")

    print("\n" + "=" * 78)
    print("SELECTED ALGORITHM PER PROCESS PARAMETER (training-CV selection)")
    print("=" * 78)
    print(f"{'Target':32} {'Best algorithm':16} {'CV MAE':>10} {'CV R²':>10} {'Test MAE':>10} {'Test R²':>10}")
    print("-" * 90)
    for result in results:
        t = result["target"]
        best = result["best_algorithm"]
        cv = result["cv"][best]
        te = result["test"][best]
        print(f"{t:32} {best:16} {cv['MAE']:10.4f} {cv['R2']:10.4f} {te['MAE']:10.4f} {te['R2']:10.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        for algo, m in result["cv"].items():
            rows.append({"target": result["target"], "algorithm": algo, "split": "cv", **m})
        for algo, m in result["test"].items():
            rows.append({"target": result["target"], "algorithm": algo, "split": "test", **m})
    metrics_csv = RESULTS_DIR / "model1_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    print(f"\n[table] {_rel(metrics_csv)}")

    plot_metric_bars(cv_by_target, "R2", "Model 1 — training CV R² by target", "model1_cv_r2.png")
    plot_metric_bars(cv_by_target, "MAE", "Model 1 — training CV MAE by target", "model1_cv_mae.png")
    plot_metric_bars(test_by_target, "R2", "Model 1 — test R² by target", "model1_test_r2.png")
    plot_metric_bars(test_by_target, "MAE", "Model 1 — test MAE by target", "model1_test_mae.png")

    panels = []
    for result in results:
        short = PROCESS_MODEL_FILENAMES[result["target"]].replace("_model.pkl", "")
        panels.append((f"{short}\n{result['best_algorithm']}", result["y_test"], result["pred_test"]))
        plot_residuals(
            result["y_test"],
            result["pred_test"],
            f"Residuals — {result['target']} ({result['best_algorithm']})",
            f"model1_residuals_{short}.png",
        )
        plot_feature_importance(
            result["feature_names"],
            result["importances"],
            f"Feature importance — {result['target']} ({result['best_algorithm']})",
            f"model1_importance_{short}.png",
        )
    plot_actual_vs_predicted_grid(
        panels,
        "model1_actual_vs_predicted.png",
        "Model 1 — actual vs predicted (selected algorithm, test set)",
    )

    # One combined importance view: mean rank across selected process models
    # helps answer "which customer factors drive process setpoints?"
    all_names = results[0]["feature_names"]
    stacked = np.vstack(
        [r["importances"] / (r["importances"].sum() + 1e-12) for r in results]
    )
    mean_imp = stacked.mean(axis=0)
    plot_feature_importance(
        all_names,
        mean_imp,
        "Mean normalized importance across selected Model 1 targets",
        "model1_importance_mean.png",
    )
    print("\nCustomer/product factors with highest mean importance for process setpoints:")
    order = np.argsort(mean_imp)[::-1]
    for i in order[:10]:
        print(f"  {all_names[i]:30} {mean_imp[i]:.4f}")

    write_metadata(df, results, saved_paths)

    print("\n" + "=" * 78)
    print("MODEL 1 SUMMARY")
    print("=" * 78)
    print("Task          : predict 9 process parameters from 8 customer/product inputs")
    print("Algorithms    : Random Forest, XGBoost, LightGBM (compared per target)")
    print("Selection     : lowest training-CV MAE, then RMSE, then R²")
    print("Saved models  : models/process/*.pkl  (full sklearn Pipelines)")
    print("Do not train Model 2/3 until this summary looks scientifically sound.")
    print(DISCLAIMER)
    print("MODEL 1 COMPLETE")


if __name__ == "__main__":
    main()
