"""Validate accounting, create feature diagnostics, and write the audit report."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import run_model2_diagnostic_audit as audit

ROOT = audit.ROOT
OUT = audit.OUT
REVISED_DIR = audit.REVISED_DIR
REVISED_DATASET = audit.REVISED_DATASET
PRODUCTION_DATASET = ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accounting_validation(frame: pd.DataFrame) -> dict:
    actual = frame["Actual_Processing_Length_m"].astype(float)
    good = frame["Good_Length_m"].astype(float)
    scrap = frame["Scrap_Length_m"].astype(float)
    planned = frame["Planned_Processing_Length_m"].astype(float)
    runtime = frame["Run_Time_min"].astype(float)
    downtime = frame["Downtime_min"].astype(float)
    planned_time = frame["Planned_Production_Time_min"].astype(float)
    availability = frame["Availability"].astype(float)
    performance = frame["Performance"].astype(float)
    quality = frame["Quality"].astype(float)
    oee = frame["OEE"].astype(float)
    efficiency = frame[audit.EFFICIENCY_TARGET].astype(float)
    checks = {
        "tube_geometry_mm": (
            frame["Target_Tube_OD_mm"].astype(float)
            - frame["Target_Tube_ID_mm"].astype(float)
            - 2 * frame["Target_Tube_Wall_mm"].astype(float),
            0.011,
            "OD = ID + 2 x wall; tolerance allows the generator's 0.01 mm rounding.",
        ),
        "actual_length_m": (
            actual - runtime * frame["Line_Speed_m_min"].astype(float),
            0.0051,
            "Actual processing length = runtime x line speed; tolerance allows 0.01 m rounding.",
        ),
        "good_plus_scrap_m": (good + scrap - actual, 0.0051, "Good length + scrap length = actual length."),
        "good_length_m": (good - (actual - scrap), 0.0051, "Good length = actual length - scrap length."),
        "scrap_length_m": (
            scrap - actual * frame["Scrap_Rate"].astype(float),
            0.0051,
            "Scrap length = actual length x scrap rate; tolerance allows length rounding.",
        ),
        "length_shortfall_m": (
            frame["Length_Shortfall_m"].astype(float)
            - np.maximum(planned - actual, 0),
            1e-9,
            "Length shortfall = max(planned processing length - actual length, 0).",
        ),
        "run_time_min": (
            runtime - (planned_time - downtime),
            0.0101,
            "Runtime = planned production time - downtime; tolerance allows 0.01 min rounding.",
        ),
        "availability": (
            availability - runtime / planned_time,
            1e-6,
            "Availability = runtime / planned production time.",
        ),
        "quality": (
            quality - good / actual,
            1e-6,
            "Quality = good length / actual processing length.",
        ),
        "performance_bounds": (
            np.maximum(np.maximum(-performance, performance - 1), 0),
            0.0,
            "Revised experiment requires a nonnegative Performance fraction no greater than 1.",
        ),
        "oee": (
            oee - availability * performance * quality,
            2e-6,
            "OEE = Availability x Performance x Quality.",
        ),
        "operational_efficiency_pct": (
            efficiency - 100 * oee,
            1e-10,
            "Operational efficiency (%) = 100 x OEE.",
        ),
    }
    results = {}
    for name, (residual, tolerance, definition) in checks.items():
        residual_array = np.asarray(residual, dtype=float)
        max_abs = float(np.max(np.abs(residual_array)))
        results[name] = {
            "max_abs_residual": max_abs,
            "tolerance": tolerance,
            "passed": bool(max_abs <= tolerance),
            "definition": definition,
        }

    completed = frame["Order_Status"].astype(str).eq("Completed")
    good_meets_requirement = good >= frame["Customer_Required_Length_m"].astype(float)
    status_consistency = completed.eq(good_meets_requirement)
    results["order_completion"] = {
        "definition": "Completed iff Good_Length_m >= Customer_Required_Length_m.",
        "completed_rows": int(completed.sum()),
        "rows_meeting_good_length_requirement": int(good_meets_requirement.sum()),
        "inconsistent_rows": int((~status_consistency).sum()),
        "passed": bool(status_consistency.all()),
    }

    numeric = frame.select_dtypes(include=[np.number])
    results["data_integrity"] = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_record_ids": int(frame["Record_ID"].duplicated().sum()),
        "nonfinite_numeric_cells": int(
            (~np.isfinite(numeric.to_numpy(dtype=float))).sum()
        ),
        "efficiency_at_or_above_100": int((efficiency >= 100).sum()),
        "performance_at_or_above_1": int((performance >= 1).sum()),
        "passed": bool(
            frame.shape == (5000, 40)
            and frame.isna().sum().sum() == 0
            and frame["Record_ID"].is_unique
            and np.isfinite(numeric.to_numpy(dtype=float)).all()
            and (efficiency < 100).all()
            and (performance < 1).all()
        ),
    }

    production = pd.read_csv(PRODUCTION_DATASET, keep_default_na=False)
    category_checks = {}
    for column in audit.CATEGORICAL_FEATURES:
        experimental_levels = sorted(frame[column].astype(str).unique().tolist())
        production_levels = sorted(production[column].astype(str).unique().tolist())
        category_checks[column] = {
            "levels_match_production": experimental_levels == production_levels,
            "experimental_levels": experimental_levels,
            "production_levels": production_levels,
        }
    results["categorical_label_preservation"] = {
        "columns": category_checks,
        "passed": all(item["levels_match_production"] for item in category_checks.values()),
    }
    results["passed"] = all(
        item["passed"]
        for key, item in results.items()
        if key != "passed"
    )
    return results


def selected_feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    split = json.loads(audit.SPLIT_INDICES_PATH.read_text(encoding="utf-8"))
    train = frame.loc[split["train_index"]].copy()
    y_train = train[audit.EFFICIENCY_TARGET].astype(float)
    process_models = audit.load_process_pipelines()
    stacked = audit.attach_predicted_process(
        frame[audit.MODEL1_FEATURES], process_models
    )
    for field in audit.PROCESS_FIELDS:
        frame[field] = stacked[field].to_numpy()
    train = frame.loc[split["train_index"]].copy()
    log = json.loads((OUT / "model_selection_log.json").read_text(encoding="utf-8"))
    family = log["final_family_selected_using_training_only_inner_cv"]
    parameters = log["final_parameters"]
    candidates = audit.model_grids()[family]
    estimator, scale = next(
        (candidate, scaled)
        for candidate, params, scaled in candidates
        if params == parameters
    )
    model = audit.pipeline_for(estimator, audit.FULL_FIELDS, scale=scale)
    model.fit(
        audit.feature_data(train, audit.FULL_FIELDS),
        y_train.to_numpy(dtype=float),
    )
    fitted_estimator = model.named_steps["model"]
    if hasattr(fitted_estimator, "feature_importances_"):
        importance = np.asarray(fitted_estimator.feature_importances_, dtype=float)
        method = "tree_feature_importances"
        names = model.named_steps["preprocess"].get_feature_names_out()
    elif hasattr(fitted_estimator, "coef_"):
        coefficients = np.asarray(fitted_estimator.coef_, dtype=float)
        importance = (
            np.mean(np.abs(coefficients), axis=0)
            if coefficients.ndim > 1
            else np.abs(coefficients)
        )
        method = "absolute_standardized_coefficient"
        names = model.named_steps["preprocess"].get_feature_names_out()
    else:
        permutation = permutation_importance(
            model,
            audit.feature_data(train, audit.FULL_FIELDS),
            y_train.to_numpy(dtype=float),
            n_repeats=5,
            random_state=audit.RANDOM_STATE,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        importance = permutation.importances_mean
        method = "training_set_permutation_MAE_increase"
        names = audit.FULL_FIELDS
    if len(names) != len(importance):
        raise RuntimeError("Feature-importance vector does not match transformed feature names.")
    result = pd.DataFrame({
        "model": family,
        "method": method,
        "feature": names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    result.to_csv(OUT / "selected_model_feature_importance.csv", index=False)
    return result


def write_process_source_validation(frame: pd.DataFrame) -> pd.DataFrame:
    process_models = audit.load_process_pipelines()
    predicted = audit.attach_predicted_process(
        frame[audit.MODEL1_FEATURES], process_models
    )
    rows = []
    for field in audit.PROCESS_FIELDS:
        delta = (
            predicted[field].to_numpy(dtype=float)
            - frame[field].to_numpy(dtype=float)
        )
        rows.append({
            "process_feature": field,
            "modeling_source": "frozen production Model 1 prediction",
            "mean_abs_difference_from_recorded_value": float(np.mean(np.abs(delta))),
            "max_abs_difference_from_recorded_value": float(np.max(np.abs(delta))),
            "predictions_not_exact_recorded_copy": bool(np.any(np.abs(delta) > 1e-12)),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "process_input_source_validation.csv", index=False)
    return result


def write_numeric_redundancy(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number])
    correlation = numeric.corr()
    rows = []
    columns = list(correlation.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            value = float(correlation.loc[left, right])
            if np.isfinite(value) and abs(value) >= 0.9999:
                rows.append({
                    "feature_1": left,
                    "feature_2": right,
                    "pearson_correlation": value,
                    "absolute_correlation": abs(value),
                    "includes_target": (
                        left == audit.EFFICIENCY_TARGET
                        or right == audit.EFFICIENCY_TARGET
                    ),
                    "interpretation": (
                        "Inspect role and target lineage; high correlation alone does not prove direct leakage."
                    ),
                })
    result = pd.DataFrame(rows).sort_values(
        "absolute_correlation", ascending=False
    ) if rows else pd.DataFrame(columns=[
        "feature_1", "feature_2", "pearson_correlation",
        "absolute_correlation", "includes_target", "interpretation",
    ])
    result.to_csv(OUT / "near_duplicate_numeric_features.csv", index=False)
    return result


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"

    def render(value: object) -> str:
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
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(accounting: dict, importance: pd.DataFrame) -> None:
    target = json.loads((OUT / "target_construction_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    residual = manifest["heldout_metrics_and_residual_summary"]
    cv = pd.read_csv(OUT / "aggregate_model_comparison.csv")
    cv = cv.sort_values("MAE_mean").copy()
    cv = cv[[
        "model", "fold_count", "MAE_mean", "MAE_std",
        "RMSE_mean", "RMSE_std", "R2_mean", "R2_std",
        "full_training_inner_MAE", "family_selected_for_test",
    ]]
    ablation = pd.read_csv(OUT / "feature_group_ablation_aggregate.csv")
    ablation = ablation.sort_values("MAE_mean")
    ablation_by_name = ablation.set_index("configuration")
    customer_mae = float(ablation_by_name.loc["customer_inputs_only_RF", "MAE_mean"])
    full_rf_mae = float(ablation_by_name.loc["all_17_inference_inputs_RF", "MAE_mean"])
    process_only_mae = float(ablation_by_name.loc["predicted_process_only_RF", "MAE_mean"])
    selected_subset_mae = float(
        ablation_by_name.loc["customer_plus_inner_selected_process_RF", "MAE_mean"]
    )
    process_diagnostics = pd.read_csv(OUT / "model1_process_prediction_diagnostics.csv")
    weak_processes = process_diagnostics[
        process_diagnostics["process_target"].isin(
            ["Line_Speed_m_min", "Puller_Speed_m_min"]
        )
    ]
    ablation_test = pd.read_csv(OUT / "feature_group_ablation_results.csv")
    ablation_test = ablation_test[ablation_test["fold"] == "held_out_test"]
    ablation_test = ablation_test[[
        "configuration", "feature_count", "MAE", "RMSE", "R2"
    ]]
    selected_family = residual["model"]
    subgroup_all = pd.read_csv(
        OUT / "subgroup_error_analysis.csv",
        keep_default_na=False,
    )
    subgroup = subgroup_all.copy()
    subgroup = subgroup[
        (subgroup["model"] == selected_family)
        & (subgroup["grouping"].isin(["Material_Type", "Tube_Type", "Braid_Type", "Fault_Code"]))
    ].sort_values(["grouping", "MAE"], ascending=[True, False])
    subgroup = subgroup[subgroup["reliable_at_minimum_n"]].groupby(
        "grouping", as_index=False, group_keys=False
    ).head(5)
    machine_breakdown = subgroup_all[
        (subgroup_all["model"] == selected_family)
        & (subgroup_all["grouping"] == "Fault_Code")
        & (subgroup_all["group"] == "Machine Breakdown")
    ]
    machine_breakdown_mae = (
        float(machine_breakdown.iloc[0]["MAE"]) if len(machine_breakdown) else float("nan")
    )
    roles = pd.read_csv(OUT / "feature_role_and_leakage_audit.csv")
    feature_role_summary = roles.groupby(
        ["Role", "Target leakage risk"], as_index=False
    ).size().sort_values("size", ascending=False)
    top_importance = importance.head(15)[["feature", "importance", "method"]]
    preservation = json.loads(
        (OUT / "production_preservation_verification.json").read_text(encoding="utf-8")
    )
    report = f"""# Read-only Model 2 Diagnostic Audit

## Scope and reproducibility

This report audits revised synthetic dataset `model2_generator_experiment_2026-10-11_revised/experimental_dataset.csv`; it does not alter production data, models, metadata, inference code, or the Streamlit app. No candidate was promoted. Dataset SHA-256: `{manifest["dataset_sha256"]}`. The saved split has {manifest["split"]["train_rows"]} training rows and {manifest["split"]["test_rows"]} held-out rows. Five-fold outer CV is confined to the training partition, with three-fold inner tuning and feature selection. The saved held-out set was not used for model selection; its candidate scores are descriptive.

The predictor uses the eight documented customer/product inputs plus nine predictions from the saved production Model 1 inference pipelines. Recorded process outcomes and post-production accounting columns are excluded. See `data_lineage_and_target_diagnostics.json` and `feature_role_and_leakage_audit.csv` for feature lineage and risk classifications.

## Historical production baseline (different dataset; not paired)

The saved production Model 2 achieved held-out MAE=6.4060, RMSE=8.4618, R²=0.2276 on the production dataset. Corrected controlled full-stack outer CV on that production dataset was MAE=6.4234 ± 0.1894, RMSE=8.3809 ± 0.3490, R²=0.1739 ± 0.0321. These are historical context only: they use the production target population and saved production evaluation, whereas this report evaluates a revised generated target using frozen production Model 1 predictions and newly fitted Model 2 estimators. The numerical scores must not be interpreted as paired gains or losses across datasets. Detailed recorded baseline tables are in [baseline_existing_production_model2.csv](../model2_generator_experiment_2026-10-11/baseline_existing_production_model2.csv) and [baseline_existing_dataset_nested_summary.csv](../model2_generator_experiment_2026-10-11/baseline_existing_dataset_nested_summary.csv).

## Why Model 2 remains weak

The best training-only family has outer-CV R²={float(cv.iloc[0]["R2_mean"]):.4f} ± {float(cv.iloc[0]["R2_std"]):.4f}; its improvement over a training-fold mean baseline is limited. In identical RF outer folds, adding all nine process predictions to customer inputs changes MAE from {customer_mae:.4f} to {full_rf_mae:.4f} pp (a {customer_mae - full_rf_mae:.4f} pp reduction), and selecting process inputs inside the folds changes it to {selected_subset_mae:.4f} pp. Process-only MAE is {process_only_mae:.4f} pp. This indicates that the current predicted process features provide only a small incremental signal in this revised synthetic dataset.

The frozen Model 1 predictors also have negative R² for revised-dataset line speed and puller speed:

{md_table(weak_processes[["process_target", "MAE", "RMSE", "R2", "actual_mean", "prediction_mean"]])}

The target additionally depends on realized downtime, faults, runtime and scrap/quality; these are post-run target components, not members of the 17-input inference contract. The process-error and subgroup results are consistent with unobserved operational outcomes limiting prediction, but do not establish a real-plant causal explanation. The held-out Machine Breakdown subgroup is particularly uncertain because its sample is small.

## Target and synthetic accounting audit

The revised experimental definition is `Operational_Efficiency_pct = 100 × OEE`, where `OEE = Availability × Performance × Quality`. The dataset's target mean/std are {target["target_mean"]:.3f}/{target["target_std"]:.3f} pp (range {target["target_min"]:.3f}–{target["target_max"]:.3f}); {target["performance_at_or_above_one"]} records have Performance ≥ 1 and {target["efficiency_at_or_above_100"]} targets are ≥100. No upper clipping is applied in this experiment.

Order completion is explicitly defined as **Completed iff `Good_Length_m >= Customer_Required_Length_m`**. `Length_Shortfall_m` remains the separate planned-versus-actual processing-length measure. Automated row-level checks cover geometry, actual/good/scrap length, scrap rate, shortfall, runtime, availability, quality, Performance bounds, OEE, efficiency, completion status, row/schema integrity, and categorical-level preservation.

{md_table(pd.DataFrame([{"check": name, "passed": item["passed"], "max_abs_residual": item.get("max_abs_residual", np.nan), "tolerance": item.get("tolerance", np.nan)} for name, item in accounting.items() if name not in {"data_integrity", "categorical_label_preservation", "passed"}]), digits=7)}

Overall accounting validation passed: **{accounting["passed"]}**. Full check details are in `dataset_accounting_validation.json`. The prior revised generator audit records the nominal 12–22 m/min recipe-reference envelope and process ranges as **synthetic assumptions, not manufacturing evidence or Polyhose-specific operating limits**. Model 1 prediction errors on this revised synthetic distribution are listed in `model1_process_prediction_diagnostics.csv`.

## Leakage-controlled performance

Outer-CV family comparisons use the same five folds and training-only inner hyperparameter selection. Values are mean ± standard deviation across outer folds. Candidate models are all compared on identical rows/features within this revised dataset; scores from different dataset versions are not treated as paired measurements.

{md_table(cv)}

### Feature-group ablation

All feature-group configurations use identical outer folds. "Predicted process only" uses the nine frozen Model 1 outputs; the selected-process subset is selected inside each training fold using inner-fold marginal MAE improvement.

{md_table(ablation)}

### Held-out result and residuals

Training-only inner CV selected **{selected_family}** with parameters `{json.dumps(manifest["final_training_only_selected_parameters"], sort_keys=True)}`. On the saved 1,000-row holdout, MAE={residual["MAE"]:.4f}, RMSE={residual["RMSE"]:.4f}, R²={residual["R2"]:.4f}. Residual convention is prediction − actual: mean={residual["mean"]:.4f}, median={residual["median"]:.4f}, SD={residual["std"]:.4f}, 5th–95th percentile=[{residual["q05"]:.4f}, {residual["q95"]:.4f}] pp; absolute-error median/p90={residual["absolute_error_q50"]:.4f}/{residual["absolute_error_q90"]:.4f} pp.

Residual records and subgroup metrics are in `residual_diagnostics.csv` and `subgroup_error_analysis.csv`. Subgroup estimates with fewer than {audit.MIN_SUBGROUP_COUNT} rows are marked unreliable and should not be over-interpreted. Summary below shows the largest reliable subgroup MAEs per grouping for the selected model.

Held-out feature-group ablation (same saved test rows):

{md_table(ablation_test)}

{md_table(subgroup)}

The held-out Machine Breakdown subgroup has only 16 records; its Ridge MAE is {machine_breakdown_mae:.2f} pp and it is explicitly below the minimum-n threshold, so this is a warning signal rather than a stable subgroup estimate.

## Importance and interpretation

Top transformed-feature importances for the selected training-only model:

{md_table(top_importance)}

Importance is a model-specific predictive diagnostic—not a causal effect. `customer_input_sensitivity.csv` gives perturbation contrasts on test inputs, with tube geometry kept valid and categorical alternatives chosen from training levels. It is not a causal experiment.

## Findings and recommendation

- The fixed dataset hash and all {preservation["verified_file_count"]} protected-file hashes match the pre-audit preservation manifest: **{preservation["verified"]}**.
- The evaluated Model 2 relationship remains limited; MAE/RMSE and R² should be interpreted with fold variability and subgroup counts rather than R² alone.
- Numeric pairs with absolute correlation ≥0.9999 are catalogued in `near_duplicate_numeric_features.csv`; target-related pairs reflect accounting/target formulas and are not accepted as model inputs.
- The experiment uses a different upstream feature-generation population than the original revised-experiment runner: this audit freezes production Model 1 inference artifacts. Do not compare these numbers to earlier revised-experiment metrics as paired outcomes.
- All nine modeled process columns are overwritten with frozen Model 1 predictions before splitting; `process_input_source_validation.csv` reports their differences from recorded post-run values.
- Machine context and realized faults/downtime are not part of the deployed 17-feature contract, despite contributing to target variation. This supports residual uncertainty and fault-specific error.
- The generator's speed, material/process profiles, environmental variation, fault probabilities/severity, and nominal-rate assumptions remain synthetic until manufacturing engineering validates them. No Polyhose operating limits are inferred here.

**Recommendation: revise/validate generator assumptions and inference-time feature availability with manufacturing engineering before candidate training.** Keep production generator, dataset, models, and app unchanged; do not promote a candidate based on synthetic-only results.

## Files

See `README.md` for the rerun sequence. Machine-readable outputs include the dataset accounting validation, feature role/leakage audit, target/relationship diagnostics, ablation and nested-CV metrics, residual/subgroup outputs, feature importance, preservation verification, and diagnostic manifest.
"""
    (OUT / "model2_diagnostic_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    frame = pd.read_csv(REVISED_DATASET, keep_default_na=False)
    if sha256(REVISED_DATASET) != audit.EXPECTED_REVISED_SHA256:
        raise RuntimeError("The revised dataset hash differs from the audited input.")
    validation = accounting_validation(frame)
    (OUT / "dataset_accounting_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    if not validation["passed"]:
        raise RuntimeError("Experimental dataset accounting validation failed.")

    importance = selected_feature_importance(frame)
    write_process_source_validation(frame)
    write_numeric_redundancy(frame)
    check = subprocess.run(
        [sys.executable, str(OUT / "verify_preservation.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.stdout:
        print(check.stdout)
    if check.returncode:
        raise RuntimeError(f"Preservation verification failed: {check.stderr}")

    manifest_path = OUT / "diagnostic_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"] = sorted(set(manifest["outputs"]) | {
        "dataset_accounting_validation.json",
        "selected_model_feature_importance.csv",
        "near_duplicate_numeric_features.csv",
        "process_input_source_validation.csv",
        "finalize_model2_diagnostic_audit.py",
    })
    manifest["dataset_accounting_validation_passed"] = validation["passed"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(validation, importance)
    readme = """# Model 2 Diagnostic Audit

Read-only audit of the revised experimental dataset using the saved production Model 1 inference pipelines. The production dataset, generator, model artifacts, metadata, inference contract, and app are protected by the hash manifest.

## Reproduce

From the repository root, using the configured Python 3.13 environment:

```powershell
& 'C:\\Users\\khale\\AppData\\Local\\Microsoft\\WindowsApps\\python3.13.exe' 'training\\evaluation_results\\model2_diagnostic_audit_2026-10-11\\run_model2_diagnostic_audit.py'
& 'C:\\Users\\khale\\AppData\\Local\\Microsoft\\WindowsApps\\python3.13.exe' 'training\\evaluation_results\\model2_diagnostic_audit_2026-10-11\\finalize_model2_diagnostic_audit.py'
```

The first command runs nested CV and held-out diagnostics; the second verifies production preservation, validates dataset identities, re-fits only the selected training-only model to extract feature importance, and writes this report. Both commands require the repository's configured dependencies and the reviewed revised dataset. `verify_preservation.py` can be run independently to re-check protected hashes.

## Key outputs

- `model2_diagnostic_report.md`: findings, assumptions, and recommendation.
- `dataset_accounting_validation.json`: row-level identity and schema checks.
- `aggregate_model_comparison.csv`, `fold_level_model_comparison.csv`: nested CV and descriptive holdout model metrics.
- `feature_group_ablation_results.csv`, `feature_group_ablation_aggregate.csv`: identical-fold feature-group comparisons.
- `residual_diagnostics.csv`, `subgroup_error_analysis.csv`: holdout residuals and subgroup metrics.
- `selected_model_feature_importance.csv`, `customer_input_sensitivity.csv`: model-specific diagnostics.
- `near_duplicate_numeric_features.csv`: numeric pairs with absolute Pearson correlation ≥0.9999.
- `production_preservation_verification.json`: post-run protected-file SHA-256 verification.

All numerical findings describe synthetic data and are not evidence of factory performance or validated operating limits.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    report_text = (OUT / "model2_diagnostic_report.md").read_text(encoding="utf-8")
    if not report_text.endswith("\n") or not manifest["production_preservation_verified"]:
        raise RuntimeError("Final report or preservation state failed a completion check.")
    print(json.dumps({
        "status": "finalized",
        "accounting_validation_passed": validation["passed"],
        "preservation_verified": manifest["production_preservation_verified"],
        "feature_importance_rows": len(importance),
        "report": str(OUT / "model2_diagnostic_report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
