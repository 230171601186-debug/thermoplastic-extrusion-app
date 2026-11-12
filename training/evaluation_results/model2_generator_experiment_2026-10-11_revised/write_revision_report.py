"""Write the revised experiment comparison report from saved evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
PREVIOUS_DIR = EXPERIMENT_DIR.parent / "model2_generator_experiment_2026-10-11"


def markdown_table(frame: pd.DataFrame, precision: int = 4) -> str:
    view = frame.copy()
    for column in view.select_dtypes(include="number").columns:
        if column in {"fold_count", "count"}:
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )
        else:
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{precision}f}"
            )
    headers = [str(column) for column in view.columns]
    rows = [headers, ["---"] * len(headers)]
    rows.extend(
        [["" if pd.isna(value) else str(value) for value in row]
         for row in view.itertuples(index=False, name=None)]
    )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    revised_audit = json.loads(
        (EXPERIMENT_DIR / "experimental_dataset_audit.json").read_text(encoding="utf-8")
    )
    prior_audit = json.loads(
        (PREVIOUS_DIR / "experimental_dataset_audit.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (EXPERIMENT_DIR / "automated_validation_results.json").read_text(encoding="utf-8")
    )
    prior_metrics = pd.read_csv(PREVIOUS_DIR / "experimental_model2_metric_summary.csv")
    revised_metrics = pd.read_csv(EXPERIMENT_DIR / "experimental_model2_metric_summary.csv")
    metric_rows = []
    for dataset, metrics in (("existing_experiment_v1", prior_metrics), ("revised_experiment_v2", revised_metrics)):
        for _, row in metrics.iterrows():
            metric_rows.append({
                "dataset": dataset,
                "split": row["split"],
                "configuration": row["configuration"],
                "fold_count": int(row["fold_count"]),
                "MAE_mean": row["MAE_mean"],
                "MAE_std": row["MAE_std"],
                "RMSE_mean": row["RMSE_mean"],
                "RMSE_std": row["RMSE_std"],
                "R2_mean": row["R2_mean"],
                "R2_std": row["R2_std"],
                "cross_dataset_scores_are_paired": False,
            })
    model_comparison = pd.DataFrame(metric_rows)
    model_comparison.to_csv(EXPERIMENT_DIR / "prior_vs_revised_model_metrics.csv", index=False)

    distribution_rows = []
    for dataset, audit in (("existing_experiment_v1", prior_audit), ("revised_experiment_v2", revised_audit)):
        for target in ("Performance", "Operational_Efficiency_pct"):
            values = audit["target_summary"][target]
            distribution_rows.append({
                "dataset": dataset,
                "target": target,
                "mean": values["mean"],
                "std": values["std"],
                "min": values["min"],
                "q25": values["25%"],
                "median": values["50%"],
                "q75": values["75%"],
                "max": values["max"],
            })
    distributions = pd.DataFrame(distribution_rows)
    distributions.to_csv(EXPERIMENT_DIR / "prior_vs_revised_target_distributions.csv", index=False)

    residuals = pd.read_csv(EXPERIMENT_DIR / "experimental_model2_predictions_residuals.csv")
    full_cv = residuals[
        (residuals["split"] == "outer_cv")
        & (residuals["configuration"] == "full_stack_fixed_RF")
    ]
    residual_quartiles = (
        full_cv.groupby("target_range_training_quantiles", as_index=False)
        .agg(
            count=("residual_pred_minus_actual", "size"),
            residual_mean=("residual_pred_minus_actual", "mean"),
            residual_std=("residual_pred_minus_actual", "std"),
            residual_median=("residual_pred_minus_actual", "median"),
            MAE=("absolute_error", "mean"),
        )
    )
    residual_quartiles.to_csv(
        EXPERIMENT_DIR / "revised_outer_cv_residual_summary.csv", index=False
    )
    subgroups = pd.read_csv(EXPERIMENT_DIR / "test_generalization_by_group.csv")
    selected_subgroups = subgroups[
        (subgroups["configuration"] == "inner_selected_full_stack")
        & (subgroups["grouping"] == "Fault_Code")
    ].sort_values("MAE", ascending=False)
    selected_subgroups.head(8).to_csv(
        EXPERIMENT_DIR / "revised_worst_fault_subgroups.csv", index=False
    )

    prior_performance = prior_audit["target_summary"]["Performance"]
    revised_performance = revised_audit["target_summary"]["Performance"]
    prior_efficiency = prior_audit["target_summary"]["Operational_Efficiency_pct"]
    revised_efficiency = revised_audit["target_summary"]["Operational_Efficiency_pct"]
    selected = revised_metrics[
        (revised_metrics["split"] == "nested_outer_cv")
        & (revised_metrics["configuration"] == "inner_selected_full_stack")
    ].iloc[0]
    fixed = revised_metrics[
        (revised_metrics["split"] == "nested_outer_cv")
        & (revised_metrics["configuration"] == "full_stack_fixed_RF")
    ].iloc[0]
    held_out = revised_metrics[
        (revised_metrics["split"] == "saved_test")
        & (revised_metrics["configuration"] == "inner_selected_full_stack")
    ].iloc[0]
    prior_selected = prior_metrics[
        (prior_metrics["split"] == "nested_outer_cv")
        & (prior_metrics["configuration"] == "inner_selected_full_stack")
    ].iloc[0]
    prior_held_out = prior_metrics[
        (prior_metrics["split"] == "saved_test")
        & (prior_metrics["configuration"] == "inner_selected_full_stack")
    ].iloc[0]
    test_residuals = residuals[
        (residuals["split"] == "held_out_test")
        & (residuals["configuration"] == "inner_selected_full_stack")
    ]["residual_pred_minus_actual"]

    lines = [
        "# Revised Isolated Model 2 Generator Experiment",
        "",
        "## Scope and preservation",
        "",
        "This revision is isolated in a new sibling directory. The production dataset, production generator, model artifacts, metadata, inference code, and Streamlit app were not changed, and no candidate was promoted.",
        "",
        f"Revised dataset SHA-256: `{validation['dataset_sha256']}`. The 5,000-row, 40-column schema, existing categorical labels, and fixed seed ({validation['fixed_seed']}) are preserved. Re-generation was exact.",
        "",
        "The existing experiment directory is retained unchanged. The same saved row-index train/test partition and leakage-controlled nested-stack procedure were used for this generated dataset; identical row positions do not make version-1 and version-2 rows paired observations.",
        "",
        "## Diagnosed and revised logic",
        "",
        "The prior generator centered realized line speed at nominal recipe speed with modest additive noise, then calculated `Performance` using `bounded(actual_rate / nominal_rate, 0, 1)`. This made values above nominal collapse to exactly 1.0; a high-throughput fault could also add speed after process generation. Those were the sources of the ceiling mass.",
        "",
        "The revised process now draws realized rate as a continuous beta-distributed fraction of the synthetic nominal recipe reference, with a small synthetic machine factor. High-throughput stress changes screw RPM but no longer injects a line-speed increase beyond the recipe reference. `Performance` is calculated as realized line rate / nominal recipe rate without target clipping; OEE and efficiency follow directly from availability, performance, and quality.",
        "",
        "The nominal 12–22 m/min recipe-reference envelope, beta distribution parameters, machine effects, inherited material/process profiles, fault probabilities, severity distributions, and all process bounds are **synthetic assumptions requiring manufacturing validation**. They are not Polyhose settings or validated operating limits. The nominal envelope was chosen only to keep the generated rate within the inherited experiment's synthetic line-speed range.",
        "",
        "Order completion now means `Good_Length_m >= Customer_Required_Length_m`. This is an explicit synthetic fulfillment definition; `Length_Shortfall_m` remains `max(Planned_Processing_Length_m - Actual_Processing_Length_m, 0)`, a separate process-planning measure. The revised data has "
        f"{validation['completed_orders']} completed records and {validation['completed_orders_below_required_good_length']} completed records below required good output.",
        "",
        "## Dataset-level changes (descriptive, not paired)",
        "",
        markdown_table(distributions),
        "",
        f"Performance's prior median/q75/max were {prior_performance['50%']:.4f}/{prior_performance['75%']:.4f}/{prior_performance['max']:.4f}; revised values are {revised_performance['50%']:.4f}/{revised_performance['75%']:.4f}/{revised_performance['max']:.4f}. No revised record is at or above 1.0.",
        "",
        f"Efficiency's prior median/max were {prior_efficiency['50%']:.2f}/{prior_efficiency['max']:.2f}%; revised median/max are {revised_efficiency['50%']:.2f}/{revised_efficiency['max']:.2f}%. The target distribution shifted downward as rate performance is now below nominal by construction. This is an intentional generator behavior change, not evidence of improved/worsened real factory efficiency.",
        "",
        "## Leakage-controlled Model 2 results",
        "",
        "Five-fold outer CV is reported as mean ± fold standard deviation. Candidate algorithm selection remains inside training folds; the saved test partition is not used to select the model.",
        "",
        markdown_table(model_comparison),
        "",
        f"Revised fixed-RF full-stack CV: MAE {fixed['MAE_mean']:.4f} ± {fixed['MAE_std']:.4f}, RMSE {fixed['RMSE_mean']:.4f} ± {fixed['RMSE_std']:.4f}, R² {fixed['R2_mean']:.4f} ± {fixed['R2_std']:.4f}.",
        "",
        f"Revised training-only selected full-stack CV: MAE {selected['MAE_mean']:.4f} ± {selected['MAE_std']:.4f}, RMSE {selected['RMSE_mean']:.4f} ± {selected['RMSE_std']:.4f}, R² {selected['R2_mean']:.4f} ± {selected['R2_std']:.4f}. Held-out test: MAE {held_out['MAE_mean']:.4f}, RMSE {held_out['RMSE_mean']:.4f}, R² {held_out['R2_mean']:.4f}. Test residual mean {test_residuals.mean():+.3f} pp; 5th/95th percentiles {test_residuals.quantile(.05):+.3f}/{test_residuals.quantile(.95):+.3f} pp.",
        "",
        f"Descriptively, the selected-model outer-CV scores moved from v1 to v2 by MAE {selected['MAE_mean'] - prior_selected['MAE_mean']:+.4f}, RMSE {selected['RMSE_mean'] - prior_selected['RMSE_mean']:+.4f}, and R² {selected['R2_mean'] - prior_selected['R2_mean']:+.4f}; held-out values moved by MAE {held_out['MAE_mean'] - prior_held_out['MAE_mean']:+.4f}, RMSE {held_out['RMSE_mean'] - prior_held_out['RMSE_mean']:+.4f}, and R² {held_out['R2_mean'] - prior_held_out['R2_mean']:+.4f}. Numerically these model scores are worse in v2, but because target populations differ these deltas are not paired estimates of a model regression.",
        "",
        "Version-1 and version-2 model metrics are measured on different generated target populations. Their numerical differences are descriptive only, not paired gains/losses or evidence of generalization to manufacturing.",
        "",
        "### Revised full-stack RF outer-CV residuals by training-derived target quartile",
        "",
        markdown_table(residual_quartiles),
        "",
        "### Highest-error revised held-out fault groups, selected full-stack model",
        "",
        markdown_table(selected_subgroups.head(8)[["value", "count", "MAE", "RMSE", "mean_residual"]]),
        "",
        "All material/tube/grade/braid/machine/shift/fault subgroup metrics are in `test_generalization_by_group.csv`; full prediction residuals and feature importances are also saved separately.",
        "",
        "## Accounting and automated validation",
        "",
        "The generator and independent validation both checked tube geometry; actual length = line speed × runtime; good length + scrap length = actual length; scrap length = actual length × scrap rate; shortfall definition; runtime = planned time − downtime; availability = runtime / planned time; performance = realized rate / reconstructed nominal rate; quality = good length / actual length; OEE = availability × performance × quality; efficiency = OEE × 100; and the good-output completion rule.",
        "",
        markdown_table(pd.DataFrame(
            [{"identity": identity, "max_abs_residual": residual}
             for identity, residual in validation["accounting_max_abs_residuals"].items()]
        )),
        "",
        f"Automated checks passed: fixed-seed regeneration, schema and categorical-label preservation, accounting, order completion ({validation['completed_orders_below_required_good_length']} inconsistent completed records), production dataset SHA-256, {validation['production_artifact_hashes_verified_unchanged']} production/model artifact hashes, and {validation['previous_experiment_files_verified_unchanged']} files in the prior experiment directory.",
        "",
        "## Recommendation",
        "",
        "The revised synthetic mechanics resolve the prior Performance ceiling and ambiguous completion flag at the data-generation level, while preserving accounting and schema. However, the new rate-loss distribution and nominal-speed assumptions are not factory-validated. Keep production unchanged and do not promote a candidate; obtain manufacturing review of the assumptions before considering a further generator or model-training phase.",
        "",
    ]
    report_text = "\n".join(lines)
    (EXPERIMENT_DIR / "revised_experiment_report.md").write_text(
        report_text, encoding="utf-8"
    )
    (EXPERIMENT_DIR / "experiment_report.md").write_text(
        report_text, encoding="utf-8"
    )
    manifest_path = EXPERIMENT_DIR / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment"] = "revised_isolated_model2_generator_and_nested_stacking"
    manifest["previous_experimental_dataset_sha256"] = (
        json.loads((PREVIOUS_DIR / "experiment_manifest.json").read_text(encoding="utf-8"))
        ["experimental_dataset_sha256"]
    )
    manifest["revised_completion_definition"] = (
        "Completed if and only if Good_Length_m >= Customer_Required_Length_m"
    )
    manifest["output_files"] = sorted(set(manifest["output_files"]) | {
        "automated_validation_results.json",
        "previous_experiment_preservation_snapshot.json",
        "prior_vs_revised_model_metrics.csv",
        "prior_vs_revised_target_distributions.csv",
        "revised_outer_cv_residual_summary.csv",
        "revised_worst_fault_subgroups.csv",
        "revised_experiment_report.md",
        "validate_revised_experiment.py",
        "write_revision_report.py",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
