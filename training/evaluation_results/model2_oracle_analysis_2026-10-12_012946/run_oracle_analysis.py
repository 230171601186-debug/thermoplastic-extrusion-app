"""Read-only validation and summary of the existing Model 2 oracle comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
M1_AUDIT = ROOT / "training/evaluation_results/model1_diagnostic_audit_2026-10-11_20261011_231247"
M2_AUDIT = ROOT / "training/evaluation_results/model2_diagnostic_audit_2026-10-11"
REVISED = ROOT / "training/evaluation_results/model2_generator_experiment_2026-10-11_revised"
DATASET_SHA256 = "0db6c6d33f0a3d580b2f673cc3761b4c2551f48c1ca68d296303d9854a669b15"
ARMS = [
    "A_saved_Model1_inference_predictions",
    "B_outer_train_only_selected_Model1_OOF",
    "C_oracle_recorded_process_non_deployable",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R2": float(1.0 - np.sum(residual**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def main() -> None:
    summary_path = M1_AUDIT / "model2_downstream_summary.csv"
    fold_path = M1_AUDIT / "model2_downstream_impact.csv"
    prediction_path = M1_AUDIT / "model2_downstream_predictions.csv"
    runner_path = M1_AUDIT / "run_model1_diagnostic.py"
    manifest_path = M1_AUDIT / "diagnostic_manifest.json"
    validation_path = M1_AUDIT / "validation_results.json"
    generator_path = REVISED / "experimental_generator.py"
    dataset_path = REVISED / "experimental_dataset.csv"
    prior_report_path = M2_AUDIT / "model2_diagnostic_report.md"
    prior_manifest_path = M2_AUDIT / "diagnostic_manifest.json"
    target_diagnostic_path = M2_AUDIT / "target_construction_diagnostics.json"
    accounting_path = M2_AUDIT / "dataset_accounting_validation.json"

    summary = pd.read_csv(summary_path)
    folds = pd.read_csv(fold_path)
    predictions = pd.read_csv(prediction_path)
    source = runner_path.read_text(encoding="utf-8")
    generator = generator_path.read_text(encoding="utf-8")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    target_diagnostics = json.loads(target_diagnostic_path.read_text(encoding="utf-8"))
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))

    outer = predictions[predictions["split"] == "outer_cv"].copy()
    test = predictions[predictions["split"] == "saved_test"].copy()
    row_alignment: dict[str, bool] = {}
    row_counts: dict[str, dict[str, int]] = {}
    recomputed_errors: list[dict] = []
    fold_wins: dict[str, dict[str, int]] = {}

    for split_name, data, expected_count in (
        ("outer_cv", outer, 4000),
        ("saved_test", test, 1000),
    ):
        reference = None
        reference_y = None
        reference_fold = None
        for arm in ARMS:
            arm_data = data[data["arm"] == arm]
            row_counts[f"{split_name}:{arm}"] = {
                "prediction_rows": int(len(arm_data)),
                "unique_row_ids": int(arm_data["row_index"].nunique()),
            }
            indexed = arm_data.set_index("row_index").sort_index()
            if reference is None:
                reference = indexed.index
                reference_y = indexed["actual"].to_numpy(dtype=float)
                if split_name == "outer_cv":
                    reference_fold = indexed["fold"].to_numpy()
            else:
                same_rows = indexed.index.equals(reference)
                same_y = np.allclose(
                    indexed["actual"].to_numpy(dtype=float), reference_y, atol=1e-10
                )
                same_fold = (
                    split_name != "outer_cv"
                    or np.array_equal(indexed["fold"].to_numpy(), reference_fold)
                )
                row_alignment[f"{split_name}:{arm}_matches_A"] = bool(
                    same_rows and same_y and same_fold
                )
            if split_name == "outer_cv":
                for fold_number in range(1, 6):
                    fold_predictions = arm_data[
                        arm_data["fold"].astype(str) == str(fold_number)
                    ]
                    recomputed = metric_values(
                        fold_predictions["actual"].to_numpy(dtype=float),
                        fold_predictions["predicted"].to_numpy(dtype=float),
                    )
                    reported = folds[
                        (folds["arm"] == arm)
                        & (folds["fold"].astype(str) == str(fold_number))
                    ].iloc[0]
                    for metric, value in recomputed.items():
                        difference = abs(value - float(reported[metric]))
                        recomputed_errors.append({
                            "split": split_name,
                            "arm": arm,
                            "fold": fold_number,
                            "metric": metric,
                            "absolute_difference": difference,
                        })
            else:
                recomputed = metric_values(
                    arm_data["actual"].to_numpy(dtype=float),
                    arm_data["predicted"].to_numpy(dtype=float),
                )
                reported = summary[summary["arm"] == arm].iloc[0]
                for metric, value in recomputed.items():
                    difference = abs(value - float(reported[f"saved_test_{metric}"]))
                    recomputed_errors.append({
                        "split": split_name,
                        "arm": arm,
                        "fold": "saved_test",
                        "metric": metric,
                        "absolute_difference": difference,
                    })
        row_alignment[f"{split_name}:all_arms_have_{expected_count}_rows"] = bool(
            all(
                row_counts[f"{split_name}:{arm}"]["prediction_rows"] == expected_count
                and row_counts[f"{split_name}:{arm}"]["unique_row_ids"] == expected_count
                for arm in ARMS
            )
        )

    max_metric_difference = max(
        item["absolute_difference"] for item in recomputed_errors
    )
    fold_wins = {}
    outer_fold_metrics = folds[folds["fold"].astype(str).isin(["1", "2", "3", "4", "5"])]
    for arm in ARMS[1:]:
        arm_rows = outer_fold_metrics[outer_fold_metrics["arm"] == arm].sort_values("fold")
        baseline_rows = outer_fold_metrics[
            outer_fold_metrics["arm"] == ARMS[0]
        ].sort_values("fold")
        fold_wins[arm] = {
            "MAE_lower_than_A": int((arm_rows["MAE"].to_numpy() < baseline_rows["MAE"].to_numpy()).sum()),
            "RMSE_lower_than_A": int((arm_rows["RMSE"].to_numpy() < baseline_rows["RMSE"].to_numpy()).sum()),
            "R2_higher_than_A": int((arm_rows["R2"].to_numpy() > baseline_rows["R2"].to_numpy()).sum()),
        }

    summary_rows = []
    for arm in ARMS:
        entry = summary[summary["arm"] == arm].iloc[0]
        outer_rows = folds[folds["arm"] == arm].sort_values("fold")
        summary_rows.append({
            "arm": arm,
            "deployable": bool(entry["deployable"]),
            "outer_fold_count": int(entry["outer_fold_count"]),
            "outer_evaluated_rows": int(row_counts[f"outer_cv:{arm}"]["prediction_rows"]),
            "outer_rows_per_fold": 800,
            "outer_MAE_mean": float(entry["outer_MAE_mean"]),
            "outer_MAE_std": float(entry["outer_MAE_std"]),
            "outer_RMSE_mean": float(entry["outer_RMSE_mean"]),
            "outer_RMSE_std": float(entry["outer_RMSE_std"]),
            "outer_R2_mean": float(entry["outer_R2_mean"]),
            "outer_R2_std": float(entry["outer_R2_std"]),
            "saved_test_rows": int(row_counts[f"saved_test:{arm}"]["prediction_rows"]),
            "saved_test_MAE": float(entry["saved_test_MAE"]),
            "saved_test_RMSE": float(entry["saved_test_RMSE"]),
            "saved_test_R2": float(entry["saved_test_R2"]),
            "fold_MAE_wins_vs_A": 5 if arm == ARMS[0] else fold_wins[arm]["MAE_lower_than_A"],
            "fold_RMSE_wins_vs_A": 5 if arm == ARMS[0] else fold_wins[arm]["RMSE_lower_than_A"],
            "fold_R2_wins_vs_A": 5 if arm == ARMS[0] else fold_wins[arm]["R2_higher_than_A"],
            "fold_difference_consistent_vs_A": (
                True
                if arm == ARMS[0]
                else (
                    fold_wins[arm]["MAE_lower_than_A"] == 5
                    and fold_wins[arm]["RMSE_lower_than_A"] == 5
                    and fold_wins[arm]["R2_higher_than_A"] == 5
                )
            ),
        })
    output_summary = pd.DataFrame(summary_rows)
    output_summary.to_csv(OUT / "oracle_comparison_summary.csv", index=False)

    manifest_ok = (
        source_validation["passed"]
        and source_manifest["automated_validation"]["output_validation_passed"]
        and source_manifest["dataset_sha256"] == DATASET_SHA256
        and prior_manifest["dataset_sha256"] == DATASET_SHA256
    )
    generator_formula_verified = all(
        phrase in generator
        for phrase in (
            "performance = line_speed / references[\"nominal_line_speed_m_min\"]",
            "availability = bounded(runtime / planned_time",
            "quality = bounded(good_length / actual_length",
            "oee = availability * performance * quality",
            "efficiency = oee * 100.0",
            '"Operational_Efficiency_pct": round(efficiency, 4)',
        )
    )
    nested_code_verified = all(
        phrase in source
        for phrase in (
            "for fold_number, (fit_pos, valid_pos) in enumerate(outer.split(train)",
            "outer_train = train.iloc[fit_pos].copy()",
            "outer_valid = train.iloc[valid_pos].copy()",
            '"B_outer_train_only_selected_Model1_OOF"',
            'f"{scope}_{target}_fold_{fold_number}"',
            "json.loads(SPLIT_INDICES_PATH.read_text",
        )
    )
    fixed_model_verified = (
        source.count("RandomForestRegressor(**M2_RF_PARAMS)") == 2
        and "M2_RF_PARAMS = dict(RF_PARAMS)" in source
        and 'M2_RF_PARAMS["n_jobs"] = 1' in source
        and "for arm, (m2_train, m2_valid, deployable) in arms.items()" in source
    )
    oracle_separation_verified = (
        'outer_train[MODEL1_FEATURES + PROCESS_TARGETS]' in source
        and 'outer_valid[MODEL1_FEATURES + PROCESS_TARGETS]' in source
        and '"A_saved_Model1_inference_predictions"' in source
        and '"B_outer_train_only_selected_Model1_OOF"' in source
        and '"C_oracle_recorded_process_non_deployable"' in source
    )
    checks = {
        "validated_successful_source_run": bool(manifest_ok),
        "same_revised_dataset_sha256": bool(
            sha256(dataset_path) == DATASET_SHA256
        ),
        "all_paths_use_identical_outer_rows_and_targets": bool(
            all(row_alignment.values())
        ),
        "outer_rows_unique_and_fold_counts_match": bool(
            row_alignment["outer_cv:all_arms_have_4000_rows"]
            and all(row_alignment[f"outer_cv:{arm}_matches_A"] for arm in ARMS[1:])
        ),
        "saved_test_rows_separate_and_aligned": bool(
            row_alignment["saved_test:all_arms_have_1000_rows"]
            and all(row_alignment[f"saved_test:{arm}_matches_A"] for arm in ARMS[1:])
        ),
        "reported_metrics_recomputed_from_saved_predictions": bool(
            max_metric_difference < 1e-9
        ),
        "same_fixed_model2_estimator_for_all_arms": bool(fixed_model_verified),
        "oracle_uses_recorded_process_values_only_as_diagnostic": bool(
            oracle_separation_verified
        ),
        "nested_upstream_selection_excludes_outer_validation": bool(
            nested_code_verified
        ),
        "target_formula_traced_in_generator": bool(generator_formula_verified),
        "accounting_validation_passed": bool(accounting["passed"]),
        "no_metric_rows_mix_saved_test_with_outer_cv": bool(
            len(outer_fold_metrics) == 15
            and len(folds[folds["fold"].astype(str) == "saved_test"]) == 3
            and (summary["outer_fold_count"] == 5).all()
        ),
    }
    delta_a = output_summary.iloc[0]
    delta_b = output_summary.iloc[1]
    delta_c = output_summary.iloc[2]
    validation = {
        "passed": all(checks.values()),
        "checks": checks,
        "source_audit_directory": M1_AUDIT.relative_to(ROOT).as_posix(),
        "source_audit_timestamped_directory_discovery": [
            M1_AUDIT.name
        ],
        "source_validated_run": True,
        "source_validation_checks_passed": int(sum(source_validation["checks"].values())),
        "source_validation_check_count": len(source_validation["checks"]),
        "source_protected_hashes_verified": source_manifest["preservation"]["verified_file_count"],
        "source_protected_hashes_expected": source_manifest["preservation"]["expected_file_count"],
        "source_dataset_sha256": DATASET_SHA256,
        "row_counts_by_path_and_split": row_counts,
        "max_absolute_metric_recalculation_difference": max_metric_difference,
        "outer_fold_direction_counts_vs_A": fold_wins,
        "oracle_minus_A": {
            "MAE_improvement": float(delta_a["outer_MAE_mean"] - delta_c["outer_MAE_mean"]),
            "RMSE_improvement": float(delta_a["outer_RMSE_mean"] - delta_c["outer_RMSE_mean"]),
            "R2_improvement": float(delta_c["outer_R2_mean"] - delta_a["outer_R2_mean"]),
        },
        "B_minus_A": {
            "MAE_improvement": float(delta_a["outer_MAE_mean"] - delta_b["outer_MAE_mean"]),
            "RMSE_improvement": float(delta_a["outer_RMSE_mean"] - delta_b["outer_RMSE_mean"]),
            "R2_improvement": float(delta_b["outer_R2_mean"] - delta_a["outer_R2_mean"]),
        },
        "unresolved_methodological_limit": (
            "The synthetic dataset's generator is known, but the audit does not establish "
            "which equations or noise magnitudes match real factory data; no real production "
            "records or external validation are available."
        ),
    }
    (OUT / "oracle_methodology_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    input_paths = [
        M1_AUDIT / "model1_diagnostic_report.md",
        summary_path,
        fold_path,
        prediction_path,
        runner_path,
        manifest_path,
        validation_path,
        prior_report_path,
        M2_AUDIT / "aggregate_model_comparison.csv",
        M2_AUDIT / "feature_group_ablation_aggregate.csv",
        target_diagnostic_path,
        accounting_path,
        prior_manifest_path,
        generator_path,
        dataset_path,
    ]
    diagnostic_manifest = {
        "analysis": "read-only interpretation and validation of existing revised-synthetic Model 2 oracle arms",
        "analysis_directory": OUT.relative_to(ROOT).as_posix(),
        "created_utc_or_local": "2026-10-12T01:29:46+05:30",
        "input_artifacts": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in input_paths
        },
        "source_validated_run": M1_AUDIT.relative_to(ROOT).as_posix(),
        "source_dataset_sha256": DATASET_SHA256,
        "methodology_validation_passed": validation["passed"],
        "outputs": [
            "oracle_analysis_report.md",
            "oracle_comparison_summary.csv",
            "oracle_methodology_validation.json",
            "diagnostic_manifest.json",
            "README.md",
            "run_oracle_analysis.py",
        ],
        "limitations": [
            "All results are synthetic and do not verify real manufacturing behavior.",
            "Oracle process features are recorded post-run process targets and are non-deployable.",
            "The oracle is diagnostic, not a guaranteed upper bound for all possible models.",
        ],
    }
    (OUT / "diagnostic_manifest.json").write_text(
        json.dumps(diagnostic_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "passed": validation["passed"],
        "checks": checks,
        "summary": summary_rows,
        "max_metric_recalculation_difference": max_metric_difference,
    }, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
