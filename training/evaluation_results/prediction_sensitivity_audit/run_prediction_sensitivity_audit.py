"""Read-only one-factor prediction sensitivity audit for candidate ML models.

Writes reports only beneath this directory. It never fits, tunes, or writes models.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = PROJECT_ROOT / "training"
for import_root in (PROJECT_ROOT, TRAINING_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    DATASET_CANDIDATES,
    EFFICIENCY_TARGET,
    MODEL1_FEATURES,
    PROCESS_TARGETS,
    SCRAP_TARGET,
)
from preprocessing import load_dataset  # noqa: E402
from training.evaluation_results.controlled_stacking.bundle_audit.candidate_inference import (  # noqa: E402
    CandidateInferenceBundle,
    InferenceValidationError,
    PROCESS_OUTPUT_BOUNDS,
)

OUTPUT_DIR = Path(__file__).resolve().parent
CONTROLLED_DIR = PROJECT_ROOT / "training/evaluation_results/controlled_stacking"
CANDIDATE_DIRS = {
    "model2": CONTROLLED_DIR / "model2_candidate",
    "model3": CONTROLLED_DIR / "model3_candidate",
}
NUMERIC_OUTPUTS = PROCESS_TARGETS + [EFFICIENCY_TARGET, SCRAP_TARGET]
OUTPUT_UNITS = {
    **{name: "degC" for name in ("Extruder_Temperature_C", "Die_Temperature_C", "Cooling_Tank_Temperature_C")},
    "Melt_Pressure_bar": "bar",
    "Air_Pressure_bar": "bar",
    "Screw_RPM": "rpm",
    "Line_Speed_m_min": "m/min",
    "Puller_Speed_m_min": "m/min",
    "Extrusion_Speed_kg_hr": "kg/hr",
    EFFICIENCY_TARGET: "percentage_points",
    SCRAP_TARGET: "fraction",
}
OUTPUT_BOUNDS = {
    **PROCESS_OUTPUT_BOUNDS,
    EFFICIENCY_TARGET: (0.0, 100.0),
    SCRAP_TARGET: (0.0, 1.0),
}
GEOMETRY_TOLERANCE_MM = 0.25
GEOMETRY_RULE_DESCRIPTION = "abs(ID - (OD - 2*wall)) < 0.25 mm; ID > 0.5 mm"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics() -> pd.DataFrame:
    files = {
        "model1": PROJECT_ROOT / "training/results/model1_metrics.csv",
        "model2": PROJECT_ROOT / "training/results/model2_metrics.csv",
        "model3": PROJECT_ROOT / "training/results/model3_metrics.csv",
    }
    metadata_path = PROJECT_ROOT / "models/model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    rows = []
    for model_name, path in files.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "model", model_name)
        if model_name == "model2":
            frame["target"] = EFFICIENCY_TARGET
            selected = metadata.get("model2", {}).get("selected_algorithm")
            frame["selected_algorithm"] = frame["algorithm"] == selected
        elif model_name == "model3":
            frame["target"] = SCRAP_TARGET
            selected = metadata.get("model3", {}).get("selected_algorithm")
            frame["selected_algorithm"] = frame["algorithm"] == selected
        else:
            selected_by_target = {
                target: detail.get("selected_algorithm")
                for target, detail in metadata.get("model1", {}).get("targets", {}).items()
            }
            frame["selected_algorithm"] = frame.apply(
                lambda row: row["algorithm"] == selected_by_target.get(row["target"]), axis=1
            )
        rows.extend(frame.to_dict(orient="records"))
    summary_path = CONTROLLED_DIR / "metrics_summary.csv"
    if summary_path.exists():
        nested = pd.read_csv(summary_path)
        nested = nested[(nested["split"] == "outer_cv") & (nested["configuration"] == "full_stack")]
        for row in nested.to_dict(orient="records"):
            rows.append(
                {
                    "model": "controlled_nested_candidate",
                    "target": row["target"],
                    "algorithm": "fixed Random Forest full stack",
                    "split": "nested_cv_mean",
                    "MAE": row["MAE_mean"],
                    "RMSE": row["RMSE_mean"],
                    "R2": row["R2_mean"],
                    "MAE_std": row["MAE_std"],
                    "RMSE_std": row["RMSE_std"],
                    "R2_std": row["R2_std"],
                    "selected_algorithm": False,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(OUTPUT_DIR / "existing_model_metrics.csv", index=False)
    return result


def choose_baseline(df: pd.DataFrame, bundle: CandidateInferenceBundle) -> tuple[dict, int]:
    """Use the first actual dataset row accepted by the candidate's validator."""
    for _, row in df.iterrows():
        values = {field: row[field] for field in MODEL1_FEATURES}
        try:
            bundle.validate_customer_inputs(values)
        except InferenceValidationError:
            continue
        return values, int(row["Record_ID"])
    raise RuntimeError("No dataset row satisfies the existing candidate input validation rules")


def scenario_values(feature: str, baseline: dict, df: pd.DataFrame, bundle: CandidateInferenceBundle) -> list[tuple[str, object]]:
    baseline_value = baseline[feature]
    candidates: list[tuple[str, object]] = [("baseline", baseline_value)]
    if feature in CATEGORICAL_FEATURES:
        options = sorted(bundle.category_values[feature])
        candidates.extend(("category", option) for option in options if option != baseline_value)
    elif feature == "Target_Tube_OD_mm":
        candidates.extend(("geometry_tolerance_offset_mm", round(float(baseline_value) + delta, 6)) for delta in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20))
    elif feature == "Target_Tube_Wall_mm":
        candidates.extend(("geometry_tolerance_offset_mm", round(float(baseline_value) + delta, 6)) for delta in (-0.10, -0.05, 0.05, 0.10))
    elif feature == "Target_Tube_ID_mm":
        candidates.extend(("geometry_tolerance_offset_mm", round(float(baseline_value) + delta, 6)) for delta in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20))
    elif feature == "Customer_Required_Length_m":
        quantiles = np.quantile(df[feature].to_numpy(dtype=float), [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
        candidates.extend((f"dataset_q{int(q * 100):02d}", round(float(value), 2)) for q, value in zip((0, .10, .25, .50, .75, .90, 1), quantiles))

    deduplicated = []
    seen = set()
    for label, value in candidates:
        key = (label, str(value))
        if key not in seen:
            deduplicated.append((label, value))
            seen.add(key)
    return deduplicated


def output_distribution_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for output in NUMERIC_OUTPUTS:
        values = df[output].to_numpy(dtype=float)
        row = {
            "output": output,
            "unit": OUTPUT_UNITS[output],
            "count": int(np.isfinite(values).sum()),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
            row[f"q{int(q * 100):02d}"] = float(np.quantile(values, q))
        row["documented_bound_low"] = OUTPUT_BOUNDS[output][0]
        row["documented_bound_high"] = OUTPUT_BOUNDS[output][1]
        rows.append(row)
    return rows


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.corr(right))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Configured source dataset is missing")
    dataset_hash = sha256(dataset_path)
    df = load_dataset()
    if df.shape != (5000, 40):
        raise RuntimeError(f"Expected the existing 5000 x 40 dataset, found {df.shape}")

    bundle = CandidateInferenceBundle.load(CONTROLLED_DIR)
    baseline, baseline_record_id = choose_baseline(df, bundle)
    baseline_result = bundle.predict(baseline)
    baseline_outputs = baseline_result.predictions.iloc[0].to_dict()
    records: list[dict] = []

    # Each case changes exactly one customer feature; all other inputs are copied from one valid data row.
    for feature in MODEL1_FEATURES:
        for scenario_index, (scenario_label, candidate_value) in enumerate(
            scenario_values(feature, baseline, df, bundle)
        ):
            case_inputs = dict(baseline)
            case_inputs[feature] = candidate_value
            geometry_residual = abs(
                float(case_inputs["Target_Tube_ID_mm"])
                - (float(case_inputs["Target_Tube_OD_mm"]) - 2.0 * float(case_inputs["Target_Tube_Wall_mm"]))
            )
            row = {
                "case_id": f"{feature}_{scenario_index:02d}",
                "varied_feature": feature,
                "scenario": scenario_label,
                "varied_value": candidate_value,
                "baseline_record_id": baseline_record_id,
                "geometry_residual_mm": geometry_residual,
                "geometry_rule_pass": geometry_residual < GEOMETRY_TOLERANCE_MM
                and float(case_inputs["Target_Tube_ID_mm"]) > 0.5,
                "accepted": False,
                "validation_error": "",
            }
            for input_name in MODEL1_FEATURES:
                row[f"input_{input_name}"] = case_inputs[input_name]
            try:
                result = bundle.predict(case_inputs)
            except InferenceValidationError as error:
                row["validation_error"] = str(error)
            else:
                prediction = result.predictions.iloc[0]
                row["accepted"] = True
                row["warnings"] = " | ".join(result.warnings)
                for output in NUMERIC_OUTPUTS:
                    row[output] = float(prediction[output])
                    row[f"delta_{output}_from_baseline"] = float(prediction[output]) - float(baseline_outputs[output])
                    lower, upper = OUTPUT_BOUNDS[output]
                    row[f"{output}_within_documented_bounds"] = lower <= float(prediction[output]) <= upper
                    observed = df[output].to_numpy(dtype=float)
                    row[f"{output}_within_dataset_min_max"] = float(observed.min()) <= float(prediction[output]) <= float(observed.max())
                    q01, q99 = np.quantile(observed, [0.01, 0.99])
                    row[f"{output}_within_dataset_q01_q99"] = float(q01) <= float(prediction[output]) <= float(q99)
            records.append(row)

    predictions = pd.DataFrame(records)
    predictions.to_csv(OUTPUT_DIR / "one_factor_predictions.csv", index=False)
    distributions = pd.DataFrame(output_distribution_rows(df))
    distributions.to_csv(OUTPUT_DIR / "dataset_output_distributions.csv", index=False)

    sensitivity_rows = []
    accepted = predictions[predictions["accepted"]].copy()
    baseline_rows = accepted[accepted["scenario"] == "baseline"]
    if len(baseline_rows) != len(MODEL1_FEATURES):
        raise RuntimeError("Expected one accepted baseline prediction for every customer input")
    for feature in MODEL1_FEATURES:
        subset = accepted[accepted["varied_feature"] == feature]
        changed = subset[subset["scenario"] != "baseline"]
        for output in NUMERIC_OUTPUTS:
            distribution = distributions.loc[distributions["output"] == output].iloc[0]
            values = subset[output].to_numpy(dtype=float)
            deltas = changed[f"delta_{output}_from_baseline"].to_numpy(dtype=float)
            output_scale = float(distribution["std"])
            sensitivity_rows.append(
                {
                    "input_feature": feature,
                    "output": output,
                    "output_unit": OUTPUT_UNITS[output],
                    "accepted_scenario_count_including_baseline": int(len(subset)),
                    "rejected_scenario_count": int((predictions["varied_feature"] == feature).sum() - len(subset)),
                    "distinct_predictions_at_1e-12": int(np.unique(np.round(values, 12)).size),
                    "all_predictions_identical_to_baseline_at_1e-12": bool(np.all(np.abs(deltas) <= 1e-12)),
                    "max_abs_delta_from_baseline": float(np.max(np.abs(deltas))) if len(deltas) else 0.0,
                    "prediction_span": float(np.ptp(values)) if len(values) else 0.0,
                    "dataset_output_std": output_scale,
                    "max_abs_delta_in_dataset_std": float(np.max(np.abs(deltas)) / output_scale) if output_scale > 0 and len(deltas) else 0.0,
                    "large_change_over_1_dataset_std": bool(output_scale > 0 and len(deltas) and np.max(np.abs(deltas)) > output_scale),
                    "dataset_numeric_input_output_correlation": (
                        safe_correlation(df[feature], df[output]) if feature in CUSTOMER_NUMERIC_FEATURES else np.nan
                    ),
                    "numeric_scenario_output_correlation": (
                        safe_correlation(changed[f"input_{feature}"].astype(float), changed[output])
                        if feature in CUSTOMER_NUMERIC_FEATURES and len(changed) > 1 and changed[f"input_{feature}"].nunique() > 1
                        else np.nan
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUTPUT_DIR / "input_output_sensitivity.csv", index=False)

    relationships = []
    data_speed_gap = (df["Line_Speed_m_min"] - df["Puller_Speed_m_min"]).abs()
    for frame_name, frame in (("dataset", df), ("audit_predictions", accepted)):
        if frame_name == "dataset":
            gap = data_speed_gap
            screw_rate_corr = safe_correlation(df["Screw_RPM"], df["Extrusion_Speed_kg_hr"])
        else:
            gap = (frame["Line_Speed_m_min"] - frame["Puller_Speed_m_min"]).abs()
            screw_rate_corr = safe_correlation(frame["Screw_RPM"], frame["Extrusion_Speed_kg_hr"])
        relationships.append(
            {
                "source": frame_name,
                "relationship": "abs(Line_Speed_m_min - Puller_Speed_m_min)",
                "count": int(len(gap)),
                "mean": float(gap.mean()),
                "q95": float(gap.quantile(0.95)),
                "q99": float(gap.quantile(0.99)),
                "max": float(gap.max()),
                "audit_rows_beyond_dataset_q99": (
                    int((gap > data_speed_gap.quantile(0.99)).sum()) if frame_name == "audit_predictions" else 0
                ),
            }
        )
        relationships.append(
            {
                "source": frame_name,
                "relationship": "corr(Screw_RPM, Extrusion_Speed_kg_hr)",
                "count": int(len(frame)),
                "pearson_correlation": float(screw_rate_corr),
                "dataset_reference_correlation": safe_correlation(df["Screw_RPM"], df["Extrusion_Speed_kg_hr"]),
            }
        )
    pd.DataFrame(relationships).to_csv(OUTPUT_DIR / "relationship_diagnostics.csv", index=False)

    metric_rows = read_metrics()
    manifest_paths = [
        CONTROLLED_DIR / "model2_candidate/candidate_manifest.json",
        CONTROLLED_DIR / "model3_candidate/candidate_manifest.json",
        CONTROLLED_DIR / "bundle_audit/bundle_audit_summary.json",
        PROJECT_ROOT / "training/results/model1_metrics.csv",
        PROJECT_ROOT / "training/results/model2_metrics.csv",
        PROJECT_ROOT / "training/results/model3_metrics.csv",
    ]
    artifact_hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
        for path in manifest_paths
        if path.exists()
    }
    artifact_hashes[str(dataset_path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = dataset_hash
    audit_manifest = {
        "audit": "one_factor_candidate_prediction_sensitivity",
        "dataset_sha256": dataset_hash,
        "dataset_shape": list(df.shape),
        "baseline_record_id": baseline_record_id,
        "customer_feature_order": MODEL1_FEATURES,
        "categorical_options_from_candidate_encoders": {
            feature: sorted(bundle.category_values[feature]) for feature in CATEGORICAL_FEATURES
        },
        "geometry_rule": GEOMETRY_RULE_DESCRIPTION,
        "geometry_test_policy": "OD +/- 0.20 mm, wall +/- 0.10 mm, ID +/- 0.20 mm; vary one field at a time and retain only inputs accepted by existing candidate validation",
        "categorical_policy": "all fitted encoder categories; literal Braid_Type='None' included",
        "large_change_flag_definition": "absolute prediction change greater than one full-dataset standard deviation for that output; statistical screening threshold, not an engineering limit",
        "output_bounds": OUTPUT_BOUNDS,
        "artifact_and_metric_hashes": artifact_hashes,
        "files_written": [
            "one_factor_predictions.csv",
            "input_output_sensitivity.csv",
            "dataset_output_distributions.csv",
            "relationship_diagnostics.csv",
            "existing_model_metrics.csv",
            "audit_manifest.json",
            "audit_report.md",
        ],
        "read_only_scope": "No dataset, generator, model artifact, metadata, or inference module is written by this script.",
    }
    (OUTPUT_DIR / "audit_manifest.json").write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")
    write_report(predictions, sensitivity, distributions, metric_rows, relationships, audit_manifest)
    print(f"Audit outputs written to {OUTPUT_DIR}")
    print(f"Baseline Record_ID: {baseline_record_id}")
    print(f"Rows: {len(predictions)} total, {int(predictions['accepted'].sum())} accepted, {int((~predictions['accepted']).sum())} rejected")
    print(f"Dataset SHA-256: {dataset_hash}")


def write_report(
    predictions: pd.DataFrame,
    sensitivity: pd.DataFrame,
    distributions: pd.DataFrame,
    metrics: pd.DataFrame,
    relationships: list[dict],
    manifest: dict,
) -> None:
    accepted = predictions[predictions["accepted"]]
    rejected = predictions[~predictions["accepted"]]
    output_lines = []
    for output in NUMERIC_OUTPUTS:
        ranked = sensitivity[sensitivity["output"] == output].sort_values(
            "max_abs_delta_in_dataset_std", ascending=False
        )
        top = ranked.iloc[0]
        output_lines.append(
            f"- `{output}` ({OUTPUT_UNITS[output]}): largest one-factor local response was `{top['input_feature']}` "
            f"(max |delta| {top['max_abs_delta_from_baseline']:.5g}, "
            f"{top['max_abs_delta_in_dataset_std']:.3f} dataset SD over tested cases)."
        )
    identical_pairs = sensitivity[sensitivity["all_predictions_identical_to_baseline_at_1e-12"]]
    identical_summary = (
        ", ".join(f"{row.input_feature} → {row.output}" for row in identical_pairs.itertuples())
        if not identical_pairs.empty
        else "none"
    )
    documented_bound_violations = int(
        (~accepted[[f"{output}_within_documented_bounds" for output in NUMERIC_OUTPUTS]].astype(bool)).to_numpy().sum()
    )
    observed_range_violations = int(
        (~accepted[[f"{output}_within_dataset_min_max" for output in NUMERIC_OUTPUTS]].astype(bool)).to_numpy().sum()
    )
    central_range_violations = int(
        (~accepted[[f"{output}_within_dataset_q01_q99" for output in NUMERIC_OUTPUTS]].astype(bool)).to_numpy().sum()
    )
    speed_relationship = next(
        row for row in relationships
        if row["source"] == "audit_predictions"
        and row["relationship"] == "abs(Line_Speed_m_min - Puller_Speed_m_min)"
    )
    large_response_rows = sensitivity[sensitivity["large_change_over_1_dataset_std"]]
    large_response_summary = (
        ", ".join(f"{row.input_feature} → {row.output}" for row in large_response_rows.itertuples())
        if not large_response_rows.empty
        else "none"
    )
    weaknesses = []
    if not metrics.empty:
        for target, display_name in ((EFFICIENCY_TARGET, "Model 2"), (SCRAP_TARGET, "Model 3")):
            nested = metrics[
                (metrics.get("target", pd.Series(index=metrics.index, dtype=object)) == target)
                & (metrics.get("split", pd.Series(index=metrics.index, dtype=object)) == "nested_cv_mean")
            ]
            if not nested.empty:
                row = nested.iloc[0]
                weaknesses.append(
                    f"- {display_name} corrected nested full-stack CV: MAE={row['MAE']:.5g} ± {row['MAE_std']:.5g}, "
                    f"RMSE={row['RMSE']:.5g} ± {row['RMSE_std']:.5g}, R²={row['R2']:.4f} ± {row['R2_std']:.4f}."
                )
            production = metrics[
                (metrics["model"] == {"Model 2": "model2", "Model 3": "model3"}[display_name])
                & (metrics.get("target", pd.Series(index=metrics.index, dtype=object)) == target)
                & (metrics["split"] == "test")
                & metrics.get("selected_algorithm", pd.Series(False, index=metrics.index)).astype(bool)
            ]
            if not production.empty:
                row = production.iloc[0]
                weaknesses.append(
                    f"- {display_name} selected saved held-out test: MAE={row['MAE']:.5g}, RMSE={row['RMSE']:.5g}, R²={row['R2']:.4f}."
                )
        model1 = metrics[
            (metrics["model"] == "model1")
            & (metrics["split"] == "test")
            & metrics.get("selected_algorithm", pd.Series(False, index=metrics.index)).astype(bool)
        ]
        if not model1.empty:
            weakest = model1.sort_values("R2").drop_duplicates("target").head(3)
            weaknesses.extend(
                f"- Model 1 weaker saved test target `{row.target}`: R²={row.R2:.4f}, MAE={row.MAE:.5g}, RMSE={row.RMSE:.5g}."
                for row in weakest.itertuples()
            )
    report = [
        "# Candidate Prediction Sensitivity Audit",
        "",
        "## Scope",
        "",
        f"Read-only one-factor audit using dataset SHA-256 `{manifest['dataset_sha256']}` ({manifest['dataset_shape'][0]} rows x {manifest['dataset_shape'][1]} columns). Baseline customer inputs come from dataset Record_ID `{manifest['baseline_record_id']}`. No models were fit or changed.",
        "",
        f"Tested {len(predictions)} scenarios: {len(accepted)} accepted by the inference validator and {len(rejected)} rejected. Every scenario changes one requested customer input while holding the other seven at the baseline.",
        "",
        "## Geometry and categorical coverage",
        "",
        f"Categorical values are enumerated from fitted candidate encoder vocabularies; `Braid_Type='None'` is included. Geometry rule from the existing synthetic generator: `{GEOMETRY_RULE_DESCRIPTION}`.",
        "",
        "OD, wall, and ID cannot be broadly varied independently while satisfying the geometry rule. Their one-factor tests therefore use only local offsets within the existing tolerance; this is a local model sensitivity check, not a broad geometry sweep. Required-length tests use dataset quantiles. No new engineering limits are imposed.",
        "",
        "## Input influence screening",
        "",
        "Influence is ranked by maximum absolute output change divided by that output's full-dataset standard deviation. A change over 1 dataset SD is flagged only as a statistical screening heuristic, not an engineering significance threshold.",
        "",
        *output_lines,
        "",
        f"No-change pairs (all tested values identical to baseline within 1e-12): {identical_summary}.",
        f"Changes greater than 1 full-dataset SD (heuristic): {large_response_summary}.",
        "",
        "## Distribution and relationship checks",
        "",
        f"Across accepted scenario-output pairs: {documented_bound_violations} predictions violated existing documented bounds, {observed_range_violations} fell outside the dataset observed min/max, and {central_range_violations} fell outside the dataset q01–q99 range.",
        f"Predicted absolute line/puller speed gaps exceeded the synthetic dataset q99 in {int(speed_relationship['audit_rows_beyond_dataset_q99'])} of {len(accepted)} accepted cases. Screw-RPM/extrusion-rate correlations are in `relationship_diagnostics.csv`.",
        "Output quantiles and existing synthetic bounds are in `dataset_output_distributions.csv`; per-case empirical-range flags are in `one_factor_predictions.csv`. These screens are not validated manufacturing limits.",
        "",
        "## Saved evaluation weaknesses",
        "",
        *(weaknesses or ["- Saved metric files were unavailable; model-performance review is inconclusive."]),
        "",
        "The candidate Model 2/3 test scores and Model 1 target metrics are historical saved estimates; sensitivity results here do not replace nested-CV or held-out evaluation.",
        "",
        "## Limitations and Next Steps",
        "",
        "All artifacts and data are synthetic. Local geometric perturbations are intentionally narrow because the repository defines a coupled OD/wall/ID rule. One-factor predictions do not establish causal effects or factory performance. Review rejected scenarios and high delta/SD rows in the CSVs; use real, validated production data and engineering review before operational use.",
        "",
        "No dataset, generator, model artifact, metadata, inference logic, or Streamlit UI is modified by this audit.",
        "",
    ]
    (OUTPUT_DIR / "audit_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
