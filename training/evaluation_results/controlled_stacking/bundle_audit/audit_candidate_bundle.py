"""Read-only compatibility audit for the Model 2 and Model 3 candidate bundle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[4]
TRAINING_DIR = ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    DATASET_CANDIDATES,
    EFFICIENCY_TARGET,
    MODEL1_FEATURES,
    PROCESS_MODEL_FILENAMES,
    PROCESS_TARGETS,
    PROJECT_ROOT,
    RANDOM_STATE,
    SCRAP_TARGET,
    SPLIT_INDICES_PATH,
    TEST_SIZE,
)
from preprocessing import load_dataset  # noqa: E402

BASE = PROJECT_ROOT / "training/evaluation_results/controlled_stacking"
M2_DIR = BASE / "model2_candidate"
M3_DIR = BASE / "model3_candidate"
OUTPUT_DIR = Path(__file__).resolve().parent
EXPECTED_DATASET_SHA256 = "cbd42684bc475ca9a7ad4f5b4512962c785034f2aae7c4944f0804e624c20396"
STACKED_FEATURES = MODEL1_FEATURES + PROCESS_TARGETS
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(directory: Path, manifest: dict, model_key: str):
    paths = {
        target: PROJECT_ROOT / relative_path
        for target, relative_path in manifest["candidate_artifacts"]["model1"].items()
    }
    downstream_path = PROJECT_ROOT / manifest["candidate_artifacts"][model_key]
    if set(paths) != set(PROCESS_TARGETS):
        raise RuntimeError(f"{model_key}: manifest does not identify all nine Model 1 targets")
    expected_process_paths = {
        target: directory / "candidate_artifacts" / filename
        for target, filename in PROCESS_MODEL_FILENAMES.items()
    }
    if paths != expected_process_paths:
        raise RuntimeError(f"{model_key}: artifact paths differ from the actual candidate bundle")
    models = {target: joblib.load(path) for target, path in paths.items()}
    downstream = joblib.load(downstream_path)
    return paths, downstream_path, models, downstream


def check_inference_pipeline(pipeline, expected_columns: list[str], label: str) -> list[str]:
    actual = list(getattr(pipeline, "feature_names_in_", []))
    if actual != expected_columns:
        raise RuntimeError(f"{label}: expected {expected_columns}, found {actual}")
    if "preprocess" not in pipeline.named_steps:
        raise RuntimeError(f"{label}: missing fitted preprocessing step")
    categorical = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    # The fitted encoder's explicit category list distinguishes None from an NA.
    if not hasattr(categorical, "handle_unknown") or categorical.handle_unknown != "ignore":
        raise RuntimeError(f"{label}: unseen-category policy is not ignore")
    return actual


def predict_bundle(customer_frame: pd.DataFrame, models: dict, downstream) -> tuple[pd.DataFrame, np.ndarray]:
    stacked = customer_frame[MODEL1_FEATURES].copy()
    for target in PROCESS_TARGETS:
        predicted = np.asarray(models[target].predict(customer_frame[MODEL1_FEATURES]))
        if predicted.shape != (len(customer_frame),) or not np.issubdtype(predicted.dtype, np.number):
            raise RuntimeError(f"Model 1 {target} returned an invalid shape or dtype")
        if not np.isfinite(predicted).all():
            raise RuntimeError(f"Model 1 {target} returned non-finite values")
        stacked[target] = predicted
    if list(stacked.columns) != MODEL1_FEATURES + PROCESS_TARGETS:
        raise RuntimeError("Stacked feature names/order differ from the training contract")
    output = np.asarray(downstream.predict(stacked))
    if output.shape != (len(customer_frame),) or not np.issubdtype(output.dtype, np.number):
        raise RuntimeError("Downstream pipeline returned an invalid shape or dtype")
    if not np.isfinite(output).all():
        raise RuntimeError("Downstream pipeline returned non-finite predictions")
    return stacked, output


def braid_none_is_encoded(pipeline) -> bool:
    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    category_map = dict(zip(CATEGORICAL_FEATURES, encoder.categories_))
    return "None" in category_map["Braid_Type"]


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def expect_missing_error(call, label: str) -> dict[str, str]:
    try:
        call()
    except (ValueError, KeyError) as error:
        return {"case": label, "result": "rejected", "exception": type(error).__name__, "message": str(error)}
    return {"case": label, "result": "accepted_without_required_input", "exception": "", "message": ""}


def capture_optional_value_behavior(call, label: str) -> dict[str, object]:
    try:
        prediction = np.asarray(call())
    except (ValueError, KeyError) as error:
        return {
            "case": label,
            "result": "rejected",
            "exception": type(error).__name__,
            "message": str(error),
            "prediction_shape": "",
            "prediction_dtype": "",
            "finite": False,
            "prediction_min": np.nan,
            "prediction_max": np.nan,
        }
    return {
        "case": label,
        "result": "accepted_missing_value",
        "exception": "",
        "message": "No explicit missing-value rejection in the serialized inference pipeline",
        "prediction_shape": str(prediction.shape),
        "prediction_dtype": str(prediction.dtype),
        "finite": bool(np.isfinite(prediction).all()),
        "prediction_min": float(np.nanmin(prediction)),
        "prediction_max": float(np.nanmax(prediction)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m2_manifest_path = M2_DIR / "candidate_manifest.json"
    m3_manifest_path = M3_DIR / "candidate_manifest.json"
    m2_manifest = load_json(m2_manifest_path)
    m3_manifest = load_json(m3_manifest_path)
    m3_cv_path = M3_DIR / "nested_cv_candidate_metrics.csv"
    m3_outer_fold_count = int(pd.read_csv(m3_cv_path)["fold"].nunique())
    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Configured dataset is unavailable")
    dataset_hash = sha256(dataset_path)
    df = load_dataset()
    split = load_json(SPLIT_INDICES_PATH)

    manifest_comparisons = {
        "dataset_sha256_equal_and_current": (
            m2_manifest["dataset_sha256"] == m3_manifest["dataset_sha256"] == dataset_hash == EXPECTED_DATASET_SHA256
        ),
        "dataset_shape_equal": m2_manifest["dataset_shape"] == m3_manifest["dataset_shape"] == list(df.shape),
        "seed_equal": m2_manifest["seed"] == m3_manifest["seed"] == RANDOM_STATE,
        "split_path_equal": m2_manifest["split_path"] == m3_manifest["split_path"] == SPLIT_INDICES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_definition_equal": m2_manifest["split_definition"] == m3_manifest["split_definition"],
        "saved_split_indices_equal": (
            m2_manifest["split_definition"]["train_indices"] == split["train_index"]
            and m2_manifest["split_definition"]["test_indices"] == split["test_index"]
            and m3_manifest["split_definition"]["train_indices"] == split["train_index"]
            and m3_manifest["split_definition"]["test_indices"] == split["test_index"]
        ),
        "outer_and_inner_cv_equal": (
            m2_manifest["outer_folds"] == m3_outer_fold_count
            and m2_manifest["training_stacking"]["inner_oof_folds"] == m3_manifest["training_stacking"]["oof_folds"]
            and m2_manifest["training_stacking"]["random_state"] == m3_manifest["training_stacking"]["random_state"]
        ),
        "selected_upstream_algorithms_equal": (
            m2_manifest["upstream_selection"]["selected_algorithms"]
            == m3_manifest["model1_selection"]["selected_algorithms"]
        ),
        "package_versions_equal": m2_manifest["package_versions"] == m3_manifest["package_versions"],
    }
    m2_features = m2_manifest["features"]["full_stack_order"]
    m3_features = m3_manifest["feature_order"]
    manifest_comparisons["full_stack_features_equal"] = m2_features == m3_features == MODEL1_FEATURES + PROCESS_TARGETS
    manifest_comparisons["customer_features_equal"] = (
        m2_manifest["features"]["customer_product_order"]
        == m3_manifest["customer_feature_order"]
        == MODEL1_FEATURES
    )
    manifest_comparisons["process_features_equal"] = (
        m2_manifest["features"]["process_parameter_order"]
        == m3_manifest["process_feature_order"]
        == PROCESS_TARGETS
    )
    manifest_comparisons["both_targets_correct"] = (
        m2_manifest["downstream"]["target"] == EFFICIENCY_TARGET
        and m3_manifest["downstream"]["target"] == SCRAP_TARGET
    )
    if not all(manifest_comparisons.values()):
        raise RuntimeError(f"Candidate manifests disagree with each other or the current split: {manifest_comparisons}")

    m2_paths, m2_downstream_path, m2_models, m2_downstream = load_bundle(M2_DIR, m2_manifest, "model2")
    m3_paths, m3_downstream_path, m3_models, m3_downstream = load_bundle(M3_DIR, m3_manifest, "model3")

    artifact_hash_rows = []
    for target in PROCESS_TARGETS:
        artifact_hash_rows.append(
            {
                "artifact_group": "shared_upstream_candidate",
                "target": target,
                "model2_path": m2_paths[target].relative_to(PROJECT_ROOT).as_posix(),
                "model3_path": m3_paths[target].relative_to(PROJECT_ROOT).as_posix(),
                "model2_sha256": sha256(m2_paths[target]),
                "model3_sha256": sha256(m3_paths[target]),
                "byte_identical": sha256(m2_paths[target]) == sha256(m3_paths[target]),
            }
        )
    for label, path in (
        ("model2_downstream_candidate", m2_downstream_path),
        ("model3_downstream_candidate", m3_downstream_path),
        ("model2_manifest", m2_manifest_path),
        ("model3_manifest", m3_manifest_path),
        ("dataset", dataset_path),
        ("saved_split", SPLIT_INDICES_PATH),
    ):
        artifact_hash_rows.append(
            {
                "artifact_group": label,
                "target": "",
                "model2_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "model3_path": "",
                "model2_sha256": sha256(path),
                "model3_sha256": "",
                "byte_identical": "",
            }
        )
    pd.DataFrame(artifact_hash_rows).to_csv(OUTPUT_DIR / "artifact_hashes.csv", index=False)

    expected_m1 = MODEL1_FEATURES
    for target in PROCESS_TARGETS:
        check_inference_pipeline(m2_models[target], expected_m1, f"Model2 upstream {target}")
        check_inference_pipeline(m3_models[target], expected_m1, f"Model3 upstream {target}")
    m2_downstream_features = check_inference_pipeline(m2_downstream, STACKED_FEATURES, "Model2 downstream")
    m3_downstream_features = check_inference_pipeline(m3_downstream, STACKED_FEATURES, "Model3 downstream")
    encoder_category_checks = {
        "model2_downstream_has_Braid_Type_None": braid_none_is_encoded(m2_downstream),
        "model3_downstream_has_Braid_Type_None": braid_none_is_encoded(m3_downstream),
        "model2_upstream_all_have_Braid_Type_None": all(
            braid_none_is_encoded(pipeline) for pipeline in m2_models.values()
        ),
        "model3_upstream_all_have_Braid_Type_None": all(
            braid_none_is_encoded(pipeline) for pipeline in m3_models.values()
        ),
    }

    test_frame = df.loc[split["test_index"]].copy()
    # Use diverse actual synthetic customer rows and explicitly include a None braid row.
    normal = test_frame.head(4)[MODEL1_FEATURES].copy()
    none_rows = test_frame.loc[test_frame["Braid_Type"].astype(str) == "None"]
    if none_rows.empty:
        raise RuntimeError("Saved test partition contains no Braid_Type='None' probe row")
    none_probe = none_rows.iloc[[0]][MODEL1_FEATURES].copy()
    normal_with_none = pd.concat([normal, none_probe], axis=0)
    unseen_probe = normal.iloc[[0]].copy()
    unseen_probe.loc[unseen_probe.index[0], "Braid_Type"] = "__unseen_category_probe__"

    infer_rows = []
    behavior_rows = []
    main_stacks = {}
    for bundle_name, models, downstream, target, lower, upper in (
        ("model2_candidate", m2_models, m2_downstream, EFFICIENCY_TARGET, 0.0, 100.0),
        ("model3_candidate", m3_models, m3_downstream, SCRAP_TARGET, 0.0, 1.0),
    ):
        for scenario, inputs in (("normal_and_None", normal_with_none), ("unseen_category", unseen_probe)):
            stack, prediction = predict_bundle(inputs, models, downstream)
            if scenario == "normal_and_None":
                main_stacks[bundle_name] = stack
            outside = (prediction < lower) | (prediction > upper)
            infer_rows.append(
                {
                    "bundle": bundle_name,
                    "scenario": scenario,
                    "rows": len(inputs),
                    "customer_input_columns": len(inputs.columns),
                    "stack_columns": len(stack.columns),
                    "output_shape": str(prediction.shape),
                    "output_dtype": str(prediction.dtype),
                    "target": target,
                    "unit": "percentage_points" if target == EFFICIENCY_TARGET else "fraction",
                    "prediction_min": float(prediction.min()),
                    "prediction_max": float(prediction.max()),
                    "outside_target_bounds_count": int(outside.sum()),
                    "finite": bool(np.isfinite(prediction).all()),
                }
            )
            for row_index, estimate in zip(inputs.index, prediction):
                behavior_rows.append(
                    {
                        "bundle": bundle_name,
                        "scenario": scenario,
                        "record_id": int(df.loc[row_index, "Record_ID"]),
                        "target": target,
                        "prediction": float(estimate),
                    }
                )

    # Independent inference through both upstream bundles should be equal if the
    # downstream candidates really share compatible Model 1 artifacts.
    common_probe = test_frame[MODEL1_FEATURES].copy()
    m2_stack_full, _ = predict_bundle(common_probe, m2_models, m2_downstream)
    m3_stack_full, _ = predict_bundle(common_probe, m3_models, m3_downstream)
    upstream_parity = {}
    parity_rows = []
    for target in PROCESS_TARGETS:
        left = m2_stack_full[target].to_numpy(dtype=float)
        right = m3_stack_full[target].to_numpy(dtype=float)
        equal = bool(np.allclose(left, right, rtol=1e-12, atol=1e-12))
        upstream_parity[target] = {
            "all_saved_test_rows_equal_within_1e-12": equal,
            "max_abs_prediction_difference": float(np.max(np.abs(left - right))),
        }
        parity_rows.append({"target": target, **upstream_parity[target]})
    pd.DataFrame(parity_rows).to_csv(OUTPUT_DIR / "upstream_prediction_parity.csv", index=False)
    pd.DataFrame(infer_rows).to_csv(OUTPUT_DIR / "end_to_end_inference_results.csv", index=False)

    process_bound_rows = []
    for target, (lower, upper) in PROCESS_BOUNDS.items():
        values = m2_stack_full[target].to_numpy(dtype=float)
        process_bound_rows.append(
            {
                "target": target,
                "synthetic_generator_bound_low": lower,
                "synthetic_generator_bound_high": upper,
                "saved_test_prediction_min": float(values.min()),
                "saved_test_prediction_max": float(values.max()),
                "below_bound_count": int(np.sum(values < lower)),
                "above_bound_count": int(np.sum(values > upper)),
                "finite": bool(np.isfinite(values).all()),
            }
        )
    pd.DataFrame(process_bound_rows).to_csv(
        OUTPUT_DIR / "process_prediction_bounds.csv", index=False
    )

    # Verify candidate downstream test metrics against the persisted candidate report.
    metric_rows = []
    m2_test = M2_DIR / "test_comparison.csv"
    m3_test = M3_DIR / "test_comparison.csv"
    m2_report = pd.read_csv(m2_test).set_index("model").loc["corrected_candidate_full_stack_RF"]
    m3_report = pd.read_csv(m3_test).set_index("model").loc["corrected_candidate_full_stack_RF"]
    for bundle_name, models, downstream, target, report, mae_col, rmse_col in (
        ("model2_candidate", m2_models, m2_downstream, EFFICIENCY_TARGET, m2_report, "MAE_percentage_points", "RMSE_percentage_points"),
        ("model3_candidate", m3_models, m3_downstream, SCRAP_TARGET, m3_report, "MAE_fraction", "RMSE_fraction"),
    ):
        stack, prediction = predict_bundle(test_frame[MODEL1_FEATURES], models, downstream)
        actual = test_frame[target].to_numpy(dtype=float)
        metrics = prediction_metrics(actual, prediction)
        metric_rows.append(
            {
                "bundle": bundle_name,
                "target": target,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "saved_report_MAE": float(report[mae_col]),
                "saved_report_RMSE": float(report[rmse_col]),
                "saved_report_R2": float(report["R2"]),
                "metrics_match_report_within_1e-10": bool(
                    np.allclose(
                        [metrics["MAE"], metrics["RMSE"], metrics["R2"]],
                        [report[mae_col], report[rmse_col], report["R2"]],
                        rtol=1e-10,
                        atol=1e-10,
                    )
                ),
                "feature_order_matches_manifest": list(stack.columns) == (
                    m2_manifest["features"]["full_stack_order"]
                    if bundle_name == "model2_candidate"
                    else m3_manifest["feature_order"]
                ),
            }
        )
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "candidate_metric_reproduction.csv", index=False)

    # Missing required-column and missing-value behavior is captured, not adapted.
    missing_results = []
    missing_customer = normal.iloc[:1].drop(columns=[MODEL1_FEATURES[0]])
    for target, model in m2_models.items():
        missing_results.append(
            expect_missing_error(
                lambda model=model: model.predict(missing_customer),
                f"Model 1 missing column {MODEL1_FEATURES[0]} ({target})",
            )
        )
        break
    missing_results.append(
        expect_missing_error(
            lambda: m2_downstream.predict(main_stacks["model2_candidate"].drop(columns=[PROCESS_TARGETS[0]])),
            "Model2 missing process feature",
        )
    )
    missing_results.append(
        expect_missing_error(
            lambda: m3_downstream.predict(main_stacks["model3_candidate"].drop(columns=[PROCESS_TARGETS[0]])),
            "Model3 missing process feature",
        )
    )
    nan_customer = normal.iloc[:1].copy()
    nan_customer.loc[nan_customer.index[0], CUSTOMER_NUMERIC_FEATURES[0]] = np.nan
    missing_results.append(
        expect_missing_error(
            lambda: m2_models[PROCESS_TARGETS[0]].predict(nan_customer),
            "Model1 NaN numeric input",
        )
    )
    missing_results[-1]["case"] = "Model1 NaN numeric input"
    nan_probe_rows = []
    for bundle_name, models, downstream, target in (
        ("model2_candidate", m2_models, m2_downstream, EFFICIENCY_TARGET),
        ("model3_candidate", m3_models, m3_downstream, SCRAP_TARGET),
    ):
        numeric_nan = normal.iloc[:1].copy()
        numeric_nan.loc[numeric_nan.index[0], CUSTOMER_NUMERIC_FEATURES[0]] = np.nan
        categorical_nan = normal.iloc[:1].copy()
        categorical_nan.loc[categorical_nan.index[0], "Braid_Type"] = np.nan
        for value_label, probe in (("numeric_nan", numeric_nan), ("categorical_nan", categorical_nan)):
            nan_probe_rows.append(
                capture_optional_value_behavior(
                    lambda probe=probe, models=models, downstream=downstream: predict_bundle(
                        probe, models, downstream
                    )[1],
                    f"{bundle_name} end_to_end {value_label}",
                )
            )
    pd.DataFrame(missing_results).to_csv(OUTPUT_DIR / "invalid_input_behavior.csv", index=False)
    pd.DataFrame(nan_probe_rows).to_csv(OUTPUT_DIR / "missing_value_inference_behavior.csv", index=False)
    pd.DataFrame(behavior_rows).to_csv(OUTPUT_DIR / "inference_probe_predictions.csv", index=False)

    # Read-only provenance hashes for the existing production artifacts/metadata.
    production_paths = [
        PROJECT_ROOT / "models/model_metadata.json",
        PROJECT_ROOT / "models/data_split.json",
        *[(PROJECT_ROOT / "models/process" / filename) for filename in PROCESS_MODEL_FILENAMES.values()],
        PROJECT_ROOT / "models/efficiency/efficiency_model.pkl",
        PROJECT_ROOT / "models/quality/scrap_rate_model.pkl",
    ]
    production_hashes = [
        {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256_at_audit": sha256(path)}
        for path in production_paths
        if path.exists()
    ]

    audit_summary = {
        "experiment": "candidate_model_bundle_integration_audit",
        "dataset_sha256": dataset_hash,
        "dataset_shape": list(df.shape),
        "model2_manifest_path": m2_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "model3_manifest_path": m3_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_compatibility": manifest_comparisons,
        "upstream_artifact_byte_identical_count": sum(row["byte_identical"] is True for row in artifact_hash_rows if row["artifact_group"] == "shared_upstream_candidate"),
        "upstream_artifact_count": len(PROCESS_TARGETS),
        "upstream_prediction_parity": upstream_parity,
        "model2_downstream_actual_features": m2_downstream_features,
        "model3_downstream_actual_features": m3_downstream_features,
        "customer_feature_order": MODEL1_FEATURES,
        "process_feature_order": PROCESS_TARGETS,
        "process_units": m2_manifest["features"]["units"],
        "target_units": {
            EFFICIENCY_TARGET: m2_manifest["downstream"]["target_unit"],
            SCRAP_TARGET: "fraction",
        },
        "categorical_preprocessing": {
            "Braid_Type_None_present_in_dataset": bool((df["Braid_Type"].astype(str) == "None").any()),
            "fitted_encoder_category_checks": encoder_category_checks,
            "m2_handle_unknown": m2_downstream.named_steps["preprocess"].named_transformers_["cat"].handle_unknown,
            "m3_handle_unknown": m3_downstream.named_steps["preprocess"].named_transformers_["cat"].handle_unknown,
        },
        "process_prediction_bounds_are_synthetic_generator_bounds_not_factory_limits": PROCESS_BOUNDS,
        "process_prediction_bound_checks": process_bound_rows,
        "end_to_end_inference_probe_results": infer_rows,
        "candidate_metrics_reproduced": metric_rows,
        "missing_input_behavior": missing_results,
        "production_artifact_hashes_at_audit": production_hashes,
        "production_files_modified_by_audit": False,
        "note": "Compatibility checks on synthetic data only; successful inference is not evidence of real factory performance.",
    }
    (OUTPUT_DIR / "bundle_audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "manifest_compatibility": manifest_comparisons,
        "upstream_byte_identical": audit_summary["upstream_artifact_byte_identical_count"],
        "upstream_prediction_parity_all": all(item["all_saved_test_rows_equal_within_1e-12"] for item in upstream_parity.values()),
        "candidate_metric_reproduced": [row["metrics_match_report_within_1e-10"] for row in metric_rows],
        "audit_output": str(OUTPUT_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
