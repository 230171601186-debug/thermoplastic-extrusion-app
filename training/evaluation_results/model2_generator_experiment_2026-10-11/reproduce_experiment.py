"""Reproduce the baseline and run an isolated Model 2 generator experiment."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from shutil import copy2

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = ROOT / "training"
for import_path in (ROOT, TRAINING_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_DIR = ROOT / "training" / "evaluation_results" / "model2_generator_experiment_2026-10-11"
BASELINE_SNAPSHOT = RUN_DIR / "baseline_snapshot.json"
PRODUCTION_DATASET = ROOT / "dataset" / "Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
PRODUCTION_MODELS_DIR = ROOT / "models"
CONTROLLED_RESULTS_DIR = ROOT / "training" / "evaluation_results" / "controlled_stacking"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_production_hashes(snapshot: dict) -> None:
    if sha256(PRODUCTION_DATASET) != snapshot["dataset_sha256"]:
        raise RuntimeError("Production dataset hash differs from the captured baseline")
    for relative_path, expected_hash in snapshot["artifact_hashes"].items():
        path = ROOT / Path(relative_path)
        if sha256(path) != expected_hash:
            raise RuntimeError(f"Production artifact changed since baseline capture: {relative_path}")


def reproduce_saved_model_test(run_dir: Path) -> dict[str, float]:
    from config import EFFICIENCY_TARGET, METADATA_PATH, MODEL1_FEATURES
    from evaluate import regression_metrics
    from preprocessing import (
        apply_saved_split,
        attach_predicted_process,
        load_dataset,
        load_process_pipelines,
    )

    frame = load_dataset()
    train_frame, test_frame = apply_saved_split(frame)
    process_models = load_process_pipelines()
    test_features = attach_predicted_process(
        test_frame[MODEL1_FEATURES], process_models
    )
    model_path = PRODUCTION_MODELS_DIR / "efficiency" / "efficiency_model.pkl"
    model = joblib.load(model_path)
    actual = test_frame[EFFICIENCY_TARGET].to_numpy(dtype=float)
    predicted = np.asarray(model.predict(test_features), dtype=float)
    metrics = regression_metrics(actual, predicted)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    saved_metrics = metadata["model2"]["test_metrics"]
    if not all(
        np.isclose(metrics[name], saved_metrics[name], atol=1e-6)
        for name in ("MAE", "RMSE", "R2")
    ):
        raise RuntimeError(
            f"Reproduced production Model 2 test metrics differ from metadata: "
            f"{metrics} vs {saved_metrics}"
        )

    pd.DataFrame(
        [
            {
                "evaluation": "saved_production_Model2_reproduced_test",
                "model": metadata["model2"]["selected_algorithm"],
                **metrics,
            }
        ]
    ).to_csv(run_dir / "production_model2_reproduced_test_metrics.csv", index=False)
    pd.DataFrame(
        {
            "record_id": test_frame["Record_ID"].to_numpy(),
            "actual": actual,
            "predicted": predicted,
            "residual_pred_minus_actual": predicted - actual,
            "absolute_error": np.abs(predicted - actual),
        }
    ).to_csv(run_dir / "production_model2_reproduced_test_predictions.csv", index=False)
    return metrics


def run() -> None:
    if not BASELINE_SNAPSHOT.exists():
        capture = importlib.import_module(
            "training.evaluation_results.model2_generator_experiment.capture_baseline"
        )
        capture.main()
    if not BASELINE_SNAPSHOT.exists():
        raise FileNotFoundError(f"Baseline capture did not create {BASELINE_SNAPSHOT}")
    baseline = json.loads(BASELINE_SNAPSHOT.read_text(encoding="utf-8"))
    verify_production_hashes(baseline)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for artifact_name in (
        "experimental_generator.py",
        "run_controlled_experiment.py",
        "capture_baseline.py",
        "reproduce_experiment.py",
    ):
        copy2(EXPERIMENT_DIR / artifact_name, RUN_DIR / artifact_name)
    production_test_metrics = reproduce_saved_model_test(RUN_DIR)

    generator = importlib.import_module(
        "training.evaluation_results.model2_generator_experiment.experimental_generator"
    )
    experimental_dataset_path = RUN_DIR / "experimental_dataset.csv"
    generator.OUTPUT_PATH = experimental_dataset_path
    frame = generator.generate_dataset(seed=generator.SEED, row_count=generator.ROW_COUNT)
    second_generation = generator.generate_dataset(
        seed=generator.SEED, row_count=generator.ROW_COUNT
    )
    if not frame.equals(second_generation):
        raise RuntimeError("Fixed-seed dataset generation is not reproducible")
    frame.to_csv(experimental_dataset_path, index=False)

    reference = importlib.import_module(
        "training.evaluation_results.controlled_stacking.run_controlled_stacking"
    )
    baseline_nested_dir = RUN_DIR / "baseline_nested_reproduction"
    reference.OUTPUT_DIR = baseline_nested_dir
    reference.main()

    experiment_runner = importlib.import_module(
        "training.evaluation_results.model2_generator_experiment.run_controlled_experiment"
    )
    experiment_runner.OUT = RUN_DIR
    experiment_runner.OUTPUT_PATH = experimental_dataset_path
    experiment_runner.BASELINE_PATH = BASELINE_SNAPSHOT
    experiment_runner.BASELINE = baseline
    experiment_runner.BASELINE_NESTED_DIR = baseline_nested_dir
    experiment_runner.CONTROLLED_DIR = CONTROLLED_RESULTS_DIR
    experiment_runner.main()

    verify_production_hashes(baseline)
    run_record = {
        "production_dataset_sha256_before_and_after": sha256(PRODUCTION_DATASET),
        "production_artifact_hashes_verified_unchanged": len(baseline["artifact_hashes"]),
        "experimental_dataset_sha256": sha256(experimental_dataset_path),
        "experimental_dataset_path": experimental_dataset_path.relative_to(ROOT).as_posix(),
        "production_model2_reproduced_test_metrics": production_test_metrics,
        "baseline_nested_metrics_path": (
            baseline_nested_dir / "metrics_summary.csv"
        ).relative_to(ROOT).as_posix(),
        "experiment_metrics_path": (
            RUN_DIR / "experimental_model2_metric_summary.csv"
        ).relative_to(ROOT).as_posix(),
        "test_set_used_for_model_selection": False,
        "production_files_written": False,
    }
    (RUN_DIR / "reproduction_manifest.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_record, indent=2))


if __name__ == "__main__":
    run()
