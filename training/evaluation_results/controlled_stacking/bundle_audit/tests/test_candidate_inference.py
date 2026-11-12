from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
AUDIT_DIR = TEST_DIR.parent
ROOT = Path(__file__).resolve().parents[5]
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from candidate_inference import (  # noqa: E402
    CUSTOMER_FEATURES,
    PROCESS_OUTPUT_BOUNDS,
    PROCESS_TARGETS,
    STACKED_FEATURES,
    CandidateInferenceBundle,
    InferenceValidationError,
    _DownstreamBundle,
)
from config import (  # noqa: E402
    DATASET_CANDIDATES,
    MODEL1_FEATURES,
    PROJECT_ROOT,
    PROCESS_MODEL_FILENAMES,
)
from preprocessing import load_dataset  # noqa: E402


class ConstantPredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        if np.isscalar(self.value):
            return np.full(len(frame), self.value)
        return np.asarray(self.value)


class CandidateInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CandidateInferenceBundle.load()
        cls.valid = {
            "Material_Type": "PA12",
            "Material_Grade": "G1",
            "Tube_Type": "Industrial",
            "Braid_Type": "Textile",
            "Target_Tube_OD_mm": 10.0,
            "Target_Tube_Wall_mm": 2.0,
            "Target_Tube_ID_mm": 6.0,
            "Customer_Required_Length_m": 5000.0,
        }
        cls.none_braid = {**cls.valid, "Braid_Type": "None"}
        cls.data_path = next(path for path in DATASET_CANDIDATES if path.exists())

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _frame_with_stack(self) -> pd.DataFrame:
        customer, _ = self.bundle.validate_customer_inputs(self.valid)
        stack = customer.copy()
        for target in PROCESS_TARGETS:
            stack[target] = [sum(PROCESS_OUTPUT_BOUNDS[target]) / 2.0]
        return stack

    def test_valid_normal_input_returns_named_finite_outputs(self):
        result = self.bundle.predict(self.valid)
        self.assertEqual(list(result.predictions.columns), PROCESS_TARGETS + [
            "Operational_Efficiency_pct", "Scrap_Rate"
        ])
        self.assertEqual(result.predictions.shape, (1, 11))
        self.assertTrue(all(np.issubdtype(dtype, np.number) for dtype in result.predictions.dtypes))
        self.assertEqual(result.units["Operational_Efficiency_pct"], "percentage points")
        self.assertEqual(result.units["Scrap_Rate"], "fraction")
        self.assertEqual(result.units["Target_Tube_OD_mm"], "mm")
        self.assertTrue(np.isfinite(result.predictions.to_numpy()).all())
        self.assertGreaterEqual(result.predictions["Operational_Efficiency_pct"].iloc[0], 0)
        self.assertLessEqual(result.predictions["Operational_Efficiency_pct"].iloc[0], 100)
        self.assertGreaterEqual(result.predictions["Scrap_Rate"].iloc[0], 0)
        self.assertLessEqual(result.predictions["Scrap_Rate"].iloc[0], 1)

    def test_literal_none_braid_is_valid(self):
        result = self.bundle.predict(self.none_braid)
        self.assertEqual(result.predictions.shape, (1, 11))
        self.assertNotIn("unseen Braid_Type", " ".join(result.warnings))

    def test_unseen_nonblank_category_is_accepted_and_reported(self):
        values = {**self.valid, "Braid_Type": "Novel-Braid"}
        result = self.bundle.predict(values)
        self.assertEqual(result.predictions.shape, (1, 11))
        self.assertTrue(any("unseen Braid_Type" in warning for warning in result.warnings))

    def test_each_required_field_missing_is_rejected(self):
        for field in CUSTOMER_FEATURES:
            values = dict(self.valid)
            del values[field]
            with self.subTest(field=field), self.assertRaisesRegex(InferenceValidationError, field):
                self.bundle.predict(values)

    def test_extra_field_is_rejected(self):
        values = {**self.valid, "Operational_Efficiency_pct": 90.0}
        with self.assertRaisesRegex(InferenceValidationError, "unexpected input field"):
            self.bundle.predict(values)

    def test_none_pandas_missing_and_blank_values_are_rejected(self):
        for field in CUSTOMER_FEATURES:
            for missing in (None, pd.NA, np.nan, "   "):
                values = dict(self.valid)
                values[field] = missing
                with self.subTest(field=field, missing=repr(missing)), self.assertRaisesRegex(
                    InferenceValidationError, field
                ):
                    self.bundle.predict(values)

    def test_numeric_nonfinite_and_nonnumeric_values_are_rejected(self):
        for field in ("Target_Tube_OD_mm", "Target_Tube_Wall_mm", "Target_Tube_ID_mm", "Customer_Required_Length_m"):
            for value in (np.nan, np.inf, -np.inf, "not-a-number"):
                values = dict(self.valid)
                values[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    InferenceValidationError, field
                ):
                    self.bundle.predict(values)

    def test_nonpositive_od_wall_and_length_are_rejected(self):
        for field in ("Target_Tube_OD_mm", "Target_Tube_Wall_mm", "Customer_Required_Length_m"):
            for value in (0, -1):
                values = dict(self.valid)
                values[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    InferenceValidationError, field
                ):
                    self.bundle.predict(values)

    def test_invalid_id_and_existing_geometry_rules_are_rejected(self):
        for value in (-1, 0, 0.5):
            values = dict(self.valid)
            values["Target_Tube_ID_mm"] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                InferenceValidationError, "Target_Tube_ID_mm"
            ):
                self.bundle.predict(values)
        values = {**self.valid, "Target_Tube_ID_mm": 5.0}
        with self.assertRaisesRegex(InferenceValidationError, "geometry"):
            self.bundle.predict(values)

    def test_reordered_or_missing_process_stack_is_rejected(self):
        stack = self._frame_with_stack()
        self.assertEqual(list(stack.columns), STACKED_FEATURES)
        with self.assertRaisesRegex(InferenceValidationError, "feature columns or order"):
            self.bundle._validate_stack(stack.drop(columns=[PROCESS_TARGETS[0]]), 1)
        reordered = stack[MODEL1_FEATURES + list(reversed(PROCESS_TARGETS))]
        with self.assertRaisesRegex(InferenceValidationError, "feature columns or order"):
            self.bundle._validate_stack(reordered, 1)

    def test_nan_or_infinite_model1_outputs_are_rejected(self):
        original = self.bundle.upstream_models[PROCESS_TARGETS[0]]
        for bad_value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad_value):
                self.bundle.upstream_models[PROCESS_TARGETS[0]] = ConstantPredictor(bad_value)
                with self.assertRaisesRegex(InferenceValidationError, PROCESS_TARGETS[0]):
                    self.bundle.predict(self.valid)
        self.bundle.upstream_models[PROCESS_TARGETS[0]] = original

    def test_model1_outputs_outside_existing_synthetic_bounds_are_rejected(self):
        target = PROCESS_TARGETS[0]
        original = self.bundle.upstream_models[target]
        self.bundle.upstream_models[target] = ConstantPredictor(PROCESS_OUTPUT_BOUNDS[target][1] + 1)
        try:
            with self.assertRaisesRegex(InferenceValidationError, "synthetic bounds"):
                self.bundle.predict(self.valid)
        finally:
            self.bundle.upstream_models[target] = original

    def test_missing_model1_pipeline_is_rejected_by_target_name(self):
        target = PROCESS_TARGETS[0]
        original = self.bundle.upstream_models.pop(target)
        try:
            with self.assertRaisesRegex(InferenceValidationError, target):
                self.bundle.predict(self.valid)
        finally:
            self.bundle.upstream_models[target] = original

    def test_downstream_out_of_bounds_and_wrong_shapes_are_rejected(self):
        original2 = self.bundle.model2.pipeline
        original3 = self.bundle.model3.pipeline
        try:
            self.bundle.model2.pipeline = ConstantPredictor(100.01)
            with self.assertRaisesRegex(InferenceValidationError, "Operational_Efficiency_pct"):
                self.bundle.predict(self.valid)
            self.bundle.model2.pipeline = ConstantPredictor(np.nan)
            with self.assertRaisesRegex(InferenceValidationError, "Operational_Efficiency_pct"):
                self.bundle.predict(self.valid)
            self.bundle.model2.pipeline = ConstantPredictor([75.0, 76.0])
            with self.assertRaisesRegex(InferenceValidationError, "expected 1 predictions"):
                self.bundle.predict(self.valid)
            self.bundle.model2.pipeline = original2
            self.bundle.model3.pipeline = ConstantPredictor(1.01)
            with self.assertRaisesRegex(InferenceValidationError, "Scrap_Rate"):
                self.bundle.predict(self.valid)
        finally:
            self.bundle.model2.pipeline = original2
            self.bundle.model3.pipeline = original3

    def test_save_reload_predictions_are_consistent(self):
        baseline = self.bundle.predict(self.valid).predictions
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            upstream = {}
            for target, pipeline in self.bundle.upstream_models.items():
                path = temp / PROCESS_MODEL_FILENAMES[target]
                joblib.dump(pipeline, path)
                upstream[target] = joblib.load(path)
            path2 = temp / "efficiency.pkl"
            path3 = temp / "scrap.pkl"
            joblib.dump(self.bundle.model2.pipeline, path2)
            joblib.dump(self.bundle.model3.pipeline, path3)
            clone_bundle = CandidateInferenceBundle(
                upstream,
                _DownstreamBundle("Operational_Efficiency_pct", joblib.load(path2), self.bundle.model2.manifest),
                _DownstreamBundle("Scrap_Rate", joblib.load(path3), self.bundle.model3.manifest),
                self.bundle.category_values,
            )
            reloaded = clone_bundle.predict(self.valid).predictions
        np.testing.assert_allclose(baseline.to_numpy(), reloaded.to_numpy(), rtol=1e-12, atol=1e-12)

    def test_production_artifacts_and_dataset_are_read_only_during_inference(self):
        protected = [
            self.data_path,
            PROJECT_ROOT / "models/model_metadata.json",
            PROJECT_ROOT / "models/data_split.json",
            *sorted((PROJECT_ROOT / "models").glob("**/*.pkl")),
        ]
        before = {path: self._hash(path) for path in protected if path.exists()}
        self.bundle.predict(self.valid)
        after = {path: self._hash(path) for path in before}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
