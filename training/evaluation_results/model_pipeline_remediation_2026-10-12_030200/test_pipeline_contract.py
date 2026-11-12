from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from config import EFFICIENCY_TARGET, MODEL1_FEATURES, PROCESS_TARGETS, SCRAP_TARGET  # noqa: E402
from evaluation_results.controlled_stacking.bundle_audit.candidate_inference import (  # noqa: E402
    CandidateInferenceBundle,
)


class PipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = CandidateInferenceBundle.load()
        cls.input_values = {
            "Material_Type": "PA12",
            "Material_Grade": "G1",
            "Tube_Type": "Industrial",
            "Braid_Type": "None",
            "Target_Tube_OD_mm": 10.0,
            "Target_Tube_Wall_mm": 2.0,
            "Target_Tube_ID_mm": 6.0,
            "Customer_Required_Length_m": 5000.0,
        }

    def test_bundle_declares_three_prediction_stages_and_shared_process_artifacts(self):
        controlled = ROOT / "training" / "evaluation_results" / "controlled_stacking"
        m2 = json.loads((controlled / "model2_candidate" / "candidate_manifest.json").read_text())
        m3 = json.loads((controlled / "model3_candidate" / "candidate_manifest.json").read_text())
        self.assertEqual(list(self.input_values), MODEL1_FEATURES)
        self.assertEqual(set(m2["candidate_artifacts"]["model1"]), set(PROCESS_TARGETS))
        for target in PROCESS_TARGETS:
            model2_path = ROOT / m2["candidate_artifacts"]["model1"][target]
            model3_path = ROOT / m3["candidate_artifacts"]["model1"][target]
            self.assertEqual(
                hashlib.sha256(model2_path.read_bytes()).hexdigest(),
                hashlib.sha256(model3_path.read_bytes()).hexdigest(),
                target,
            )
        self.assertEqual(m2["downstream"]["target"], EFFICIENCY_TARGET)
        self.assertEqual(m3["downstream"]["target"], SCRAP_TARGET)

    def test_inference_returns_nine_process_and_two_downstream_values(self):
        result = self.bundle.predict(self.input_values)
        self.assertEqual(list(result.predictions.columns), PROCESS_TARGETS + [EFFICIENCY_TARGET, SCRAP_TARGET])
        self.assertTrue(np.isfinite(result.predictions.to_numpy()).all())
        self.assertGreaterEqual(result.predictions[EFFICIENCY_TARGET].iloc[0], 0)
        self.assertLessEqual(result.predictions[EFFICIENCY_TARGET].iloc[0], 100)
        self.assertGreaterEqual(result.predictions[SCRAP_TARGET].iloc[0], 0)
        self.assertLessEqual(result.predictions[SCRAP_TARGET].iloc[0], 1)
        self.assertEqual(result.units[EFFICIENCY_TARGET], "percentage points")
        self.assertEqual(result.units[SCRAP_TARGET], "fraction")

    def test_unseen_category_is_warning_and_not_silent(self):
        result = self.bundle.predict({**self.input_values, "Braid_Type": "Audit-Unseen-Braid"})
        self.assertTrue(any("unseen Braid_Type" in warning for warning in result.warnings))

    def test_saved_upstream_audit_excludes_outer_and_heldout_evaluation_rows(self):
        audit = pd.read_csv(BASE / "upstream_fold_audit.csv")
        outer = audit[audit["outer_partition"].eq("outer_cv")]
        saved_test = audit[audit["outer_partition"].eq("saved_test")]
        self.assertEqual(len(outer), 5 * 9 * 4)
        self.assertTrue(outer["outer_fit_rows"].eq(3200).all())
        self.assertTrue(outer["outer_evaluation_rows"].eq(800).all())
        self.assertTrue(outer["outer_fit_evaluation_index_overlap"].eq(0).all())
        self.assertEqual(len(saved_test), 9 * 4)
        self.assertTrue(saved_test["outer_fit_rows"].eq(4000).all())
        self.assertTrue(saved_test["outer_evaluation_rows"].eq(1000).all())
        self.assertTrue(saved_test["outer_fit_evaluation_index_overlap"].eq(0).all())


if __name__ == "__main__":
    unittest.main()
