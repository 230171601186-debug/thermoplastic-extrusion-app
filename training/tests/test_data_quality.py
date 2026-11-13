from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from training.config import (
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    EFFICIENCY_TARGET,
    PROCESS_TARGETS,
    SCRAP_TARGET,
)
from training.data_quality import audit_modeling_dataset, validate_modeling_dataset


def valid_frame() -> pd.DataFrame:
    row = {
        **{name: "None" if name == "Braid_Type" else "PA12" for name in CATEGORICAL_FEATURES},
        **{name: 1.0 for name in CUSTOMER_NUMERIC_FEATURES},
        **{name: 1.0 for name in PROCESS_TARGETS},
        EFFICIENCY_TARGET: 75.0,
        SCRAP_TARGET: 0.05,
    }
    row["Target_Tube_OD_mm"] = 10.0
    row["Target_Tube_Wall_mm"] = 2.0
    row["Target_Tube_ID_mm"] = 6.0
    return pd.DataFrame([row, row.copy()])


class DatasetQualityTests(unittest.TestCase):
    def test_valid_dataset_passes_without_mutating_source(self):
        frame = valid_frame()
        before = frame.copy(deep=True)
        report = validate_modeling_dataset(frame)
        self.assertTrue(report["passed"])
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["duplicate_rows_reported"], 1)
        pd.testing.assert_frame_equal(frame, before)

    def test_missing_required_column_fails_closed(self):
        frame = valid_frame().drop(columns=[PROCESS_TARGETS[0]])
        report = audit_modeling_dataset(frame)
        self.assertFalse(report["passed"])
        self.assertIn(PROCESS_TARGETS[0], report["missing_columns"])
        with self.assertRaisesRegex(ValueError, "failed quality validation"):
            validate_modeling_dataset(frame)

    def test_missing_and_non_numeric_values_are_reported(self):
        frame = valid_frame()
        frame.loc[0, CUSTOMER_NUMERIC_FEATURES[0]] = np.nan
        frame[PROCESS_TARGETS[0]] = frame[PROCESS_TARGETS[0]].astype(object)
        frame.loc[1, PROCESS_TARGETS[0]] = "not-a-number"
        report = audit_modeling_dataset(frame)
        self.assertFalse(report["passed"])
        self.assertEqual(report["missing_cells"][CUSTOMER_NUMERIC_FEATURES[0]], 1)
        self.assertEqual(report["non_numeric_values"][PROCESS_TARGETS[0]], 1)

    def test_blank_category_and_infinite_numeric_are_reported(self):
        frame = valid_frame()
        frame.loc[0, "Material_Type"] = " "
        frame.loc[1, CUSTOMER_NUMERIC_FEATURES[0]] = np.inf
        report = audit_modeling_dataset(frame)
        self.assertFalse(report["passed"])
        self.assertEqual(report["blank_categories"]["Material_Type"], 1)
        self.assertEqual(report["non_finite_values"][CUSTOMER_NUMERIC_FEATURES[0]], 1)


if __name__ == "__main__":
    unittest.main()
