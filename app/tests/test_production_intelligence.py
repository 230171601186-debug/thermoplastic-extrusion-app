from __future__ import annotations

import unittest

import pandas as pd

from app.production_intelligence import (
    append_prediction_history,
    audit_downtime_records,
    history_frame,
)


class ProductionIntelligenceTests(unittest.TestCase):
    def test_history_is_newest_first_bounded_and_does_not_mutate_prior_entries(self):
        prior = [{"Generated_at_UTC": "older"}]
        result = append_prediction_history(
            prior,
            {"Material_Type": "PA12"},
            {"Scrap_Rate": 0.05},
            "2026-11-13T10:00:00+00:00",
            max_entries=2,
        )
        self.assertEqual(len(prior), 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["Input_Material_Type"], "PA12")
        self.assertEqual(result[0]["Prediction_Scrap_Rate"], 0.05)
        self.assertEqual(history_frame(result).shape[0], 2)

    def test_history_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            append_prediction_history([], {}, {}, "now", max_entries=0)

    def test_downtime_audit_reports_machine_and_fault_summaries(self):
        frame = pd.DataFrame(
            {
                "Timestamp": ["2026-11-01T10:00:00Z", "2026-11-01T11:00:00Z"],
                "Machine_ID": ["EXT-01", "EXT-01"],
                "Downtime_min": [12, 18],
                "Fault_Code": ["F01", "F02"],
            }
        )
        report = audit_downtime_records(frame)
        self.assertTrue(report["passed"])
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["machine_count"], 1)
        self.assertEqual(report["total_downtime_min"], 30.0)
        self.assertEqual(report["by_fault"].shape[0], 2)
        self.assertEqual(report["by_day"].shape[0], 1)
        self.assertEqual(float(report["by_day"]["Total_downtime_min"].iloc[0]), 30.0)

    def test_downtime_audit_fails_closed_for_missing_or_invalid_data(self):
        missing = audit_downtime_records(pd.DataFrame({"Machine_ID": ["EXT-01"]}))
        self.assertFalse(missing["passed"])
        self.assertTrue(any("Missing required columns" in item for item in missing["issues"]))

        invalid = audit_downtime_records(
            pd.DataFrame(
                {
                    "Timestamp": ["not-a-date"],
                    "Machine_ID": ["EXT-01"],
                    "Downtime_min": [-4],
                    "Fault_Code": ["F01"],
                }
            )
        )
        self.assertFalse(invalid["passed"])
        self.assertTrue(any("Timestamp" in item for item in invalid["issues"]))
        self.assertTrue(any("negative" in item for item in invalid["issues"]))

        missing_machine = audit_downtime_records(
            pd.DataFrame(
                {
                    "Timestamp": ["2026-11-01T10:00:00Z"],
                    "Machine_ID": [None],
                    "Downtime_min": [5],
                    "Fault_Code": ["F01"],
                }
            )
        )
        self.assertFalse(missing_machine["passed"])
        self.assertTrue(any("Machine_ID" in item for item in missing_machine["issues"]))


if __name__ == "__main__":
    unittest.main()
