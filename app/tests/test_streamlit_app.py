from __future__ import annotations

import csv
import io
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from app.streamlit_app import (
    OUTPUT_FIELDS,
    OUTPUT_UNITS,
    REPORT_DISCLAIMER,
    REPORT_FILENAME,
    PredictionOutputError,
    build_customer_inputs,
    build_prediction_report,
    build_process_chart_data,
    build_validation_summary,
    efficiency_percentage,
    execute_prediction,
    format_scrap_percentage,
    load_inference_bundle,
    scrap_percentage,
    validate_customer_inputs,
    validate_prediction_outputs,
)
from training.config import MODEL1_FEATURES
from training.evaluation_results.controlled_stacking.bundle_audit.candidate_inference import (
    CandidateInferenceBundle,
    InferenceValidationError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"


def valid_customer_inputs() -> dict[str, object]:
    return {
        "Material_Type": "PA12",
        "Material_Grade": "G1",
        "Tube_Type": "Industrial",
        "Braid_Type": "None",
        "Target_Tube_OD_mm": 10.0,
        "Target_Tube_Wall_mm": 2.0,
        "Target_Tube_ID_mm": 6.0,
        "Customer_Required_Length_m": 5000.0,
    }


def valid_categories() -> dict[str, set[str]]:
    return {
        "Material_Type": {"PA12"},
        "Material_Grade": {"G1"},
        "Tube_Type": {"Industrial"},
        "Braid_Type": {"None", "Textile"},
    }


def valid_inference_result() -> SimpleNamespace:
    values = {
        field: float(index + 1)
        for index, field in enumerate(OUTPUT_FIELDS[:9])
    }
    values["Operational_Efficiency_pct"] = 92.34
    values["Scrap_Rate"] = 0.0532
    return SimpleNamespace(
        predictions=pd.DataFrame([values]),
        units=dict(OUTPUT_UNITS),
        warnings=(),
    )


class StubBundle:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else valid_inference_result()
        self.error = error
        self.calls = 0
        self.received_inputs = None

    def predict(self, values):
        self.calls += 1
        self.received_inputs = dict(values)
        if self.error is not None:
            raise self.error
        return self.result


class StreamlitAppHelperTests(unittest.TestCase):
    def test_customer_values_map_to_exact_model_feature_order(self):
        values = build_customer_inputs(
            "PA12",
            "G1",
            "Industrial",
            "None",
            10.0,
            2.0,
            6.0,
            5000.0,
        )
        self.assertEqual(list(values), MODEL1_FEATURES)
        self.assertEqual(values["Braid_Type"], "None")
        self.assertEqual(values["Target_Tube_ID_mm"], 6.0)

    def test_scrap_fraction_converts_to_display_percentage(self):
        self.assertAlmostEqual(scrap_percentage(0.0532), 5.32)
        self.assertEqual(scrap_percentage(0.0), 0.0)
        self.assertEqual(format_scrap_percentage(0.0532), "5.32%")
        self.assertEqual(efficiency_percentage(92.34), "92.3%")

    def test_valid_customer_inputs_and_none_braid_pass_validation(self):
        self.assertEqual(
            validate_customer_inputs(valid_customer_inputs(), valid_categories()),
            (),
        )

    def test_invalid_and_inconsistent_tube_geometry_is_rejected(self):
        for invalid_values, expected in (
            ({"Target_Tube_ID_mm": 0.5}, "greater than 0.5 mm"),
            ({"Target_Tube_ID_mm": 5.0}, "geometry is inconsistent"),
            ({"Target_Tube_OD_mm": 10.25}, "geometry is inconsistent"),
        ):
            with self.subTest(invalid_values=invalid_values):
                values = {**valid_customer_inputs(), **invalid_values}
                errors = validate_customer_inputs(values, valid_categories())
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_missing_nonfinite_nonpositive_and_nonnumeric_inputs_are_rejected(self):
        missing = valid_customer_inputs()
        del missing["Target_Tube_OD_mm"]
        errors = validate_customer_inputs(missing, valid_categories())
        self.assertTrue(any("Target_Tube_OD_mm: a value is required" in error for error in errors))

        test_values = (
            (float("nan"), "must be finite"),
            (float("inf"), "must be finite"),
            (0.0, "greater than zero"),
            ("not-a-number", "finite numeric value"),
            (True, "finite numeric value"),
        )
        for value, expected in test_values:
            with self.subTest(value=value):
                values = {**valid_customer_inputs(), "Customer_Required_Length_m": value}
                errors = validate_customer_inputs(values, valid_categories())
                self.assertTrue(
                    any(
                        "Customer_Required_Length_m" in error and expected in error
                        for error in errors
                    ),
                    errors,
                )

    def test_categories_are_validated_without_removing_none(self):
        values = {**valid_customer_inputs(), "Braid_Type": "Unsupported"}
        errors = validate_customer_inputs(values, valid_categories())
        self.assertTrue(any("Braid_Type" in error for error in errors))
        self.assertEqual(
            validate_customer_inputs(valid_customer_inputs(), valid_categories()),
            (),
        )

    def test_prediction_output_contract_rejects_missing_malformed_and_wrong_scale_values(self):
        missing_field = valid_inference_result()
        missing_field.predictions = missing_field.predictions.drop(columns=["Scrap_Rate"])
        with self.assertRaisesRegex(PredictionOutputError, "missing Scrap_Rate"):
            validate_prediction_outputs(missing_field)

        malformed = valid_inference_result()
        malformed.predictions.loc[0, "Extruder_Temperature_C"] = float("nan")
        with self.assertRaisesRegex(PredictionOutputError, "must be finite"):
            validate_prediction_outputs(malformed)

        wrong_units = valid_inference_result()
        wrong_units.units["Scrap_Rate"] = "percentage points"
        with self.assertRaisesRegex(PredictionOutputError, "units are invalid"):
            validate_prediction_outputs(wrong_units)

        for target, invalid in (
            ("Operational_Efficiency_pct", 100.01),
            ("Scrap_Rate", 1.01),
        ):
            with self.subTest(target=target):
                result = valid_inference_result()
                result.predictions.loc[0, target] = invalid
                with self.assertRaises(PredictionOutputError):
                    validate_prediction_outputs(result)

    def test_validation_summary_uses_saved_held_out_candidate_rows(self):
        metadata = {
            "model1": {
                "targets": {
                    "Extruder_Temperature_C": {
                        "test_metrics": {"MAE": 3.7, "R2": 0.62}
                    }
                }
            }
        }
        model2 = pd.DataFrame(
            [
                {
                    "split": "saved_test_1000_rows",
                    "model": "customer_only_fixed_RF",
                    "MAE_percentage_points": 6.49,
                    "R2": 0.21,
                },
                {
                    "split": "saved_test_1000_rows",
                    "model": "corrected_candidate_full_stack_RF",
                    "MAE_percentage_points": 6.37,
                    "R2": 0.23,
                },
            ]
        )
        model3 = pd.DataFrame(
            [
                {
                    "split": "saved_test_1000_rows",
                    "model": "corrected_candidate_full_stack_RF",
                    "MAE_percentage_points": 2.09,
                    "R2": 0.35,
                }
            ]
        )

        summary = build_validation_summary(metadata, model2, model3)

        self.assertEqual(len(summary), 3)
        self.assertAlmostEqual(
            summary.loc[summary["Model"] == "Operational efficiency", "Held-out R²"].iloc[0],
            0.23,
        )
        self.assertAlmostEqual(
            summary.loc[summary["Model"] == "Scrap rate", "MAE"].iloc[0],
            2.09,
        )
        self.assertTrue((summary["Test rows"] == 1000).all())

    def test_process_chart_data_groups_shared_units_and_preserves_values(self):
        predictions = {
            field: float(index + 11)
            for index, field in enumerate(OUTPUT_FIELDS[:9])
        }
        charts = build_process_chart_data(predictions)
        self.assertEqual(
            [(title, unit) for title, unit, _ in charts],
            [
                ("Temperatures", "°C"),
                ("Pressures", "bar"),
                ("Linear speeds", "m/min"),
                ("Screw speed", "RPM"),
                ("Extrusion rate", "kg/hr"),
            ],
        )
        chart_features = []
        for _, unit, frame in charts:
            self.assertEqual(list(frame.columns), ["Process parameter", f"Predicted value ({unit})"])
            chart_features.extend(frame["Process parameter"].tolist())
            self.assertEqual(len(set(frame[f"Predicted value ({unit})"])), len(frame))
        self.assertEqual(len(chart_features), 9)
        self.assertEqual(len(set(chart_features)), 9)
        self.assertEqual(charts[0][2][f"Predicted value (°C)"].tolist(), [11.0, 12.0, 13.0])

    def test_csv_report_contains_inputs_outputs_units_timestamp_and_disclaimer(self):
        inputs = valid_customer_inputs()
        predictions = {
            field: float(index + 21)
            for index, field in enumerate(OUTPUT_FIELDS[:9])
        }
        predictions["Operational_Efficiency_pct"] = 87.654
        predictions["Scrap_Rate"] = 0.04123
        generated_at = "2026-10-12T08:30:45+00:00"

        report = build_prediction_report(inputs, predictions, generated_at)
        rows = list(csv.DictReader(io.StringIO(report)))
        by_field = {row["Field"]: row for row in rows}

        self.assertEqual(REPORT_FILENAME, "thermoplastic_extrusion_prediction_report.csv")
        self.assertEqual(len(rows), 21)
        self.assertEqual(by_field["Generated at (UTC)"]["Value"], generated_at)
        self.assertEqual(by_field["Disclaimer"]["Value"], REPORT_DISCLAIMER)
        for field in MODEL1_FEATURES:
            self.assertEqual(by_field[field]["Value"], str(inputs[field]))
        for field in OUTPUT_FIELDS[:9]:
            self.assertEqual(by_field[field]["Value"], str(predictions[field]))
            self.assertEqual(by_field[field]["Units"], OUTPUT_UNITS[field])
        self.assertEqual(by_field["Operational_Efficiency_pct"]["Value"], "87.654")
        self.assertEqual(by_field["Operational_Efficiency_pct"]["Units"], "%")
        self.assertEqual(by_field["Scrap_Rate"]["Value"], "4.12")
        self.assertEqual(by_field["Scrap_Rate"]["Units"], "%")

    def test_csv_report_uses_the_latest_success_values_not_fixture_defaults(self):
        inputs = {**valid_customer_inputs(), "Customer_Required_Length_m": 6400.0}
        predictions = {
            field: float(index + 31)
            for index, field in enumerate(OUTPUT_FIELDS[:9])
        }
        predictions["Operational_Efficiency_pct"] = 73.21
        predictions["Scrap_Rate"] = 0.0765

        report = build_prediction_report(
            inputs,
            predictions,
            "2026-10-12T08:31:00+00:00",
        )
        rows = {row["Field"]: row for row in csv.DictReader(io.StringIO(report))}
        self.assertEqual(rows["Customer_Required_Length_m"]["Value"], "6400.0")
        self.assertEqual(rows["Extruder_Temperature_C"]["Value"], str(predictions["Extruder_Temperature_C"]))
        self.assertEqual(rows["Operational_Efficiency_pct"]["Value"], "73.21")
        self.assertEqual(rows["Scrap_Rate"]["Value"], "7.65")

    def test_inference_is_skipped_for_invalid_input_and_none_braid_is_passed_unchanged(self):
        bundle = StubBundle()
        invalid = {**valid_customer_inputs(), "Target_Tube_ID_mm": 5.0}
        predictions, warnings, errors = execute_prediction(
            bundle,
            invalid,
            valid_categories(),
        )
        self.assertIsNone(predictions)
        self.assertEqual(warnings, ())
        self.assertTrue(any("geometry" in error for error in errors))
        self.assertEqual(bundle.calls, 0)

        predictions, warnings, errors = execute_prediction(
            bundle,
            valid_customer_inputs(),
            valid_categories(),
        )
        self.assertFalse(errors)
        self.assertEqual(bundle.received_inputs["Braid_Type"], "None")
        self.assertAlmostEqual(predictions["Scrap_Rate"], 0.0532)

    def test_inference_exceptions_and_invalid_outputs_never_return_predictions(self):
        bundle = StubBundle(error=InferenceValidationError("Target_Tube_ID_mm: invalid geometry"))
        predictions, _, errors = execute_prediction(
            bundle,
            valid_customer_inputs(),
            valid_categories(),
        )
        self.assertIsNone(predictions)
        self.assertIn("invalid geometry", errors[0])

        bundle = StubBundle(error=RuntimeError("C:\\private\\model.pkl"))
        predictions, _, errors = execute_prediction(
            bundle,
            valid_customer_inputs(),
            valid_categories(),
        )
        self.assertIsNone(predictions)
        self.assertIn("No results were produced", errors[0])
        self.assertNotIn("private", errors[0])

        malformed_result = valid_inference_result()
        malformed_result.predictions = malformed_result.predictions.drop(
            columns=["Operational_Efficiency_pct"]
        )
        bundle = StubBundle(result=malformed_result)
        predictions, _, errors = execute_prediction(
            bundle,
            valid_customer_inputs(),
            valid_categories(),
        )
        self.assertIsNone(predictions)
        self.assertIn("missing Operational_Efficiency_pct", errors[0])
        self.assertIn("No prediction is displayed", errors[0])

        malformed_result = valid_inference_result()
        malformed_result.warnings = None
        bundle = StubBundle(result=malformed_result)
        predictions, _, errors = execute_prediction(
            bundle,
            valid_customer_inputs(),
            valid_categories(),
        )
        self.assertIsNone(predictions)
        self.assertIn("warning metadata is invalid", errors[0])

    def test_candidate_bundle_loader_is_cached(self):
        load_inference_bundle.clear()
        cached_bundle = object()
        try:
            with patch.object(
                CandidateInferenceBundle,
                "load",
                return_value=cached_bundle,
            ) as loader:
                self.assertIs(load_inference_bundle(), cached_bundle)
                self.assertIs(load_inference_bundle(), cached_bundle)
                loader.assert_called_once_with()
        finally:
            load_inference_bundle.clear()


class StreamlitAppTests(unittest.TestCase):
    def tearDown(self):
        load_inference_bundle.clear()

    def _start_app(self):
        app = AppTest.from_file(str(APP_PATH)).run()
        self.assertFalse(app.exception, [element.value for element in app.exception])
        return app

    @staticmethod
    def _submit(app):
        app.selectbox(key="braid_type").select("None")
        app.button(key="predict_button").click().run()
        return app

    def test_app_starts_and_valid_submission_displays_all_predictions(self):
        app = self._start_app()
        self.assertEqual(len(app.metric), 0)
        app = self._submit(app)

        self.assertFalse(app.exception, [element.value for element in app.exception])
        self.assertEqual(len(app.metric), 2)
        metric_values = {metric.value for metric in app.metric}
        self.assertTrue(any(re.fullmatch(r"\d+\.\d%", value) for value in metric_values))
        self.assertTrue(any(re.fullmatch(r"\d+\.\d{2}%", value) for value in metric_values))
        subheaders = [item.value for item in app.subheader]
        self.assertIn("Submitted input summary", subheaders)
        self.assertIn("Estimated process parameters", subheaders)
        self.assertIn(
            "Synthetic-data proof of concept only.",
            "\n".join(item.value for item in app.info),
        )
        self.assertEqual(len(app.get("vega_lite_chart")), 5)
        self.assertEqual(len(app.get("download_button")), 1)

    def test_invalid_geometry_does_not_display_a_prediction(self):
        app = self._start_app()
        app.number_input(key="target_tube_id_mm").set_value(5.0)
        app.button(key="predict_button").click().run()

        errors = "\n".join(item.value for item in app.error)
        self.assertIn("geometry is inconsistent", errors)
        self.assertEqual(len(app.metric), 0)
        self.assertNotIn("Submitted input summary", [item.value for item in app.subheader])
        self.assertEqual(len(app.get("download_button")), 0)

    def test_inference_failure_shows_safe_error_without_a_prediction(self):
        with patch.object(
            CandidateInferenceBundle,
            "predict",
            side_effect=RuntimeError("C:\\private\\model.pkl"),
        ):
            app = self._submit(self._start_app())

        errors = "\n".join(item.value for item in app.error)
        self.assertIn("No results were produced", errors)
        self.assertNotIn("private", errors)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("download_button")), 0)

    def test_malformed_inference_output_shows_error_without_a_prediction(self):
        malformed = valid_inference_result()
        malformed.predictions = malformed.predictions.drop(columns=["Scrap_Rate"])
        with patch.object(CandidateInferenceBundle, "predict", return_value=malformed):
            app = self._submit(self._start_app())

        errors = "\n".join(item.value for item in app.error)
        self.assertIn("missing Scrap_Rate", errors)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("download_button")), 0)

    def test_latest_successful_result_drives_report_and_failure_clears_previous_report(self):
        first = valid_inference_result()
        first.predictions.loc[0, "Operational_Efficiency_pct"] = 89.12
        first.predictions.loc[0, "Scrap_Rate"] = 0.0321
        second = valid_inference_result()
        second.predictions.loc[0, "Extruder_Temperature_C"] = 222.25
        second.predictions.loc[0, "Operational_Efficiency_pct"] = 71.23
        second.predictions.loc[0, "Scrap_Rate"] = 0.0876

        with patch.object(
            CandidateInferenceBundle,
            "predict",
            side_effect=[first, second],
        ) as predict:
            app = self._submit(self._start_app())
            app.number_input(key="customer_required_length_m").set_value(6400.0)
            app.button(key="predict_button").click().run()

        self.assertEqual(predict.call_count, 2)
        self.assertEqual(len(app.get("download_button")), 1)
        self.assertIn("71.2%", {metric.value for metric in app.metric})
        self.assertIn("8.76%", {metric.value for metric in app.metric})
        report = build_prediction_report(
            app.session_state["submitted_inputs"],
            app.session_state["validated_predictions"],
            app.session_state["prediction_generated_at_utc"],
        )
        rows = {
            row["Field"]: row
            for row in csv.DictReader(io.StringIO(report))
        }
        self.assertEqual(rows["Customer_Required_Length_m"]["Value"], "6400.0")
        self.assertEqual(rows["Extruder_Temperature_C"]["Value"], "222.25")
        self.assertEqual(rows["Operational_Efficiency_pct"]["Value"], "71.23")
        self.assertEqual(rows["Scrap_Rate"]["Value"], "8.76")

        with patch.object(
            CandidateInferenceBundle,
            "predict",
            side_effect=[first, RuntimeError("inference failed")],
        ):
            app = self._submit(self._start_app())
            app.number_input(key="customer_required_length_m").set_value(6400.0)
            app.button(key="predict_button").click().run()
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("download_button")), 0)

    def test_invalid_submission_after_success_hides_previous_report(self):
        with patch.object(
            CandidateInferenceBundle,
            "predict",
            return_value=valid_inference_result(),
        ) as predict:
            app = self._submit(self._start_app())
            self.assertEqual(len(app.get("download_button")), 1)
            app.number_input(key="target_tube_id_mm").set_value(5.0)
            app.button(key="predict_button").click().run()

        self.assertEqual(predict.call_count, 1)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("download_button")), 0)


if __name__ == "__main__":
    unittest.main()
