"""Streamlit demonstration app for synthetic thermoplastic extrusion estimates."""

from __future__ import annotations

import csv
import io
import logging
import math
import numbers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import CATEGORICAL_FEATURES, CUSTOMER_NUMERIC_FEATURES, MODEL1_FEATURES  # noqa: E402
from app.production_intelligence import (  # noqa: E402
    append_prediction_history,
    audit_downtime_records,
    history_frame,
)
from training.evaluation_results.controlled_stacking.bundle_audit.candidate_inference import (  # noqa: E402
    CandidateInferenceBundle,
    InferenceValidationError,
)

LOGGER = logging.getLogger(__name__)
APP_TITLE = "Thermoplastic extrusion analytics"
EFFICIENCY_TARGET = "Operational_Efficiency_pct"
SCRAP_TARGET = "Scrap_Rate"

PROCESS_DISPLAY = (
    ("Extruder_Temperature_C", "Extruder temperature", "°C", "degC", 1),
    ("Die_Temperature_C", "Die temperature", "°C", "degC", 1),
    ("Cooling_Tank_Temperature_C", "Cooling tank temperature", "°C", "degC", 1),
    ("Melt_Pressure_bar", "Melt pressure", "bar", "bar", 1),
    ("Air_Pressure_bar", "Air pressure", "bar", "bar", 2),
    ("Screw_RPM", "Screw speed", "RPM", "rpm", 1),
    ("Line_Speed_m_min", "Line speed", "m/min", "m/min", 2),
    ("Puller_Speed_m_min", "Puller speed", "m/min", "m/min", 2),
    ("Extrusion_Speed_kg_hr", "Extrusion rate", "kg/hr", "kg/hr", 1),
)

OUTPUT_FIELDS = tuple(item[0] for item in PROCESS_DISPLAY) + (
    EFFICIENCY_TARGET,
    SCRAP_TARGET,
)
OUTPUT_UNITS = {
    **{item[0]: item[3] for item in PROCESS_DISPLAY},
    EFFICIENCY_TARGET: "percentage points",
    SCRAP_TARGET: "fraction",
}
PROCESS_CHART_GROUPS = (
    ("Temperatures", "°C", (0, 1, 2)),
    ("Pressures", "bar", (3, 4)),
    ("Linear speeds", "m/min", (6, 7)),
    ("Screw speed", "RPM", (5,)),
    ("Extrusion rate", "kg/hr", (8,)),
)
INPUT_LABELS = {
    "Material_Type": ("Material type", ""),
    "Material_Grade": ("Material grade", ""),
    "Tube_Type": ("Tube type", ""),
    "Braid_Type": ("Braid type", ""),
    "Target_Tube_OD_mm": ("Target tube OD", "mm"),
    "Target_Tube_Wall_mm": ("Target tube wall", "mm"),
    "Target_Tube_ID_mm": ("Target tube ID", "mm"),
    "Customer_Required_Length_m": ("Required length", "m"),
}
INPUT_UNITS = {
    feature: unit or "category"
    for feature, (_, unit) in INPUT_LABELS.items()
}
REPORT_DISCLAIMER = (
    "Synthetic-data proof of concept only. These candidate estimates are not validated on "
    "Polyhose production data and are not approved factory settings or production advice."
)
REPORT_FILENAME = "thermoplastic_extrusion_prediction_report.csv"
VALIDATION_DIR = PROJECT_ROOT / "training" / "evaluation_results" / "controlled_stacking"
MODEL2_TEST_COMPARISON = VALIDATION_DIR / "model2_candidate" / "test_comparison.csv"
MODEL3_TEST_COMPARISON = VALIDATION_DIR / "model3_candidate" / "test_comparison.csv"
CONTROLLED_CV_SUMMARY = VALIDATION_DIR / "metrics_summary.csv"
DATASET_PATH = PROJECT_ROOT / "dataset" / "Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"


class PredictionOutputError(ValueError):
    """A candidate model response does not match the verified output contract."""


def build_customer_inputs(
    material_type: str,
    material_grade: str,
    tube_type: str,
    braid_type: str,
    target_tube_od_mm: float,
    target_tube_wall_mm: float,
    target_tube_id_mm: float,
    customer_required_length_m: float,
) -> dict[str, Any]:
    """Map UI values to the exact customer-facing inference feature names."""
    values = {
        "Material_Type": material_type,
        "Material_Grade": material_grade,
        "Tube_Type": tube_type,
        "Braid_Type": braid_type,
        "Target_Tube_OD_mm": target_tube_od_mm,
        "Target_Tube_Wall_mm": target_tube_wall_mm,
        "Target_Tube_ID_mm": target_tube_id_mm,
        "Customer_Required_Length_m": customer_required_length_m,
    }
    if list(values) != MODEL1_FEATURES:
        raise RuntimeError("The UI input mapping no longer matches the candidate inference contract")
    return values


def validate_customer_inputs(
    values: Mapping[str, object],
    categories: Mapping[str, set[str]],
) -> tuple[str, ...]:
    """Validate the eight customer inputs without inventing unsupported limits."""
    errors: list[str] = []
    for feature in MODEL1_FEATURES:
        if feature not in values:
            errors.append(f"{feature}: a value is required.")
    for feature in values:
        if feature not in MODEL1_FEATURES:
            errors.append(f"{feature}: this is not a supported customer input.")

    for feature in CATEGORICAL_FEATURES:
        if feature not in values:
            continue
        value = values[feature]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{feature}: select a nonblank value from the available categories.")
        elif value not in categories.get(feature, set()):
            errors.append(f"{feature}: select a value from the available categories.")

    numeric_values: dict[str, float] = {}
    for feature in CUSTOMER_NUMERIC_FEATURES:
        if feature not in values:
            continue
        value = values[feature]
        if isinstance(value, (bool, np.bool_)):
            errors.append(f"{feature}: enter a finite numeric value.")
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{feature}: enter a finite numeric value.")
            continue
        if not math.isfinite(numeric_value):
            errors.append(f"{feature}: the value must be finite.")
            continue
        numeric_values[feature] = numeric_value

    for feature in (
        "Target_Tube_OD_mm",
        "Target_Tube_Wall_mm",
        "Customer_Required_Length_m",
    ):
        if feature in numeric_values and numeric_values[feature] <= 0:
            errors.append(f"{feature}: enter a value greater than zero.")

    id_field = "Target_Tube_ID_mm"
    if id_field in numeric_values and numeric_values[id_field] <= 0.5:
        errors.append(f"{id_field}: enter a value greater than 0.5 mm.")

    geometry_fields = (
        "Target_Tube_OD_mm",
        "Target_Tube_Wall_mm",
        "Target_Tube_ID_mm",
    )
    if all(field in numeric_values for field in geometry_fields):
        expected_id = numeric_values["Target_Tube_OD_mm"] - (
            2 * numeric_values["Target_Tube_Wall_mm"]
        )
        if abs(numeric_values[id_field] - expected_id) >= 0.25:
            errors.append(
                "Tube geometry is inconsistent: |ID - (OD - 2 × wall)| must be less than 0.25 mm."
            )

    return tuple(errors)


def validate_prediction_outputs(result: object) -> dict[str, float]:
    """Check output schema, units, numeric validity, and target scales before display."""
    predictions = getattr(result, "predictions", None)
    if predictions is None or not hasattr(predictions, "columns"):
        raise PredictionOutputError("The model did not return a prediction table.")

    columns = list(predictions.columns)
    missing = [field for field in OUTPUT_FIELDS if field not in columns]
    if missing:
        raise PredictionOutputError(f"Prediction output is missing {missing[0]}.")
    if len(columns) != len(OUTPUT_FIELDS) or set(columns) != set(OUTPUT_FIELDS):
        raise PredictionOutputError("Prediction output fields do not match the expected model contract.")

    try:
        row_count = len(predictions)
    except TypeError as error:
        raise PredictionOutputError("The model returned an unreadable prediction table.") from error
    if row_count != 1:
        raise PredictionOutputError("The model must return exactly one prediction for this request.")

    units = getattr(result, "units", None)
    if not isinstance(units, Mapping):
        raise PredictionOutputError("Prediction output units are missing.")
    for field, expected_unit in OUTPUT_UNITS.items():
        if units.get(field) != expected_unit:
            raise PredictionOutputError(f"Prediction output units are invalid for {field}.")

    validated: dict[str, float] = {}
    for field in OUTPUT_FIELDS:
        try:
            raw_value = predictions[field].iloc[0]
        except (KeyError, IndexError, AttributeError, TypeError) as error:
            raise PredictionOutputError(f"Prediction output is unreadable for {field}.") from error
        if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Number):
            raise PredictionOutputError(f"Prediction for {field} must be numeric.")
        value = float(raw_value)
        if not math.isfinite(value):
            raise PredictionOutputError(f"Prediction for {field} must be finite.")
        validated[field] = value

    if not 0 <= validated[EFFICIENCY_TARGET] <= 100:
        raise PredictionOutputError(
            "Operational efficiency must be reported in percentage points from 0 to 100."
        )
    if not 0 <= validated[SCRAP_TARGET] <= 1:
        raise PredictionOutputError("Scrap rate must be returned as a fraction from 0 to 1.")
    return validated


def execute_prediction(
    bundle: CandidateInferenceBundle,
    customer_inputs: Mapping[str, object],
    categories: Mapping[str, set[str]],
) -> tuple[dict[str, float] | None, tuple[str, ...], tuple[str, ...]]:
    """Validate, infer, and validate outputs while returning only caller-safe errors."""
    input_errors = validate_customer_inputs(customer_inputs, categories)
    if input_errors:
        return None, (), input_errors

    try:
        result = bundle.predict(customer_inputs)
    except InferenceValidationError as error:
        return None, (), (str(error),)
    except Exception:
        LOGGER.exception("Unexpected candidate inference failure")
        return (
            None,
            (),
            ("Prediction failed unexpectedly. No results were produced; retry or contact the application owner.",),
        )

    try:
        validated = validate_prediction_outputs(result)
    except PredictionOutputError as error:
        LOGGER.error("Candidate inference returned an invalid result: %s", error)
        return None, (), (f"{error} No prediction is displayed.",)

    result_warnings = getattr(result, "warnings", ())
    if not isinstance(result_warnings, (tuple, list)) or any(
        not isinstance(warning, str) for warning in result_warnings
    ):
        LOGGER.error("Candidate inference returned malformed warning metadata")
        return None, (), ("Prediction warning metadata is invalid. No prediction is displayed.",)
    warnings = tuple(result_warnings)
    return validated, warnings, ()


def scrap_percentage(scrap_fraction: float) -> float:
    """Convert the validated internal scrap fraction to display percentage."""
    return scrap_fraction * 100.0


def efficiency_percentage(efficiency_points: float) -> str:
    """Format efficiency, which the inference contract already reports in percentage points."""
    return f"{efficiency_points:.1f}%"


def format_scrap_percentage(scrap_fraction: float) -> str:
    return f"{scrap_percentage(scrap_fraction):.2f}%"


def build_process_chart_data(
    predictions: Mapping[str, float],
) -> tuple[tuple[str, str, pd.DataFrame], ...]:
    """Build independent chart datasets so each chart has a single measurement unit."""
    chart_data = []
    for title, unit, indices in PROCESS_CHART_GROUPS:
        features = [PROCESS_DISPLAY[index] for index in indices]
        chart_data.append(
            (
                title,
                unit,
                pd.DataFrame(
                    {
                        "Process parameter": [item[1] for item in features],
                        f"Predicted value ({unit})": [
                            predictions[item[0]] for item in features
                        ],
                    }
                ),
            )
        )
    return tuple(chart_data)


@st.cache_data(show_spinner=False)
def load_training_feature_ranges() -> dict[str, tuple[float, float]]:
    """Read numeric input ranges from the checked-in synthetic training data."""
    frame = pd.read_csv(DATASET_PATH, usecols=CUSTOMER_NUMERIC_FEATURES)
    ranges: dict[str, tuple[float, float]] = {}
    for feature in CUSTOMER_NUMERIC_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No numeric training values are available for {feature}.")
        ranges[feature] = (float(values.min()), float(values.max()))
    return ranges


def build_training_domain_warnings(
    customer_inputs: Mapping[str, object],
    training_ranges: Mapping[str, tuple[float, float]],
) -> tuple[str, ...]:
    """Flag extrapolation beyond synthetic training data without inventing factory limits."""
    warnings = []
    for feature in CUSTOMER_NUMERIC_FEATURES:
        if feature not in training_ranges or feature not in customer_inputs:
            continue
        lower, upper = training_ranges[feature]
        value = float(customer_inputs[feature])
        if value < lower or value > upper:
            label = INPUT_LABELS[feature][0]
            warnings.append(
                f"{label}: {value:g} is outside the synthetic training-data range "
                f"({lower:g}–{upper:g}). The estimate may be unreliable."
            )
    return tuple(warnings)


def build_validation_summary(
    model2_comparison: pd.DataFrame,
    model3_comparison: pd.DataFrame,
    cv_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine nested-CV and held-out metrics for the exact downstream candidates."""
    candidate_rows = (
        (
            "Operational efficiency",
            EFFICIENCY_TARGET,
            model2_comparison,
            "MAE_percentage_points",
        ),
        (
            "Scrap rate",
            SCRAP_TARGET,
            model3_comparison,
            "MAE_percentage_points",
        ),
    )
    rows: list[dict[str, object]] = []
    for label, target, test_frame, test_mae_column in candidate_rows:
        selected_test = test_frame.loc[
            (test_frame["split"] == "saved_test_1000_rows")
            & (test_frame["model"] == "corrected_candidate_full_stack_RF")
        ]
        selected_cv = cv_summary.loc[
            (cv_summary["split"] == "outer_cv")
            & (cv_summary["target"] == target)
            & (cv_summary["configuration"] == "full_stack")
        ]
        if len(selected_test) != 1 or len(selected_cv) != 1:
            raise ValueError(f"Expected one CV and one held-out validation row for {label}.")

        test_row = selected_test.iloc[0]
        cv_row = selected_cv.iloc[0]
        cv_mae_column = (
            "MAE_percentage_points_mean"
            if target == SCRAP_TARGET
            else "MAE_mean"
        )
        rows.append(
            {
                "Model": label,
                "CV R² (5-fold)": float(cv_row["R2_mean"]),
                "CV MAE": float(cv_row[cv_mae_column]),
                "Held-out R²": float(test_row["R2"]),
                "Held-out MAE": float(test_row[test_mae_column]),
                "Error unit": "percentage points",
                "Test rows": 1000,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_validation_summary() -> pd.DataFrame:
    """Load nested-CV and held-out metrics for the app's candidate models."""
    model2 = pd.read_csv(MODEL2_TEST_COMPARISON)
    model3 = pd.read_csv(MODEL3_TEST_COMPARISON)
    cv_summary = pd.read_csv(CONTROLLED_CV_SUMMARY)
    return build_validation_summary(model2, model3, cv_summary)


def render_model_validation() -> None:
    with st.expander("Model validation & limitations", expanded=False):
        st.caption(
            "Five-fold outer cross-validation and the fixed 1,000-row held-out test split "
            "for the exact Model 2 and Model 3 candidate artifacts used by this app."
        )
        try:
            summary = load_validation_summary()
        except (OSError, ValueError, KeyError):
            LOGGER.exception("Saved model validation metrics could not be loaded")
            st.warning("Model validation details are temporarily unavailable.")
            return
        st.dataframe(summary, hide_index=True, width="stretch")
        scores = dict(zip(summary["Model"], summary["Held-out R²"]))
        st.warning(
            "The downstream models explain only part of the held-out variation "
            f"(efficiency R²={scores.get('Operational efficiency', float('nan')):.2f}; "
            f"scrap-rate R²={scores.get('Scrap rate', float('nan')):.2f}). "
            "Treat these outputs as exploratory estimates, not dependable production forecasts."
        )
        st.caption(
            "The dataset is synthetic. No Polyhose production records, live telemetry, "
            "controlled factory trial, or approved operating limits are represented here."
        )


def build_prediction_report(
    customer_inputs: Mapping[str, object],
    predictions: Mapping[str, float],
    generated_at_utc: str,
) -> str:
    """Create a CSV snapshot from one successfully validated prediction."""
    if tuple(customer_inputs) != tuple(MODEL1_FEATURES):
        raise ValueError("Report customer inputs do not match the eight-field contract.")
    if set(predictions) != set(OUTPUT_FIELDS):
        raise ValueError("Report predictions do not match the verified output contract.")
    if not generated_at_utc:
        raise ValueError("Report generation timestamp is required.")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=("Section", "Field", "Value", "Units"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(
        {
            "Section": "Report metadata",
            "Field": "Generated at (UTC)",
            "Value": generated_at_utc,
            "Units": "UTC",
        }
    )
    writer.writerow(
        {
            "Section": "Report metadata",
            "Field": "Disclaimer",
            "Value": REPORT_DISCLAIMER,
            "Units": "",
        }
    )
    for feature in MODEL1_FEATURES:
        writer.writerow(
            {
                "Section": "Submitted customer input",
                "Field": feature,
                "Value": customer_inputs[feature],
                "Units": INPUT_UNITS[feature],
            }
        )
    for feature, *_ in PROCESS_DISPLAY:
        writer.writerow(
            {
                "Section": "Predicted process parameter",
                "Field": feature,
                "Value": predictions[feature],
                "Units": OUTPUT_UNITS[feature],
            }
        )
    writer.writerow(
        {
            "Section": "Predicted production indicator",
            "Field": EFFICIENCY_TARGET,
            "Value": predictions[EFFICIENCY_TARGET],
            "Units": "%",
        }
    )
    writer.writerow(
        {
            "Section": "Predicted production indicator",
            "Field": SCRAP_TARGET,
            "Value": f"{scrap_percentage(predictions[SCRAP_TARGET]):.2f}",
            "Units": "%",
        }
    )
    return output.getvalue()


@st.cache_resource(show_spinner="Loading candidate demonstration models...")
def load_inference_bundle() -> CandidateInferenceBundle:
    """Cache candidate artifacts loaded relative to their source module."""
    return CandidateInferenceBundle.load()


def render_input_summary(customer_inputs: Mapping[str, object]) -> None:
    st.subheader("Submitted input summary")
    st.caption("The exact customer specification used for this candidate estimate.")
    labels = [
        ("Material", str(customer_inputs["Material_Type"])),
        ("Material grade", str(customer_inputs["Material_Grade"])),
        ("Tube construction", str(customer_inputs["Tube_Type"])),
        ("Reinforcement", str(customer_inputs["Braid_Type"])),
        ("Outside diameter", f"{float(customer_inputs['Target_Tube_OD_mm']):.2f} mm"),
        ("Wall thickness", f"{float(customer_inputs['Target_Tube_Wall_mm']):.2f} mm"),
        ("Inside diameter", f"{float(customer_inputs['Target_Tube_ID_mm']):.2f} mm"),
        ("Order length", f"{float(customer_inputs['Customer_Required_Length_m']):,.0f} m"),
    ]
    columns = st.columns(4, gap="small")
    for index, (label, value) in enumerate(labels):
        with columns[index % 4]:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"**{value}**")


def render_results(predictions: Mapping[str, float]) -> None:
    efficiency_value = float(predictions[EFFICIENCY_TARGET])
    scrap_value = scrap_percentage(float(predictions[SCRAP_TARGET]))
    order_length = st.session_state.get("submitted_inputs", {}).get("Customer_Required_Length_m")

    st.markdown('<div class="app-section-kicker">RESULTS / CANDIDATE MODEL OUTPUT</div>', unsafe_allow_html=True)
    st.subheader("Production intelligence")
    st.caption("A structured view of the latest validated model response. Values are estimates from synthetic training data.")

    overview_tab, process_tab, profile_tab = st.tabs(
        ["◈  Outcome overview", "⌁  Process breakdown", "▤  Order profile"]
    )

    with overview_tab:
        st.markdown("#### Production outcome")
        metric_left, metric_right = st.columns(2, gap="large")
        with metric_left:
            with st.container(border=True):
                st.metric(
                    "Operational efficiency · OEE",
                    efficiency_percentage(efficiency_value),
                    help="Candidate OEE estimate in percentage points. Not a measured factory result.",
                )
                st.progress(int(round(efficiency_value)), text="Efficiency scale · 0–100")
                st.caption("Scale indicator only. No factory target or acceptance threshold is configured.")
        with metric_right:
            with st.container(border=True):
                st.metric(
                    "Predicted scrap rate",
                    f"{scrap_value:.2f}%",
                    help="Estimated scrap length as a share of actual processing length.",
                )
                st.progress(int(round(scrap_value)), text="Scrap share · 0–100%")
                st.caption("Share of processing length. This is not a defect probability.")
        summary_left, summary_right = st.columns(2, gap="large")
        with summary_left:
            with st.container(border=True):
                st.caption("CUSTOMER ORDER")
                st.markdown(f"### {float(order_length):,.0f} m" if order_length is not None else "### Not supplied")
                st.caption("Requested length entered by the user; not a predicted production quantity.")
        with summary_right:
            with st.container(border=True):
                st.caption("MODEL OUTPUTS")
                st.markdown("### 9 process estimates")
                st.caption("Temperatures, pressures, line speeds, screw speed and extrusion rate.")
        try:
            validation = load_validation_summary().set_index("Model")
            efficiency_mae = float(validation.loc["Operational efficiency", "CV MAE"])
            scrap_mae = float(validation.loc["Scrap rate", "CV MAE"])
            st.caption(
                f"Historical outer-CV MAE on synthetic test folds: efficiency ±{efficiency_mae:.2f} percentage points; "
                f"scrap rate ±{scrap_mae:.2f} percentage points. These are average absolute errors, "
                "not confidence intervals or guarantees for this individual prediction."
            )
        except (OSError, ValueError, KeyError):
            LOGGER.exception("Prediction error context could not be loaded")
            st.caption("Cross-validation error context is unavailable.")
        st.info("All values are candidate estimates. They are not live sensor readings, approved setpoints or production recommendations.")

    with process_tab:
        st.subheader("Estimated process parameters")
        st.caption("Charts are separated by engineering unit. Hover over bars for exact values; compare only within each unit group.")
        chart_data = build_process_chart_data(predictions)
        for row_start in range(0, len(chart_data), 2):
            left, right = st.columns(2, gap="large")
            for offset, column in enumerate((left, right)):
                index = row_start + offset
                if index >= len(chart_data):
                    continue
                title, unit, chart_frame = chart_data[index]
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        value_column = f"Predicted value ({unit})"
                        st.bar_chart(
                            chart_frame,
                            x="Process parameter",
                            y=value_column,
                            x_label="Parameter",
                            y_label=unit,
                            horizontal=True,
                            sort=False,
                            height=250,
                        )
                        with st.expander("View exact values", expanded=False):
                            st.dataframe(chart_frame, hide_index=True, width="stretch")

    with profile_tab:
        render_input_summary(st.session_state["submitted_inputs"])


def render_production_intelligence() -> None:
    """Render session-scoped prediction history and descriptive downtime analytics."""
    st.markdown('<div class="app-section-kicker">04 / Operational analytics</div>', unsafe_allow_html=True)
    st.title("Production intelligence")
    st.caption(
        "Explore predictions generated in this browser session and audit historical downtime records. "
        "Uploaded records are summarized descriptively; no failure prediction is inferred."
    )

    st.subheader("Prediction history")
    history = st.session_state.get("prediction_history", [])
    if history:
        frame = history_frame(history)
        st.metric("Predictions this session", len(frame))
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button(
            "Download prediction history",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="thermoplastic_prediction_history.csv",
            mime="text/csv",
            key="download_prediction_history",
        )
    else:
        st.info("No predictions have been generated in this session yet. Run a production estimate first.")

    if history:
        latest = history[0]
        st.divider()
        st.subheader("Order-length scenario comparison")
        st.caption(
            "Compare a different requested order length using the most recent successful input profile. "
            "All other product specifications remain unchanged. This is a model what-if estimate, not an approved production plan."
        )
        base_inputs = {
            feature: latest.get(f"Input_{feature}")
            for feature in MODEL1_FEATURES
        }
        if all(value is not None for value in base_inputs.values()):
            baseline_length = float(base_inputs["Customer_Required_Length_m"])
            alternative_length = st.number_input(
                "Alternative order length (m)",
                min_value=1.0,
                value=max(1.0, round(baseline_length * 1.2, 2)),
                step=100.0,
                key="scenario_order_length_m",
            )
            if st.button("Compare scenario", type="primary", key="compare_scenario_button"):
                scenario_inputs = dict(base_inputs)
                scenario_inputs["Customer_Required_Length_m"] = float(alternative_length)
                try:
                    scenario_bundle = load_inference_bundle()
                    scenario_predictions, scenario_warnings, scenario_errors = execute_prediction(
                        scenario_bundle,
                        scenario_inputs,
                        scenario_bundle.category_values,
                    )
                except Exception:
                    LOGGER.exception("Scenario comparison failed")
                    scenario_predictions, scenario_warnings, scenario_errors = (
                        None,
                        (),
                        ("Scenario comparison is temporarily unavailable. No result was produced.",),
                    )
                if scenario_errors:
                    for issue in scenario_errors:
                        st.error(issue)
                elif scenario_predictions is not None:
                    baseline_efficiency = float(latest.get("Prediction_Operational_Efficiency_pct", float("nan")))
                    baseline_scrap = float(latest.get("Prediction_Scrap_Rate", float("nan"))) * 100.0
                    efficiency_delta = scenario_predictions[EFFICIENCY_TARGET] - baseline_efficiency
                    scrap_delta = scrap_percentage(scenario_predictions[SCRAP_TARGET]) - baseline_scrap
                    left, right = st.columns(2)
                    left.metric(
                        "Alternative efficiency estimate",
                        efficiency_percentage(scenario_predictions[EFFICIENCY_TARGET]),
                        f"{efficiency_delta:+.2f} percentage points vs baseline",
                    )
                    right.metric(
                        "Alternative scrap estimate",
                        f"{scrap_percentage(scenario_predictions[SCRAP_TARGET]):.2f}%",
                        f"{scrap_delta:+.2f} percentage points vs baseline",
                        delta_color="inverse",
                    )
                    st.caption(
                        f"Baseline order: {baseline_length:,.0f} m · Alternative order: {alternative_length:,.0f} m. "
                        "A positive or negative change is a model output, not evidence of a causal process effect."
                    )
                    for warning in scenario_warnings:
                        st.warning(warning)
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Indicator": "Operational efficiency", "Baseline": baseline_efficiency, "Alternative": scenario_predictions[EFFICIENCY_TARGET], "Change": efficiency_delta, "Unit": "percentage points"},
                                {"Indicator": "Scrap rate", "Baseline": baseline_scrap, "Alternative": scrap_percentage(scenario_predictions[SCRAP_TARGET]), "Change": scrap_delta, "Unit": "percentage points"},
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
        else:
            st.warning("The latest history record is incomplete; scenario comparison is unavailable.")

    st.divider()
    st.subheader("Downtime records analysis")
    st.caption(
        "Upload a CSV containing Timestamp, Machine_ID, Downtime_min and Fault_Code. "
        "Optional fields such as Shift and Production_Order_ID are retained in the uploaded source."
    )
    uploaded = st.file_uploader(
        "Upload historical downtime CSV",
        type=["csv"],
        key="downtime_csv_upload",
    )
    if uploaded is None:
        st.info("No downtime file uploaded. The analytics will appear after a valid CSV is supplied.")
        return
    try:
        downtime_frame = pd.read_csv(uploaded)
        audit = audit_downtime_records(downtime_frame)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read this CSV: {exc}")
        return

    st.metric("Uploaded records", f"{audit['rows']:,}")
    if not audit["passed"]:
        for issue in audit["issues"]:
            st.error(issue)
        return

    first, second = st.columns(2)
    first.metric("Machines represented", int(audit["machine_count"]))
    second.metric("Total recorded downtime", f"{audit['total_downtime_min']:,.1f} min")
    machine_col, fault_col = st.columns(2, gap="large")
    with machine_col:
        st.markdown("**Downtime by machine**")
        st.dataframe(audit["by_machine"], hide_index=True, width="stretch")
        st.bar_chart(audit["by_machine"], x="Machine_ID", y="Total_downtime_min")
    with fault_col:
        st.markdown("**Downtime by fault code**")
        st.dataframe(audit["by_fault"], hide_index=True, width="stretch")
        st.bar_chart(audit["by_fault"], x="Fault_Code", y="Total_downtime_min")
    if not audit["by_day"].empty:
        st.markdown("**Daily downtime trend**")
        st.line_chart(audit["by_day"], x="Date", y="Total_downtime_min")
    st.caption("Descriptive historical analysis only. Results depend on the completeness and accuracy of the uploaded records. No downtime prediction or root-cause attribution is performed.")


def main() -> None:
    st.session_state.setdefault("prediction_history", [])
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/precision_manufacturing:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root {
            --bg:#080f1b; --panel:#111d2d; --panel-2:#162438;
            --line:rgba(148,174,207,.17); --text:#e8f0fb; --muted:#91a4bd;
            --cyan:#42c8f5; --blue:#4c83ff; --green:#39d6a0;
        }
        html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
        .stApp {
            background:
              radial-gradient(ellipse at 78% -12%,rgba(42,111,202,.23),transparent 36%),
              radial-gradient(ellipse at 8% 25%,rgba(23,172,190,.09),transparent 28%),
              linear-gradient(145deg,#080f1b 0%,#0b1422 55%,#0c1727 100%);
            color:var(--text);
        }
        [data-testid="stHeader"] { background:rgba(8,15,27,.84); backdrop-filter:blur(16px); }
        [data-testid="stMainBlockContainer"] { max-width:1580px; padding:1.5rem 2.5rem 4rem; }
        [data-testid="stSidebar"] {
            background:linear-gradient(180deg,#0d1a2b 0%,#101f33 58%,#0c1929 100%);
            border-right:1px solid var(--line);
        }
        [data-testid="stSidebar"] * { color:#dce8f7; }
        [data-testid="stSidebar"] hr { border-color:var(--line); }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { color:#c8d8eb; font-weight:650; }
        h1,h2,h3 { color:#edf5ff; letter-spacing:-.045em; }
        h1 { font-size:clamp(2.1rem,4vw,3.25rem)!important; font-weight:800!important; }
        h2 { font-weight:760!important; }
        h3 { font-weight:700!important; }
        p, label, [data-testid="stMarkdownContainer"] { color:inherit; }
        [data-testid="stMetric"] {
            background:linear-gradient(145deg,rgba(22,39,60,.98),rgba(15,29,46,.98));
            border:1px solid rgba(115,161,213,.22); border-radius:18px;
            padding:1.2rem 1.3rem; min-height:132px;
            box-shadow:0 14px 34px rgba(0,0,0,.19),inset 0 1px 0 rgba(255,255,255,.035);
        }
        [data-testid="stMetricLabel"] { color:#9eb3cc!important; font-weight:700; }
        [data-testid="stMetricValue"] { color:#f2f8ff!important; font-weight:800; }
        [data-testid="stMetricDelta"] { font-family:'DM Mono',monospace; }
        [data-testid="stForm"],[data-testid="stVerticalBlockBorderWrapper"] {
            background:linear-gradient(145deg,rgba(18,32,50,.97),rgba(13,25,41,.98))!important;
            border:1px solid var(--line)!important; border-radius:20px!important;
            box-shadow:0 18px 48px rgba(0,0,0,.17),inset 0 1px 0 rgba(255,255,255,.035);
        }
        [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background:#0b1726!important; color:#edf5ff!important;
            border-color:#2a405b!important; border-radius:10px!important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] * { color:#edf5ff!important; }
        [data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,
        [data-testid="stNumberInput"] label,[data-testid="stSelectbox"] label {
            color:#c6d7eb!important; font-weight:650!important;
        }
        [data-testid="stNumberInput"] button { color:#dcecff!important; }
        [data-testid="stTabs"] [role="tab"] {
            color:#a9bdd4!important; font-weight:750!important;
            padding:.8rem 1rem; border-bottom:2px solid transparent;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color:#61d2f5!important; border-bottom-color:#42c8f5!important;
        }
        [data-testid="stProgress"] > div > div {
            background:linear-gradient(90deg,#2368e8,#42c8f5)!important;
            border-radius:999px;
        }
        [data-testid="stProgress"] > div {
            background:#22354c!important; border-radius:999px; height:.55rem;
        }
        [data-testid="stMetric"] { min-height:115px; }
        
        [data-testid="stFormSubmitButton"] button,[data-testid="stDownloadButton"] button {
            border-radius:11px; min-height:3rem; font-weight:800;
            border:1px solid rgba(76,131,255,.45);
            box-shadow:0 8px 24px rgba(41,103,220,.18);
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background:linear-gradient(105deg,#2368e8,#19a9d1); color:white;
        }
        [data-testid="stAlert"] { border-radius:14px; border:1px solid var(--line); }
        [data-testid="stDataFrame"],[data-testid="stTable"] {
            border:1px solid var(--line); border-radius:13px; overflow:hidden;
        }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:13px; background:rgba(17,29,45,.65); }
        [data-testid="stExpander"] summary { color:#e4efff; font-weight:700; }
        [data-testid="stPlotlyChart"],[data-testid="stVegaLiteChart"] {
            background:rgba(12,24,39,.55); border:1px solid var(--line);
            border-radius:15px; padding:.35rem;
        }
        .app-eyebrow { color:#6bbbe7; font-family:'DM Mono',monospace; font-size:.72rem; font-weight:500; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.7rem; }
        .app-subtitle { color:#9db0c8; font-size:1.02rem; line-height:1.7; max-width:880px; margin-top:-.35rem; margin-bottom:1.3rem; }
        .app-header { padding:.4rem 0 1.3rem; border-bottom:1px solid var(--line); margin-bottom:1.35rem; }
        .app-status { display:inline-flex; align-items:center; gap:.5rem; color:#83f0c3; background:rgba(35,174,125,.11); border:1px solid rgba(57,214,160,.28); border-radius:999px; padding:.38rem .8rem; font-family:'DM Mono',monospace; font-size:.7rem; font-weight:500; letter-spacing:.08em; }
        .app-status-dot { width:.46rem; height:.46rem; border-radius:50%; background:#39d6a0; box-shadow:0 0 12px #39d6a0; display:inline-block; }
        .app-section-kicker { color:#61bce8; font-family:'DM Mono',monospace; font-size:.7rem; font-weight:500; letter-spacing:.15em; text-transform:uppercase; margin-top:.7rem; }
        .app-hero-line { height:2px; width:72px; background:linear-gradient(90deg,#42c8f5,#4c83ff); border-radius:2px; margin:.8rem 0 1.1rem; }

        /* High-contrast industrial dashboard treatment */
        .stApp:before { content:""; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.22; background-image:linear-gradient(rgba(120,166,214,.075) 1px,transparent 1px),linear-gradient(90deg,rgba(120,166,214,.075) 1px,transparent 1px); background-size:42px 42px; mask-image:linear-gradient(to bottom,black,transparent 82%); }
        [data-testid="stMainBlockContainer"] { position:relative; z-index:1; }
        .app-header { min-height:116px; padding:1.1rem 1.25rem 1rem; margin:0 0 1.1rem; border:1px solid rgba(66,200,245,.19); border-radius:20px; background:linear-gradient(110deg,rgba(17,39,61,.92),rgba(12,25,43,.68) 58%,rgba(25,64,101,.55)); box-shadow:inset 0 1px rgba(255,255,255,.04),0 18px 55px rgba(0,0,0,.18); }
        .app-header:before { right:8%; top:-135px; width:290px; height:290px; border-color:rgba(66,200,245,.3); }
        .app-header:after { right:12%; top:-105px; width:190px; height:190px; }
        .app-eyebrow { color:#82d9f7; font-family:'DM Mono',monospace; font-size:.67rem; letter-spacing:.19em; text-transform:uppercase; margin-bottom:.75rem; }
        .app-status { background:rgba(20,177,139,.12); border-color:rgba(57,214,160,.32); color:#8cf0ca; }
        .app-status-dot { background:#55edb7; box-shadow:0 0 16px #39d6a0; }
        .app-hero-line { width:108px; height:3px; box-shadow:0 0 18px rgba(66,200,245,.38); }
        [data-testid="stSidebar"] { box-shadow:12px 0 42px rgba(0,0,0,.16); }
        [data-testid="stSidebar"] h3 { color:#f1f7ff; letter-spacing:.025em; }
        [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:.42rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { border:1px solid transparent; border-radius:10px; padding:.48rem .6rem; transition:all .2s ease; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:linear-gradient(90deg,rgba(44,153,220,.2),rgba(44,153,220,.045)); border-color:rgba(66,200,245,.24); color:#f3fbff; }
        [data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(19,36,56,.88),rgba(13,27,44,.9)); border:1px solid rgba(112,157,205,.2)!important; border-radius:18px!important; }
        [data-testid="stForm"] { background:linear-gradient(145deg,rgba(18,35,55,.94),rgba(12,25,42,.94)); border:1px solid rgba(66,200,245,.23)!important; border-radius:20px!important; padding:1.25rem!important; box-shadow:0 20px 60px rgba(0,0,0,.16); }
        [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input { background:#0b1727!important; color:#edf6ff!important; border:1px solid rgba(124,164,207,.3)!important; border-radius:10px!important; }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#0b1727!important; color:#edf6ff!important; border:1px solid rgba(124,164,207,.3)!important; border-radius:10px!important; }
        [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] { background:#12243a!important; border:1px solid rgba(66,200,245,.25)!important; }
        [role="option"] { color:#eaf4ff!important; }
        [role="option"]:hover { background:rgba(66,200,245,.16)!important; }
        [data-testid="stFormSubmitButton"] button,[data-testid="stButton"] button { border:1px solid rgba(95,207,247,.35)!important; border-radius:11px!important; font-weight:800!important; letter-spacing:.015em; box-shadow:0 8px 24px rgba(29,112,201,.18); }
        [data-testid="stTabs"] [role="tablist"] { gap:.45rem; border-bottom:1px solid rgba(126,166,209,.2); }
        [data-testid="stTabs"] [role="tab"] { border:1px solid transparent; border-radius:10px 10px 0 0; padding:.55rem .9rem; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#8de1ff!important; background:rgba(47,153,214,.1); border-color:rgba(66,200,245,.25); }
        [data-testid="stExpander"] { border:1px solid rgba(112,157,205,.2); border-radius:13px; background:rgba(15,30,48,.55); }
        [data-testid="stDataFrame"] { border:1px solid rgba(112,157,205,.2); border-radius:13px; overflow:hidden; }
        [data-testid="stAlert"] { border-radius:13px; }
        [data-testid="stMetric"] { min-height:132px; padding:1rem 1.1rem; border-radius:17px; background:linear-gradient(145deg,rgba(23,48,73,.98),rgba(13,29,47,.98)); }
        [data-testid="stMetricLabel"] { color:#9fb7d2!important; font-size:.78rem!important; letter-spacing:.045em; text-transform:uppercase; }
        [data-testid="stMetricValue"] { color:#f2f8ff!important; font-weight:800!important; }
        @media(max-width:900px) { .app-header { min-height:96px; padding:1rem; } [data-testid="stForm"] { padding:.8rem!important; } }

        /* Motion and visual polish: CSS-only, no new runtime dependency. */
        @keyframes appFadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
        @keyframes appGlowPulse { 0%,100% { box-shadow:0 0 0 0 rgba(57,214,160,.12),0 0 12px rgba(57,214,160,.12); } 50% { box-shadow:0 0 0 6px rgba(57,214,160,0),0 0 20px rgba(57,214,160,.32); } }
        @keyframes appGradientDrift { 0% { background-position:0% 50%; } 50% { background-position:100% 50%; } 100% { background-position:0% 50%; } }
        @keyframes appOrbit { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
        @keyframes appShimmer { 0% { background-position:-180% 0; } 100% { background-position:180% 0; } }
        .app-header { position:relative; overflow:hidden; isolation:isolate; animation:appFadeUp .7s cubic-bezier(.2,.75,.25,1) both; }
        .app-header:before { content:""; position:absolute; z-index:-1; width:250px; height:250px; right:3%; top:-175px; border:1px solid rgba(66,200,245,.24); border-radius:50%; box-shadow:0 0 0 22px rgba(66,200,245,.035),0 0 0 48px rgba(76,131,255,.025),0 0 70px rgba(66,200,245,.12); animation:appOrbit 32s linear infinite; }
        .app-header:after { content:""; position:absolute; z-index:-1; width:130px; height:130px; right:7%; top:-95px; border:1px dashed rgba(76,131,255,.35); border-radius:50%; animation:appOrbit 22s linear infinite reverse; }
        .app-status-dot { animation:appGlowPulse 2.6s ease-in-out infinite; }
        [data-testid="stMainBlockContainer"] > div { animation:appFadeUp .55s ease both; }
        [data-testid="stMetric"] { position:relative; overflow:hidden; transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease; animation:appFadeUp .65s ease both; }
        [data-testid="stMetric"]:before { content:""; position:absolute; inset:0 auto 0 -65%; width:48%; pointer-events:none; background:linear-gradient(105deg,transparent,rgba(255,255,255,.075),transparent); transform:skewX(-18deg); transition:left .65s ease; }
        [data-testid="stMetric"]:hover { transform:translateY(-4px); border-color:rgba(66,200,245,.48); box-shadow:0 20px 42px rgba(0,0,0,.28),0 0 24px rgba(66,200,245,.08); }
        [data-testid="stMetric"]:hover:before { left:125%; }
        [data-testid="stVerticalBlockBorderWrapper"],[data-testid="stForm"] { transition:border-color .22s ease,box-shadow .22s ease,transform .22s ease; }
        [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:rgba(66,200,245,.3)!important; box-shadow:0 20px 52px rgba(0,0,0,.23),0 0 24px rgba(66,200,245,.045)!important; }
        [data-testid="stFormSubmitButton"] button,[data-testid="stDownloadButton"] button { background-size:200% 100%; background-image:linear-gradient(105deg,#2368e8 0%,#19a9d1 48%,#2368e8 100%); transition:transform .2s ease,box-shadow .2s ease,background-position .45s ease; }
        [data-testid="stFormSubmitButton"] button:hover,[data-testid="stDownloadButton"] button:hover { transform:translateY(-2px); background-position:100% 0; box-shadow:0 12px 30px rgba(41,130,230,.3); }
        [data-testid="stTabs"] [role="tab"] { transition:color .2s ease,background .2s ease; }
        [data-testid="stTabs"] [role="tab"]:hover { color:#e9f8ff!important; background:rgba(66,200,245,.06); border-radius:9px 9px 0 0; }
        [data-testid="stPlotlyChart"],[data-testid="stVegaLiteChart"] { transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease; }
        [data-testid="stPlotlyChart"]:hover,[data-testid="stVegaLiteChart"]:hover { transform:translateY(-2px); border-color:rgba(66,200,245,.32); box-shadow:0 14px 32px rgba(0,0,0,.18); }
        [data-testid="stProgress"] > div > div { background-size:200% 100%!important; animation:appGradientDrift 3.5s ease infinite; }
        [data-testid="stSpinner"] > div { border-top-color:#42c8f5!important; }
        @media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms!important; animation-iteration-count:1!important; scroll-behavior:auto!important; transition-duration:.01ms!important; } }
        @media(max-width:900px) {
            [data-testid="stMainBlockContainer"] { padding:1rem 1rem 2.5rem; }
            h1 { font-size:2rem!important; }
        }

        /* Final control-system pass: keep Streamlit numeric steppers inside the dark theme. */
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stNumberInput"] [data-baseweb="base-input"] {
            background:#0b1727!important;
            border:1px solid #38516f!important;
            border-radius:11px!important;
            box-shadow:inset 0 1px 2px rgba(0,0,0,.28)!important;
            overflow:hidden!important;
        }
        [data-testid="stNumberInput"] input {
            background:#0b1727!important;
            color:#f1f6fd!important;
            -webkit-text-fill-color:#f1f6fd!important;
            border:0!important;
            border-radius:10px!important;
            caret-color:#54c8ed!important;
            font-variant-numeric:tabular-nums;
        }
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] button[kind="secondary"],
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] {
            min-width:2.65rem!important;
            width:2.65rem!important;
            height:2.65rem!important;
            padding:0!important;
            color:#c7d9ed!important;
            background:#14263b!important;
            border:0!important;
            border-left:1px solid #304964!important;
            border-radius:0!important;
            box-shadow:none!important;
            opacity:1!important;
            transition:background .18s ease,color .18s ease!important;
        }
        [data-testid="stNumberInput"] button:hover,
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"]:hover,
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"]:hover {
            color:#ffffff!important;
            background:#1d4567!important;
        }
        [data-testid="stNumberInput"] button:disabled {
            color:#66819d!important;
            background:#101e30!important;
            opacity:.75!important;
        }
        [data-testid="stNumberInput"] button svg {
            fill:currentColor!important;
            color:inherit!important;
            opacity:1!important;
        }
        [data-testid="stNumberInput"] button:focus-visible,
        [data-testid="stNumberInput"] input:focus-visible {
            outline:2px solid #42c8f5!important;
            outline-offset:-2px!important;
        }
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background:#0b1727!important;
            border:1px solid #38516f!important;
            border-radius:11px!important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] svg {
            fill:#d6e6f7!important;
            color:#d6e6f7!important;
        }
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
            color:#f7fbff!important;
        }
        </style>
        <div class="app-header">
          <div class="app-eyebrow">Manufacturing intelligence / decision support</div>
          <div class="app-status"><span class="app-status-dot"></span> DEMONSTRATION ENVIRONMENT</div>
          <div style="display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1rem">
            <span style="font:500 .64rem 'DM Mono',monospace;letter-spacing:.1em;color:#b6cce4;border:1px solid rgba(132,174,217,.22);border-radius:7px;padding:.32rem .55rem;background:rgba(255,255,255,.025)">01 / PROCESS CONFIGURATION</span>
            <span style="font:500 .64rem 'DM Mono',monospace;letter-spacing:.1em;color:#b6cce4;border:1px solid rgba(132,174,217,.22);border-radius:7px;padding:.32rem .55rem;background:rgba(255,255,255,.025)">02 / QUALITY ESTIMATION</span>
            <span style="font:500 .64rem 'DM Mono',monospace;letter-spacing:.1em;color:#b6cce4;border:1px solid rgba(132,174,217,.22);border-radius:7px;padding:.32rem .55rem;background:rgba(255,255,255,.025)">03 / PRODUCTION REVIEW</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Thermoplastic extrusion")
    st.markdown(
        '<div class="app-subtitle">Engineering estimates for product configuration, process conditions and production quality. Designed as a decision-support workspace, not a machine-control system.</div>',
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("### EXTRUSION INTELLIGENCE")
        st.caption("PROCESS ANALYTICS PLATFORM")
        st.divider()
        st.markdown("**WORKSPACE**")
        workspace = st.radio(
            "Workspace navigation",
            ("Production planning", "Process intelligence", "Production intelligence", "Model governance"),
            index=0,
            key="workspace_navigation",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("MODEL STATUS")
        st.markdown("Candidate bundle · Loaded on demand")
        st.caption("Data basis: synthetic demonstration dataset")
    st.warning(
        "SYNTHETIC-DATA PROOF OF CONCEPT — This candidate model has not been validated on "
        "Polyhose production data. Do not use its outputs as factory settings or production advice."
    )

    if workspace == "Production intelligence":
        render_production_intelligence()
        st.stop()

    if workspace == "Model governance":
        st.markdown('<div class="app-section-kicker">03 / Model governance</div>', unsafe_allow_html=True)
        st.title("Model validation & governance")
        render_model_validation()
        st.info(
            "This prototype uses synthetic training data. Model scores describe performance on "
            "the project's held-out synthetic test split; they do not establish factory readiness."
        )
        st.stop()

    if workspace == "Process intelligence":
        st.markdown('<div class="app-section-kicker">02 / Process intelligence</div>', unsafe_allow_html=True)
        st.title("Process intelligence")
        saved_predictions = st.session_state.get("validated_predictions")
        if saved_predictions is None:
            st.info("Generate a prediction in Production planning to inspect its process estimates here.")
        else:
            render_results(saved_predictions)
            st.download_button(
                "Download prediction report",
                data=build_prediction_report(
                    st.session_state["submitted_inputs"],
                    saved_predictions,
                    st.session_state["prediction_generated_at_utc"],
                ),
                file_name=REPORT_FILENAME,
                mime="text/csv",
                icon=":material/download:",
                on_click="ignore",
                key="download_prediction_report_process",
            )
        st.stop()

    st.markdown('<div class="app-section-kicker">01 / Production planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-hero-line"></div>', unsafe_allow_html=True)

    try:
        bundle = load_inference_bundle()
    except Exception:
        LOGGER.exception("Candidate model bundle could not be loaded")
        st.error("Candidate models could not be loaded. No prediction is available; contact the application owner.")
        return

    categories = bundle.category_values
    if "None" not in categories.get("Braid_Type", set()):
        LOGGER.error("Candidate category vocabulary is missing the required Braid_Type='None' category")
        st.error("The candidate category list is incomplete. No prediction is available; contact the application owner.")
        return

    with st.form("customer_requirements", border=True):
        st.subheader("Customer inputs")

        st.markdown("**Material selection**")
        material_columns = st.columns(2)
        material_type = material_columns[0].selectbox(
            "Material type",
            sorted(categories["Material_Type"]),
            key="material_type",
            help="Choose a material category from the fitted candidate model.",
        )
        material_grade = material_columns[1].selectbox(
            "Material grade",
            sorted(categories["Material_Grade"]),
            key="material_grade",
            help="Choose a grade from the fitted candidate model.",
        )
        tube_type, braid_type = st.columns(2)
        tube_type_value = tube_type.selectbox(
            "Tube type",
            sorted(categories["Tube_Type"]),
            key="tube_type",
        )
        braid_options = sorted(categories["Braid_Type"])
        braid_type_value = braid_type.selectbox(
            "Braid type",
            braid_options,
            index=braid_options.index("None"),
            key="braid_type",
            help="“None” is a valid fitted category.",
        )

        st.markdown("**Tube specifications**")
        dimension_columns = st.columns(3)
        target_tube_od_mm = dimension_columns[0].number_input(
            "Target tube OD (mm)",
            min_value=0.0,
            value=10.0,
            step=0.1,
            format="%.2f",
            key="target_tube_od_mm",
            help="Must be greater than zero. No approved maximum is defined in the project.",
        )
        target_tube_wall_mm = dimension_columns[1].number_input(
            "Target tube wall (mm)",
            min_value=0.0,
            value=2.0,
            step=0.1,
            format="%.2f",
            key="target_tube_wall_mm",
            help="Must be greater than zero and consistent with OD and ID.",
        )
        target_tube_id_mm = dimension_columns[2].number_input(
            "Target tube ID (mm)",
            min_value=0.0,
            value=6.0,
            step=0.1,
            format="%.2f",
            key="target_tube_id_mm",
            help="Must exceed 0.5 mm and satisfy the existing OD/wall geometry rule.",
        )

        st.markdown("**Order requirements**")
        customer_required_length_m = st.number_input(
            "Customer-required length (m)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            format="%.2f",
            key="customer_required_length_m",
            help="Must be greater than zero. No approved maximum is defined in the project.",
        )
        submitted = st.form_submit_button(
            "Generate estimates",
            type="primary",
            icon=":material/analytics:",
            key="predict_button",
        )

    if submitted:
        for state_key in (
            "validated_predictions",
            "prediction_warnings",
            "submitted_inputs",
            "prediction_generated_at_utc",
        ):
            st.session_state.pop(state_key, None)
        customer_inputs = build_customer_inputs(
            material_type,
            material_grade,
            tube_type_value,
            braid_type_value,
            target_tube_od_mm,
            target_tube_wall_mm,
            target_tube_id_mm,
            customer_required_length_m,
        )
        with st.spinner("Validating inputs and calculating candidate estimates..."):
            predictions, warnings, errors = execute_prediction(bundle, customer_inputs, categories)
            if predictions is not None:
                try:
                    domain_warnings = build_training_domain_warnings(
                        customer_inputs,
                        load_training_feature_ranges(),
                    )
                    warnings = tuple(warnings) + domain_warnings
                except (OSError, ValueError, KeyError):
                    LOGGER.exception("Synthetic training-domain check could not be completed")
                    warnings = tuple(warnings) + (
                        "The app could not verify whether these inputs are within the synthetic training-data range.",
                    )

        for error in errors:
            st.error(error)
        if predictions is not None:
            st.session_state["validated_predictions"] = predictions
            st.session_state["prediction_warnings"] = warnings
            st.session_state["submitted_inputs"] = customer_inputs
            generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            st.session_state["prediction_generated_at_utc"] = generated_at
            st.session_state["prediction_history"] = append_prediction_history(
                st.session_state["prediction_history"],
                customer_inputs,
                predictions,
                generated_at,
            )

    predictions = st.session_state.get("validated_predictions")
    if predictions is not None:
        st.divider()
        st.info(REPORT_DISCLAIMER)
        for warning in st.session_state.get("prediction_warnings", ()):
            st.warning(warning)
        render_results(predictions)
        report = build_prediction_report(
            st.session_state["submitted_inputs"],
            predictions,
            st.session_state["prediction_generated_at_utc"],
        )
        st.download_button(
            "Download prediction report",
            data=report,
            file_name=REPORT_FILENAME,
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            key="download_prediction_report",
        )


if __name__ == "__main__":
    main()
