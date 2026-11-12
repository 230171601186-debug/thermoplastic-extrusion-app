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
    rows = []
    for field in MODEL1_FEATURES:
        label, unit = INPUT_LABELS[field]
        value = customer_inputs[field]
        display_value = f"{float(value):,.2f} {unit}" if unit else str(value)
        rows.append({"Input": label, "Submitted value": display_value})
    st.table(rows)


def render_results(predictions: Mapping[str, float]) -> None:
    st.subheader("Predicted production indicators")
    efficiency, scrap = st.columns(2)
    efficiency.metric(
        "Operational efficiency",
        efficiency_percentage(predictions[EFFICIENCY_TARGET]),
        border=True,
        help="The candidate output is OEE expressed in percentage points (0–100). It is an estimate, not a measured factory result.",
    )
    efficiency.caption(
        "Estimated OEE expressed in percentage points (0–100); not a measured factory result."
    )
    scrap.metric(
        "Predicted scrap rate",
        format_scrap_percentage(predictions[SCRAP_TARGET]),
        border=True,
        help="Estimated scrap length divided by actual processing length. The model returns a fraction, shown here as a percentage.",
    )
    scrap.caption(
        "Estimated scrap length as a share of actual processing length; the model fraction is displayed as a percentage."
    )

    st.subheader("Estimated process parameters")
    st.caption(
        "Charts are separated by measurement unit. Bar lengths compare values only within "
        "the same unit; displayed values are model estimates."
    )
    chart_columns = st.columns(2)
    for index, (title, unit, chart_frame) in enumerate(
        build_process_chart_data(predictions)
    ):
        with chart_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                value_column = f"Predicted value ({unit})"
                st.bar_chart(
                    chart_frame,
                    x="Process parameter",
                    y=value_column,
                    x_label="Process parameter",
                    y_label=value_column,
                    horizontal=True,
                    sort=False,
                    height=230,
                )
                st.dataframe(
                    chart_frame,
                    hide_index=True,
                    width="stretch",
                )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/precision_manufacturing:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        :root {
            --app-ink: #17233b;
            --app-muted: #64748b;
            --app-border: #e5eaf2;
            --app-surface: #ffffff;
            --app-accent: #1769aa;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(23, 105, 170, .07), transparent 34%),
                #f5f7fb;
            color: var(--app-ink);
        }
        [data-testid="stHeader"] { background: rgba(245,247,251,.82); }
        [data-testid="stMainBlockContainer"] {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 { color: var(--app-ink); letter-spacing: -.025em; }
        h1 { font-weight: 760 !important; }
        [data-testid="stMetric"] {
            background: var(--app-surface);
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, .035);
        }
        [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--app-border) !important;
            border-radius: 16px !important;
            background: rgba(255,255,255,.88);
        }
        [data-testid="stDownloadButton"] button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            min-height: 2.8rem;
            font-weight: 650;
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--app-border);
            border-radius: 12px;
            overflow: hidden;
        }
        .app-eyebrow {
            color: var(--app-accent);
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }
        .app-subtitle {
            color: var(--app-muted);
            font-size: 1rem;
            line-height: 1.6;
            max-width: 850px;
            margin-top: -.25rem;
            margin-bottom: 1.25rem;
        }
        @media (max-width: 768px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }
        }
        </style>
        <div class="app-eyebrow">Manufacturing intelligence · Decision support prototype</div>
        """,
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)
    st.caption(
        "Enter customer product and order requirements to view candidate estimates "
        "for process parameters and production indicators."
    )
    st.info(
        "Synthetic-data proof of concept only. This candidate model has not been validated "
        "on Polyhose production data and must not be used as factory settings or production advice."
    )
    render_model_validation()

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
            st.session_state["prediction_generated_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat(timespec="seconds")

    predictions = st.session_state.get("validated_predictions")
    if predictions is not None:
        st.divider()
        st.info(REPORT_DISCLAIMER)
        for warning in st.session_state.get("prediction_warnings", ()):
            st.warning(warning)
        render_results(predictions)
        render_input_summary(st.session_state["submitted_inputs"])
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
