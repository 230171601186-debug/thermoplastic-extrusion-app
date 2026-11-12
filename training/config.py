"""Project-wide constants for the thermoplastic extrusion ML training stack.

Paths are resolved from this file so scripts work regardless of the
current working directory. Do not hard-code machine-specific paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project root: APP/)
# ---------------------------------------------------------------------------
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
# The checked-in file is CSV. An .xlsx name appears in some docs; we never
# invent data — load_dataset() will use whichever of these actually exists.
DATASET_CANDIDATES = (
    DATASET_DIR / "Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv",
    DATASET_DIR / "Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.xlsx",
)

MODELS_DIR = PROJECT_ROOT / "models"
PROCESS_MODELS_DIR = MODELS_DIR / "process"
EFFICIENCY_MODELS_DIR = MODELS_DIR / "efficiency"
QUALITY_MODELS_DIR = MODELS_DIR / "quality"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SPLIT_INDICES_PATH = MODELS_DIR / "data_split.json"

RESULTS_DIR = TRAINING_DIR / "results"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Modest, fixed hyperparameters. We compare algorithms with CV rather than
# running an expensive search that would overfit this 5,000-row synthetic set.
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}
XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}
LGBM_PARAMS = {
    "n_estimators": 250,
    "max_depth": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

# ---------------------------------------------------------------------------
# Model 1 — customer/product inputs and process-parameter targets
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "Material_Type",
    "Material_Grade",
    "Tube_Type",
    "Braid_Type",
]

CUSTOMER_NUMERIC_FEATURES = [
    "Target_Tube_OD_mm",
    "Target_Tube_Wall_mm",
    "Target_Tube_ID_mm",
    "Customer_Required_Length_m",
]

# The planned Streamlit UI will collect only these fields.
MODEL1_FEATURES = CATEGORICAL_FEATURES + CUSTOMER_NUMERIC_FEATURES

PROCESS_TARGETS = [
    "Extruder_Temperature_C",
    "Die_Temperature_C",
    "Cooling_Tank_Temperature_C",
    "Melt_Pressure_bar",
    "Air_Pressure_bar",
    "Screw_RPM",
    "Line_Speed_m_min",
    "Puller_Speed_m_min",
    "Extrusion_Speed_kg_hr",
]

# Saved pipeline filenames (one independent regressor per process target).
PROCESS_MODEL_FILENAMES = {
    "Extruder_Temperature_C": "extruder_temperature_model.pkl",
    "Die_Temperature_C": "die_temperature_model.pkl",
    "Cooling_Tank_Temperature_C": "cooling_temperature_model.pkl",
    "Melt_Pressure_bar": "melt_pressure_model.pkl",
    "Air_Pressure_bar": "air_pressure_model.pkl",
    "Screw_RPM": "screw_rpm_model.pkl",
    "Line_Speed_m_min": "line_speed_model.pkl",
    "Puller_Speed_m_min": "puller_speed_model.pkl",
    "Extrusion_Speed_kg_hr": "extrusion_rate_model.pkl",
}

# ---------------------------------------------------------------------------
# Model 2 / 3 targets (defined now so leakage lists stay consistent)
# ---------------------------------------------------------------------------
EFFICIENCY_TARGET = "Operational_Efficiency_pct"
SCRAP_TARGET = "Scrap_Rate"

# Post-production / outcome columns. Using any of these as predictors would
# leak the answer or information that is not known at planning time.
POST_PRODUCTION_LEAKAGE = [
    "Operational_Efficiency_pct",
    "OEE",
    "Availability",
    "Performance",
    "Quality",
    "Good_Length_m",
    "Scrap_Length_m",
    "Scrap_Rate",
    "Actual_Processing_Length_m",
    "Length_Shortfall_m",
    "Order_Status",
    "Fault_Code",
    "Power_Consumption_kWh",
    "Downtime_min",
    "Run_Time_min",
]

# Known at planning time in some plants, but NOT customer-facing inputs in
# the planned UI, so they are withheld from Model 1 (and from the stacked
# Model 2/3 customer branch) unless a later study justifies them.
PLANNING_CONTEXT_EXCLUDED = [
    "Record_ID",
    "Machine_ID",
    "Shift",
    "Ambient_Temperature_C",
    "Humidity_percent",
    "Planned_Scrap_Allowance_pct",
    "Planned_Processing_Length_m",
    "Planned_Production_Time_min",
]

DISCLAIMER = (
    "These models are a synthetic-data proof of concept for thermoplastic "
    "extrusion predictive analysis. They do not represent actual Polyhose "
    "machine behaviour."
)
