"""Shared data loading and sklearn preprocessing for all three ML tasks.

Categorical encoding lives inside a ColumnTransformer so training and later
inference (Streamlit) use the same fitted objects. Never hand-map categories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import joblib

from config import (
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    DATASET_CANDIDATES,
    MODEL1_FEATURES,
    PROCESS_MODEL_FILENAMES,
    PROCESS_MODELS_DIR,
    PROCESS_TARGETS,
    RANDOM_STATE,
    SPLIT_INDICES_PATH,
    TEST_SIZE,
)


# pandas' default NA list includes the token "None". In this dataset
# Braid_Type uses "None" as a real product category (unbraided tube),
# so that token must be preserved. Empty cells and standard NA strings
# are still treated as missing. The source file is not modified.
_CSV_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
]


def load_dataset() -> pd.DataFrame:
    """Load the existing extrusion dataset. Does not create or alter rows."""
    for path in DATASET_CANDIDATES:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(
                path,
                keep_default_na=False,
                na_values=_CSV_NA_VALUES,
            )
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
            # Excel loaders can also coerce the braid label "None" to NaN.
            if "Braid_Type" in df.columns and df["Braid_Type"].isna().any():
                n_null = int(df["Braid_Type"].isna().sum())
                df["Braid_Type"] = df["Braid_Type"].apply(
                    lambda v: "None" if pd.isna(v) else v
                )
                print(
                    f"[data] Restored {n_null} Braid_Type values from Excel NA "
                    "to the category 'None' (unbraided). File on disk unchanged."
                )
        else:
            continue
        print(f"[data] Loaded {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
        if "Braid_Type" in df.columns:
            print(
                "[data] Braid_Type includes the explicit category 'None' "
                "(unbraided product), not missing data."
            )
        return df

    tried = "\n  ".join(str(p) for p in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "No dataset file found. Looked for:\n  "
        + tried
        + "\nThe original file must be present; training will not synthesise data."
    )


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: required columns are missing from the dataset: {missing}. "
            "Training stopped. Columns will not be invented."
        )


def build_customer_preprocessor() -> ColumnTransformer:
    """One-hot encode customer categoricals; pass numeric columns through.

    Tree models do not need scaling. Skipping a scaler keeps inference simple
    and avoids transforming diameters/length into z-scores at prediction time.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", CUSTOMER_NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_regressor_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_customer_preprocessor()),
            ("model", estimator),
        ]
    )


def build_stacked_preprocessor(process_numeric_features: list[str]) -> ColumnTransformer:
    """Customer categoricals + customer numerics + predicted process numerics.

    Process columns must be Model 1 predictions (or the same columns at
    inference time), never post-production outcomes.
    """
    numeric = list(CUSTOMER_NUMERIC_FEATURES) + list(process_numeric_features)
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_stacked_pipeline(estimator, process_numeric_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_stacked_preprocessor(process_numeric_features)),
            ("model", estimator),
        ]
    )


def load_process_pipelines() -> dict:
    """Load the fitted Model 1 pipelines. Fails if Model 1 has not been trained."""
    missing = []
    pipelines = {}
    for target, filename in PROCESS_MODEL_FILENAMES.items():
        path = PROCESS_MODELS_DIR / filename
        if not path.exists():
            missing.append(str(path))
        else:
            pipelines[target] = joblib.load(path)
    if missing:
        raise FileNotFoundError(
            "Model 1 artifacts are missing. Run training/model1_process_parameters.py first.\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    print(f"[stack] Loaded {len(pipelines)} Model 1 process pipelines")
    return pipelines


def attach_predicted_process(X_customer: pd.DataFrame, pipelines: dict) -> pd.DataFrame:
    """Append Model 1 predicted process parameters to customer/product inputs."""
    require_columns(X_customer, MODEL1_FEATURES, "attach_predicted_process")
    stacked = X_customer[MODEL1_FEATURES].copy()
    X_m1 = X_customer[MODEL1_FEATURES]
    for target in PROCESS_TARGETS:
        stacked[target] = pipelines[target].predict(X_m1)
    return stacked


def customer_xy(df: pd.DataFrame, target_columns: list[str]):
    require_columns(df, CATEGORICAL_FEATURES + CUSTOMER_NUMERIC_FEATURES + target_columns, "customer_xy")
    X = df[CATEGORICAL_FEATURES + CUSTOMER_NUMERIC_FEATURES].copy()
    y = df[target_columns].copy()
    return X, y


def split_frame(
    X: pd.DataFrame,
    y: pd.DataFrame,
    record_ids: pd.Series | None = None,
    save_split: bool = True,
):
    """80/20 split with a fixed seed. Indices are saved so Model 2/3 reuse it."""
    split_kwargs = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "shuffle": True,
    }
    if record_ids is not None:
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, record_ids, **split_kwargs
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        id_train = id_test = None

    print(
        f"[split] train={len(X_train)} ({100 * (1 - TEST_SIZE):.0f}%)  "
        f"test={len(X_test)} ({100 * TEST_SIZE:.0f}%)  random_state={RANDOM_STATE}"
    )
    print("[split] Test rows are held out until final evaluation. CV uses training only.")

    if save_split:
        payload = {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_index": X_train.index.tolist(),
            "test_index": X_test.index.tolist(),
            "train_record_ids": None if id_train is None else id_train.tolist(),
            "test_record_ids": None if id_test is None else id_test.tolist(),
        }
        SPLIT_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_INDICES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[split] Saved split indices to {SPLIT_INDICES_PATH.relative_to(SPLIT_INDICES_PATH.parents[1])}")

    return X_train, X_test, y_train, y_test


def load_saved_split() -> dict:
    if not SPLIT_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SPLIT_INDICES_PATH}. Run Model 1 first so Model 2/3 share the same hold-out set."
        )
    return json.loads(SPLIT_INDICES_PATH.read_text(encoding="utf-8"))


def apply_saved_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = load_saved_split()
    train = df.loc[payload["train_index"]]
    test = df.loc[payload["test_index"]]
    return train, test


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocess = pipeline.named_steps["preprocess"]
    return list(preprocess.get_feature_names_out())


def merge_metadata(section_name: str, section: dict) -> None:
    """Update models/model_metadata.json without wiping Model 1 entries."""
    from datetime import datetime, timezone

    from config import DISCLAIMER, METADATA_PATH

    if METADATA_PATH.exists():
        payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    else:
        payload = {"disclaimer": DISCLAIMER}
    payload[section_name] = section
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[meta] Updated {METADATA_PATH.name} section '{section_name}'")
