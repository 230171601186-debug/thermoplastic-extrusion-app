"""Validated inference entry point for the isolated Model 2/3 candidate bundle.

This module only loads candidate artifacts named in their manifests. It never
loads or writes production model artifacts or metadata.
"""

from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

TRAINING_DIR = Path(__file__).resolve().parents[3]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CUSTOMER_NUMERIC_FEATURES,
    EFFICIENCY_TARGET,
    MODEL1_FEATURES,
    PLANNING_CONTEXT_EXCLUDED,
    POST_PRODUCTION_LEAKAGE,
    PROCESS_MODEL_FILENAMES,
    PROCESS_TARGETS,
    SCRAP_TARGET,
)

CUSTOMER_FEATURES = MODEL1_FEATURES
STACKED_FEATURES = CUSTOMER_FEATURES + PROCESS_TARGETS
PROHIBITED_FEATURES = set(POST_PRODUCTION_LEAKAGE) | set(PLANNING_CONTEXT_EXCLUDED)
PREDICTION_UNITS = {
    "Material_Type": "categorical",
    "Material_Grade": "categorical",
    "Tube_Type": "categorical",
    "Braid_Type": "categorical; 'None' is a valid category",
    "Target_Tube_OD_mm": "mm",
    "Target_Tube_Wall_mm": "mm",
    "Target_Tube_ID_mm": "mm",
    "Customer_Required_Length_m": "m",
    "Extruder_Temperature_C": "degC",
    "Die_Temperature_C": "degC",
    "Cooling_Tank_Temperature_C": "degC",
    "Melt_Pressure_bar": "bar",
    "Air_Pressure_bar": "bar",
    "Screw_RPM": "rpm",
    "Line_Speed_m_min": "m/min",
    "Puller_Speed_m_min": "m/min",
    "Extrusion_Speed_kg_hr": "kg/hr",
    EFFICIENCY_TARGET: "percentage points",
    SCRAP_TARGET: "fraction",
}

# These limits are copied from the generator's existing assertions, not factory specifications.
PROCESS_OUTPUT_BOUNDS = {
    "Extruder_Temperature_C": (180.0, 245.0),
    "Die_Temperature_C": (180.0, 235.0),
    "Cooling_Tank_Temperature_C": (12.0, 28.0),
    "Melt_Pressure_bar": (90.0, 200.0),
    "Air_Pressure_bar": (3.0, 12.0),
    "Screw_RPM": (48.0, 100.0),
    "Line_Speed_m_min": (8.0, 24.0),
    "Puller_Speed_m_min": (7.0, 25.0),
    "Extrusion_Speed_kg_hr": (25.0, 170.0),
}


class InferenceValidationError(ValueError):
    """A caller-safe input, schema, or model-output validation failure."""


@dataclass(frozen=True)
class InferenceResult:
    predictions: pd.DataFrame
    warnings: tuple[str, ...]
    units: dict[str, str]


@dataclass
class _DownstreamBundle:
    target: str
    pipeline: Pipeline
    manifest: dict


class CandidateInferenceBundle:
    """Load and run the shared upstream and two downstream candidate pipelines."""

    def __init__(
        self,
        upstream_models: dict[str, Pipeline],
        model2: _DownstreamBundle,
        model3: _DownstreamBundle,
        category_values: dict[str, set[str]],
    ) -> None:
        self.upstream_models = upstream_models
        self.model2 = model2
        self.model3 = model3
        self.category_values = category_values

    @classmethod
    def load(cls, controlled_stacking_dir: str | Path | None = None) -> "CandidateInferenceBundle":
        """Load the candidate paths recorded in both manifests; never load production artifacts."""
        base = (
            Path(controlled_stacking_dir).resolve()
            if controlled_stacking_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        model2_dir = base / "model2_candidate"
        model3_dir = base / "model3_candidate"
        try:
            model2_manifest = json.loads(
                (model2_dir / "candidate_manifest.json").read_text(encoding="utf-8")
            )
            model3_manifest = json.loads(
                (model3_dir / "candidate_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise InferenceValidationError("Candidate manifests could not be loaded") from error

        cls._validate_manifest_pair(model2_manifest, model3_manifest)
        try:
            model2_paths = model2_manifest["candidate_artifacts"]["model1"]
            model3_paths = model3_manifest["candidate_artifacts"]["model1"]
            model2_downstream_path = Path(model2_manifest["candidate_artifacts"]["model2"])
            model3_downstream_path = Path(model3_manifest["candidate_artifacts"]["model3"])
        except (KeyError, TypeError) as error:
            raise InferenceValidationError("Candidate manifests are missing artifact paths") from error

        # Resolve manifest paths from the project root, preserving artifact layout as recorded.
        project_root = TRAINING_DIR.parent
        model2_upstream_paths = {
            target: project_root / Path(path) for target, path in model2_paths.items()
        }
        model3_upstream_paths = {
            target: project_root / Path(path) for target, path in model3_paths.items()
        }
        if set(model2_upstream_paths) != set(PROCESS_TARGETS) or set(model3_upstream_paths) != set(PROCESS_TARGETS):
            raise InferenceValidationError("Each candidate bundle must name all nine process pipelines")
        if set(model2_upstream_paths) != set(model3_upstream_paths):
            raise InferenceValidationError("Model 2 and Model 3 manifests reference different Model 1 targets")

        try:
            # Both candidate manifests must reference byte-identical Model 1 artifacts.
            # Hash the Model 3 paths, then load the shared pipelines only once.
            for target in PROCESS_TARGETS:
                if cls._file_sha256(model2_upstream_paths[target]) != cls._file_sha256(
                    model3_upstream_paths[target]
                ):
                    raise InferenceValidationError(
                        f"Model 1 {target}: candidate bundles contain different upstream artifacts"
                    )
            upstream = {
                target: joblib.load(path)
                for target, path in model2_upstream_paths.items()
            }
            downstream2_path = project_root / model2_downstream_path
            downstream3_path = project_root / model3_downstream_path
            downstream2 = joblib.load(downstream2_path)
            downstream3 = joblib.load(downstream3_path)
        except InferenceValidationError:
            raise
        except (OSError, ValueError, KeyError, ImportError) as error:
            raise InferenceValidationError("A candidate model artifact could not be loaded") from error

        cls._validate_pipeline_schemas(upstream, downstream2, downstream3)
        category_values = cls._category_values(downstream2, downstream3)
        return cls(
            upstream,
            _DownstreamBundle(EFFICIENCY_TARGET, downstream2, model2_manifest),
            _DownstreamBundle(SCRAP_TARGET, downstream3, model3_manifest),
            category_values,
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_manifest_pair(model2_manifest: dict, model3_manifest: dict) -> None:
        if model2_manifest.get("dataset_sha256") != model3_manifest.get("dataset_sha256"):
            raise InferenceValidationError("Candidate manifests refer to different datasets")
        if model2_manifest.get("seed") != model3_manifest.get("seed"):
            raise InferenceValidationError("Candidate manifests use different random seeds")
        if model2_manifest.get("split_definition") != model3_manifest.get("split_definition"):
            raise InferenceValidationError("Candidate manifests use different saved splits")
        if model2_manifest.get("feature_order") is None:
            m2_features = model2_manifest.get("features", {}).get("full_stack_order")
        else:
            m2_features = model2_manifest.get("feature_order")
        m3_features = model3_manifest.get("feature_order")
        if m2_features != STACKED_FEATURES or m3_features != STACKED_FEATURES:
            raise InferenceValidationError("Candidate manifest feature order differs from the 17-feature contract")
        if model2_manifest.get("features", {}).get("prohibited_intersection", []) or model3_manifest.get(
            "prohibited_feature_intersection", []
        ):
            raise InferenceValidationError("A candidate manifest lists prohibited inference features")
        if model2_manifest.get("downstream", {}).get("target") != EFFICIENCY_TARGET:
            raise InferenceValidationError("Model 2 candidate target is not operational efficiency")
        if model3_manifest.get("downstream", {}).get("target") != SCRAP_TARGET:
            raise InferenceValidationError("Model 3 candidate target is not scrap rate")

    @staticmethod
    def _validate_pipeline_schemas(upstream: dict, downstream2: Pipeline, downstream3: Pipeline) -> None:
        for target in PROCESS_TARGETS:
            pipeline = upstream.get(target)
            if not isinstance(pipeline, Pipeline):
                raise InferenceValidationError(f"Model 1 artifact for {target} is not a fitted pipeline")
            if list(getattr(pipeline, "feature_names_in_", [])) != CUSTOMER_FEATURES:
                raise InferenceValidationError(f"Model 1 {target} input feature order is incompatible")
            CandidateInferenceBundle._validate_encoder(pipeline, f"Model 1 {target}")
        for label, downstream in (("Model 2", downstream2), ("Model 3", downstream3)):
            if not isinstance(downstream, Pipeline):
                raise InferenceValidationError(f"{label} artifact is not a fitted pipeline")
            if list(getattr(downstream, "feature_names_in_", [])) != STACKED_FEATURES:
                raise InferenceValidationError(f"{label} input feature order is incompatible")
            CandidateInferenceBundle._validate_encoder(downstream, label)
        pipelines = [*(upstream[target] for target in PROCESS_TARGETS), downstream2, downstream3]
        baseline_categories = CandidateInferenceBundle._encoder_categories(pipelines[0])
        for label, pipeline in zip(
            [*(f"Model 1 {target}" for target in PROCESS_TARGETS), "Model 2", "Model 3"],
            pipelines,
        ):
            if CandidateInferenceBundle._encoder_categories(pipeline) != baseline_categories:
                raise InferenceValidationError(f"{label} categorical vocabularies differ across the candidate bundle")

    @staticmethod
    def _validate_encoder(pipeline: Pipeline, label: str) -> None:
        try:
            encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        except (KeyError, AttributeError) as error:
            raise InferenceValidationError(f"{label} is missing its fitted categorical encoder") from error
        if getattr(encoder, "handle_unknown", None) != "ignore":
            raise InferenceValidationError(f"{label} does not support unseen categories by ignoring them")

    @staticmethod
    def _encoder_categories(pipeline: Pipeline) -> tuple[tuple[str, ...], ...]:
        encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        return tuple(tuple(map(str, categories)) for categories in encoder.categories_)

    @staticmethod
    def _category_values(model2: Pipeline, model3: Pipeline) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for feature_index, feature in enumerate(CATEGORICAL_FEATURES):
            first = set(map(str, model2.named_steps["preprocess"].named_transformers_["cat"].categories_[feature_index]))
            second = set(map(str, model3.named_steps["preprocess"].named_transformers_["cat"].categories_[feature_index]))
            if first != second:
                raise InferenceValidationError(f"Model 2 and Model 3 categorical vocabularies differ for {feature}")
            result[feature] = first
        return result

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None or value is pd.NA:
            return True
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False

    def validate_customer_inputs(self, values: pd.DataFrame | Mapping[str, object]) -> tuple[pd.DataFrame, tuple[str, ...]]:
        if isinstance(values, Mapping):
            frame = pd.DataFrame([dict(values)])
        elif isinstance(values, pd.DataFrame):
            frame = values.copy()
        else:
            raise InferenceValidationError("inputs: expected a mapping or pandas DataFrame")

        if frame.empty:
            raise InferenceValidationError("inputs: at least one row is required")
        missing_fields = [field for field in CUSTOMER_FEATURES if field not in frame.columns]
        if missing_fields:
            field = missing_fields[0]
            raise InferenceValidationError(f"{field}: required input field is missing")
        unexpected_fields = [field for field in frame.columns if field not in CUSTOMER_FEATURES]
        if unexpected_fields:
            raise InferenceValidationError(f"{unexpected_fields[0]}: unexpected input field")

        frame = frame[CUSTOMER_FEATURES].copy()
        warnings: list[str] = []
        for feature in CATEGORICAL_FEATURES:
            for row_index, value in frame[feature].items():
                if self._is_missing(value):
                    raise InferenceValidationError(f"{feature}: value is required; missing values are not accepted")
                if not isinstance(value, str):
                    raise InferenceValidationError(f"{feature}: expected a nonblank category string")
                if not value.strip():
                    raise InferenceValidationError(f"{feature}: blank category is not allowed")
                if value not in self.category_values[feature]:
                    if self._unknown_handling(feature) != "ignore":
                        raise InferenceValidationError(f"{feature}: unsupported category")
                    warnings.append(
                        f"row {row_index}: unseen {feature} category accepted; fitted encoder ignores unknown categories"
                    )

        for feature in CUSTOMER_NUMERIC_FEATURES:
            converted = []
            for row_index, value in frame[feature].items():
                if self._is_missing(value):
                    raise InferenceValidationError(f"{feature}: numeric value is required; missing values are not accepted")
                if isinstance(value, (bool, np.bool_)):
                    raise InferenceValidationError(f"{feature}: boolean is not a valid numeric input")
                if isinstance(value, str) and not value.strip():
                    raise InferenceValidationError(f"{feature}: blank numeric value is not allowed")
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as error:
                    raise InferenceValidationError(f"{feature}: expected a numeric value") from error
                if not np.isfinite(number):
                    raise InferenceValidationError(f"{feature}: value must be finite")
                converted.append(number)
            frame[feature] = np.asarray(converted, dtype=np.float64)

        for feature in ("Target_Tube_OD_mm", "Target_Tube_Wall_mm", "Customer_Required_Length_m"):
            if (frame[feature] <= 0).any():
                raise InferenceValidationError(f"{feature}: value must be greater than zero")
        if (frame["Target_Tube_ID_mm"] <= 0).any():
            raise InferenceValidationError("Target_Tube_ID_mm: value must be positive")
        if (frame["Target_Tube_ID_mm"] <= 0.5).any():
            raise InferenceValidationError(
                "Target_Tube_ID_mm: value must exceed 0.5 mm under the generator's geometry rule"
            )
        geometry_error = (
            frame["Target_Tube_ID_mm"]
            - (frame["Target_Tube_OD_mm"] - 2 * frame["Target_Tube_Wall_mm"])
        ).abs()
        if (geometry_error >= 0.25).any():
            raise InferenceValidationError(
                "Target_Tube_ID_mm: geometry must satisfy |ID - (OD - 2*wall)| < 0.25 mm"
            )
        return frame, tuple(dict.fromkeys(warnings))

    def _unknown_handling(self, feature: str) -> str:
        encoder = self.model2.pipeline.named_steps["preprocess"].named_transformers_["cat"]
        feature_index = CATEGORICAL_FEATURES.index(feature)
        if feature_index >= len(encoder.categories_):
            return "reject"
        return str(encoder.handle_unknown)

    @staticmethod
    def _validate_stack(stack: pd.DataFrame, row_count: int) -> None:
        if list(stack.columns) != STACKED_FEATURES:
            raise InferenceValidationError("stack: process feature columns or order differ from the 17-feature contract")
        missing = stack.isna().any()
        if missing.any():
            field = str(missing[missing].index[0])
            raise InferenceValidationError(f"{field}: missing value in model input stack")
        if len(stack) != row_count:
            raise InferenceValidationError("stack: row count differs from customer inputs")
        prohibited = set(stack.columns) & PROHIBITED_FEATURES
        if prohibited:
            raise InferenceValidationError(f"{sorted(prohibited)[0]}: prohibited downstream feature")

    @staticmethod
    def _validate_process_output(target: str, values: object, row_count: int) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != (row_count,):
            raise InferenceValidationError(f"{target}: expected {row_count} predictions, received shape {array.shape}")
        if not np.issubdtype(array.dtype, np.number):
            raise InferenceValidationError(f"{target}: predictions must be numeric")
        array = array.astype(np.float64, copy=False)
        if not np.isfinite(array).all():
            raise InferenceValidationError(f"{target}: model returned NaN or infinite predictions")
        lower, upper = PROCESS_OUTPUT_BOUNDS[target]
        if ((array < lower) | (array > upper)).any():
            raise InferenceValidationError(
                f"{target}: prediction outside existing synthetic bounds [{lower}, {upper}]"
            )
        return array

    @staticmethod
    def _validate_target_output(target: str, values: object, row_count: int) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != (row_count,):
            raise InferenceValidationError(f"{target}: expected {row_count} predictions, received shape {array.shape}")
        if not np.issubdtype(array.dtype, np.number):
            raise InferenceValidationError(f"{target}: predictions must be numeric")
        array = array.astype(np.float64, copy=False)
        if not np.isfinite(array).all():
            raise InferenceValidationError(f"{target}: model returned NaN or infinite predictions")
        upper = 100.0 if target == EFFICIENCY_TARGET else 1.0
        if ((array < 0.0) | (array > upper)).any():
            unit = "percentage points" if target == EFFICIENCY_TARGET else "fraction"
            raise InferenceValidationError(f"{target}: prediction outside documented [0, {upper}] {unit} bounds")
        return array

    def predict(self, values: pd.DataFrame | Mapping[str, object]) -> InferenceResult:
        """Validate customer rows, predict Model 1 processes, then Model 2 and Model 3."""
        customer, warnings = self.validate_customer_inputs(values)
        missing_models = [target for target in PROCESS_TARGETS if target not in self.upstream_models]
        if missing_models:
            raise InferenceValidationError(f"{missing_models[0]}: required Model 1 pipeline is missing")
        process_predictions = {}
        for target in PROCESS_TARGETS:
            try:
                raw = self.upstream_models[target].predict(customer[CUSTOMER_FEATURES])
            except Exception as error:
                raise InferenceValidationError(f"{target}: Model 1 inference failed for validated inputs") from None
            process_predictions[target] = self._validate_process_output(target, raw, len(customer))

        stack = customer[CUSTOMER_FEATURES].copy()
        for target in PROCESS_TARGETS:
            stack[target] = process_predictions[target]
        self._validate_stack(stack, len(customer))

        try:
            raw_efficiency = self.model2.pipeline.predict(stack)
        except Exception:
            raise InferenceValidationError(f"{EFFICIENCY_TARGET}: Model 2 inference failed for validated inputs") from None
        try:
            raw_scrap = self.model3.pipeline.predict(stack)
        except Exception:
            raise InferenceValidationError(f"{SCRAP_TARGET}: Model 3 inference failed for validated inputs") from None
        efficiency = self._validate_target_output(EFFICIENCY_TARGET, raw_efficiency, len(customer))
        scrap = self._validate_target_output(SCRAP_TARGET, raw_scrap, len(customer))
        output = stack[PROCESS_TARGETS].copy()
        output[EFFICIENCY_TARGET] = efficiency
        output[SCRAP_TARGET] = scrap
        return InferenceResult(output, warnings, dict(PREDICTION_UNITS))
