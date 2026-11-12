# Pipeline lineage and artifact disposition

## Scope and decision

This is an isolated investigation of the existing synthetic-data proof of concept. No production dataset, generator, model artifact, metadata, inference code, app, or deployment contract was edited. No model was promoted. Hash evidence is in `artifact_integrity_manifest.json` and the final verification is in `preservation_verification.json`.

**Decision:** the Streamlit application currently serves the *controlled-stacking candidate* bundle. This is explicit in source code, not an accidental fallback to `models/`. Whether that candidate-serving configuration was authorized as the current demo build is unknown. Retain the clear synthetic-data disclaimer, do not describe outputs as factory-validated, and obtain stakeholder approval of the bundle choice before presenting it as a production model. No model replacement is supported by this experiment.

## Prediction graph

`app/streamlit_app.py` maps exactly eight UI fields in `training.config.MODEL1_FEATURES` into `CandidateInferenceBundle`:

1. Nine independent Model 1 process pipelines, each using the eight customer inputs.
2. Model 2 consumes those eight fields plus the nine Model 1 *predictions* (17 ordered inputs) and estimates `Operational_Efficiency_pct`.
3. Model 3 consumes the same 17 ordered inputs and estimates `Scrap_Rate`.

At no stage does the inference stack include recorded process outcomes, OEE components, downtime, run time, scrap/good lengths, fault codes, or other post-run labels. `Braid_Type="None"` is a valid category. The app converts scrap fraction to percent for display; Model 2 is already in percentage points.

## Active app artifacts

`app/streamlit_app.py` imports `training/evaluation_results/controlled_stacking/bundle_audit/candidate_inference.py`; its default loader reads both candidate manifests and the artifact files named there. It does **not** load the `models/` pickle files.

| Stage | Active bundle artifact | Target / features | Training and model provenance |
|---|---|---|---|
| Model 1 (9 pipelines) | `training/evaluation_results/controlled_stacking/model2_candidate/candidate_artifacts/` (verified byte-identical per-target copies also appear in `model3_candidate/candidate_artifacts/`) | Each process target; eight customer fields | Selected per-target candidates: LightGBM for extruder temperature, cooling temperature, melt pressure, air pressure, screw RPM, line speed, puller speed, and extrusion rate; XGBoost for die temperature. Selected by the isolated controlled-stacking training procedure. The loaded transforms are fitted `ColumnTransformer` pipelines containing categorical `OneHotEncoder(handle_unknown="ignore")` and passed-through numerics. |
| Model 2 | `.../model2_candidate/candidate_artifacts/efficiency_model_candidate.pkl` | `Operational_Efficiency_pct`; 17 ordered inputs | RandomForestRegressor: 160 trees, depth 20, leaf minimum 2, seed 42. Candidate manifest records the existing 5,000 × 40 synthetic CSV SHA-256 `cbd42684bc475ca9a7ad4f5b4512962c785034f2aae7c4944f0804e624c20396`, seed 42, and saved 4,000/1,000 split. |
| Model 3 | `.../model3_candidate/candidate_artifacts/scrap_rate_model_candidate.pkl` | `Scrap_Rate`; same 17 ordered inputs | RandomForestRegressor: 160 trees, depth 20, leaf minimum 2, seed 42. Same dataset, seed, and saved split. Scrap target unit is a fraction. |

The manifests specify the same dataset hash, split definition, and feature order. The two active candidate bundles contain byte-identical Model 1 files for all nine targets. The independent hash list for loaded app artifacts is also preserved in `experiment_manifest.json`. The metadata in `models/model_metadata.json` instead records the separate model files under `models/process`, `models/efficiency`, and `models/quality`; it is not a manifest for the candidate bundle the app loads.

## Production artifact comparison

The `models/` files are not the active app artifacts. Their `models/model_metadata.json` records the same synthetic dataset and split context, but different trained artifacts and reported held-out scores:

| Production metadata section | Recorded selected families | Recorded held-out MAE / RMSE / R² |
|---|---|---|
| Model 1 | LightGBM, XGBoost, or Random Forest, target-specific | Per-target metrics are preserved in the existing metadata. They are not metrics for the active candidate files. |
| Model 2 | Random Forest | 6.4060 pp / 8.4618 pp / 0.2276 |
| Model 3 | XGBoost | 0.020436 fraction / 0.026646 fraction / 0.3347 |

The candidate Model 2/3 artifacts use fixed Random Forests, so neither their identity nor their results can be inferred from the production metadata. The app's candidate-only import and the candidate inference module's explicit “never load production artifacts” behavior establish the technical routing. Repository evidence alone does not establish who authorized it.

## Training / inference consistency

- The active artifacts are sklearn `Pipeline`s that include their fitted preprocessing; inference validates the fitted input feature names, ordered feature lists, encoder policy, and matching category vocabularies.
- Both downstream models receive the exact eight customer features followed by the nine named Model 1 predictions. They do not receive recorded process targets.
- Candidate bundle tests verify stage targets, shared Model 1 hashes, finite/bounded outputs, units, and unknown-category warnings.
- Existing training scripts for production models have a cross-validation weakness: Model 2 creates one Model 1 OOF matrix across the full training partition, then evaluates downstream folds on that already-created matrix; an outer validation fold can therefore have influenced upstream fits generating some other fold's training rows. Model 3's legacy training path stacks recorded Model 1 predictions made by models fitted on the entire training partition, making its downstream training stack in-sample. The held-out test rows remain separate. These CV limitations do not mean post-run target leakage into inference, but they make old CV estimates unsuitable as clean nested-CV evidence.
- This experiment corrects that issue: for each of five outer folds, all upstream preprocessing/model fitting is limited to the outer training rows; the outer training stack is built from four-fold row-excluded OOF predictions; validation process predictions come from pipelines fitted on the outer training partition. The 1,000 saved test rows were evaluated only after CV, never for selection.

## Limits

Every dataset and score discussed here is synthetic evidence. The candidate artifacts and metadata establish file lineage and reproducibility, not the accuracy of any process recommendation on a Polyhose line. Real machine, shift, operator, material-lot, maintenance, environmental, schedule, and label-accounting behavior is not established by these files.
