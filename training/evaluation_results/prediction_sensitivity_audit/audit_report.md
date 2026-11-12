# Candidate Prediction Sensitivity Audit

## Scope

Read-only one-factor audit using dataset SHA-256 `cbd42684bc475ca9a7ad4f5b4512962c785034f2aae7c4944f0804e624c20396` (5000 rows x 40 columns). Baseline customer inputs come from dataset Record_ID `1`. No models were fit or changed.

Tested 45 scenarios: 45 accepted by the inference validator and 0 rejected. Every scenario changes one requested customer input while holding the other seven at the baseline.

## Geometry and categorical coverage

Categorical values are enumerated from fitted candidate encoder vocabularies; `Braid_Type='None'` is included. Geometry rule from the existing synthetic generator: `abs(ID - (OD - 2*wall)) < 0.25 mm; ID > 0.5 mm`.

OD, wall, and ID cannot be broadly varied independently while satisfying the geometry rule. Their one-factor tests therefore use only local offsets within the existing tolerance; this is a local model sensitivity check, not a broad geometry sweep. Required-length tests use dataset quantiles. No new engineering limits are imposed.

## Input influence screening

Influence is ranked by maximum absolute output change divided by that output's full-dataset standard deviation. A change over 1 dataset SD is flagged only as a statistical screening heuristic, not an engineering significance threshold.

- `Extruder_Temperature_C` (degC): largest one-factor local response was `Material_Type` (max |delta| 11.445, 1.475 dataset SD over tested cases).
- `Die_Temperature_C` (degC): largest one-factor local response was `Material_Type` (max |delta| 10.653, 1.251 dataset SD over tested cases).
- `Cooling_Tank_Temperature_C` (degC): largest one-factor local response was `Tube_Type` (max |delta| 1.2319, 0.783 dataset SD over tested cases).
- `Melt_Pressure_bar` (bar): largest one-factor local response was `Material_Type` (max |delta| 6.0161, 0.707 dataset SD over tested cases).
- `Air_Pressure_bar` (bar): largest one-factor local response was `Braid_Type` (max |delta| 0.54423, 0.576 dataset SD over tested cases).
- `Screw_RPM` (rpm): largest one-factor local response was `Material_Grade` (max |delta| 4.9408, 0.691 dataset SD over tested cases).
- `Line_Speed_m_min` (m/min): largest one-factor local response was `Customer_Required_Length_m` (max |delta| 1.9541, 0.883 dataset SD over tested cases).
- `Puller_Speed_m_min` (m/min): largest one-factor local response was `Customer_Required_Length_m` (max |delta| 1.4848, 0.642 dataset SD over tested cases).
- `Extrusion_Speed_kg_hr` (kg/hr): largest one-factor local response was `Tube_Type` (max |delta| 6.5476, 0.550 dataset SD over tested cases).
- `Operational_Efficiency_pct` (percentage_points): largest one-factor local response was `Tube_Type` (max |delta| 6.5707, 0.706 dataset SD over tested cases).
- `Scrap_Rate` (fraction): largest one-factor local response was `Material_Type` (max |delta| 0.025867, 0.790 dataset SD over tested cases).

No-change pairs (all tested values identical to baseline within 1e-12): Target_Tube_OD_mm → Extruder_Temperature_C, Target_Tube_OD_mm → Die_Temperature_C, Target_Tube_OD_mm → Air_Pressure_bar, Target_Tube_Wall_mm → Cooling_Tank_Temperature_C, Target_Tube_ID_mm → Extruder_Temperature_C, Target_Tube_ID_mm → Die_Temperature_C, Target_Tube_ID_mm → Cooling_Tank_Temperature_C, Target_Tube_ID_mm → Extrusion_Speed_kg_hr.
Changes greater than 1 full-dataset SD (heuristic): Material_Type → Extruder_Temperature_C, Material_Type → Die_Temperature_C.

## Distribution and relationship checks

Across accepted scenario-output pairs: 0 predictions violated existing documented bounds, 0 fell outside the dataset observed min/max, and 0 fell outside the dataset q01–q99 range.
Predicted absolute line/puller speed gaps exceeded the synthetic dataset q99 in 0 of 45 accepted cases. Screw-RPM/extrusion-rate correlations are in `relationship_diagnostics.csv`.
Output quantiles and existing synthetic bounds are in `dataset_output_distributions.csv`; per-case empirical-range flags are in `one_factor_predictions.csv`. These screens are not validated manufacturing limits.

## Saved evaluation weaknesses

- Model 2 corrected nested full-stack CV: MAE=6.4234 ± 0.18939, RMSE=8.3809 ± 0.34904, R²=0.1739 ± 0.0321.
- Model 2 selected saved held-out test: MAE=6.406, RMSE=8.4618, R²=0.2276.
- Model 3 corrected nested full-stack CV: MAE=0.021876 ± 0.00051987, RMSE=0.026933 ± 0.00056706, R²=0.3222 ± 0.0375.
- Model 3 selected saved held-out test: MAE=0.020436, RMSE=0.026646, R²=0.3347.
- Model 1 weaker saved test target `Cooling_Tank_Temperature_C`: R²=0.2869, MAE=0.88726, RMSE=1.2648.
- Model 1 weaker saved test target `Air_Pressure_bar`: R²=0.4459, MAE=0.54751, RMSE=0.68457.
- Model 1 weaker saved test target `Die_Temperature_C`: R²=0.5097, MAE=4.758, RMSE=5.9909.

The candidate Model 2/3 test scores and Model 1 target metrics are historical saved estimates; sensitivity results here do not replace nested-CV or held-out evaluation.

## Limitations and Next Steps

All artifacts and data are synthetic. Local geometric perturbations are intentionally narrow because the repository defines a coupled OD/wall/ID rule. One-factor predictions do not establish causal effects or factory performance. Review rejected scenarios and high delta/SD rows in the CSVs; use real, validated production data and engineering review before operational use.

No dataset, generator, model artifact, metadata, inference logic, or Streamlit UI is modified by this audit.
