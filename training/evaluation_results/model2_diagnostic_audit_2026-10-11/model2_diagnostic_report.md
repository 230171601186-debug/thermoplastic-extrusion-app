# Read-only Model 2 Diagnostic Audit

## Scope and reproducibility

This report audits revised synthetic dataset `model2_generator_experiment_2026-10-11_revised/experimental_dataset.csv`; it does not alter production data, models, metadata, inference code, or the Streamlit app. No candidate was promoted. Dataset SHA-256: `0db6c6d33f0a3d580b2f673cc3761b4c2551f48c1ca68d296303d9854a669b15`. The saved split has 4000 training rows and 1000 held-out rows. Five-fold outer CV is confined to the training partition, with three-fold inner tuning and feature selection. The saved held-out set was not used for model selection; its candidate scores are descriptive.

The predictor uses the eight documented customer/product inputs plus nine predictions from the saved production Model 1 inference pipelines. Recorded process outcomes and post-production accounting columns are excluded. See `data_lineage_and_target_diagnostics.json` and `feature_role_and_leakage_audit.csv` for feature lineage and risk classifications.

## Historical production baseline (different dataset; not paired)

The saved production Model 2 achieved held-out MAE=6.4060, RMSE=8.4618, R²=0.2276 on the production dataset. Corrected controlled full-stack outer CV on that production dataset was MAE=6.4234 ± 0.1894, RMSE=8.3809 ± 0.3490, R²=0.1739 ± 0.0321. These are historical context only: they use the production target population and saved production evaluation, whereas this report evaluates a revised generated target using frozen production Model 1 predictions and newly fitted Model 2 estimators. The numerical scores must not be interpreted as paired gains or losses across datasets. Detailed recorded baseline tables are in [baseline_existing_production_model2.csv](../model2_generator_experiment_2026-10-11/baseline_existing_production_model2.csv) and [baseline_existing_dataset_nested_summary.csv](../model2_generator_experiment_2026-10-11/baseline_existing_dataset_nested_summary.csv).

## Why Model 2 remains weak

The best training-only family has outer-CV R²=0.0877 ± 0.0114; its improvement over a training-fold mean baseline is limited. In identical RF outer folds, adding all nine process predictions to customer inputs changes MAE from 4.7848 to 4.7706 pp (a 0.0142 pp reduction), and selecting process inputs inside the folds changes it to 4.7571 pp. Process-only MAE is 4.7801 pp. This indicates that the current predicted process features provide only a small incremental signal in this revised synthetic dataset.

The frozen Model 1 predictors also have negative R² for revised-dataset line speed and puller speed:

| process_target | MAE | RMSE | R2 | actual_mean | prediction_mean |
| --- | --- | --- | --- | --- | --- |
| Line_Speed_m_min | 1.8833 | 2.0763 | -0.4006 | 12.5045 | 14.3548 |
| Puller_Speed_m_min | 1.9103 | 2.1827 | -0.3206 | 12.5475 | 14.3794 |

The target additionally depends on realized downtime, faults, runtime and scrap/quality; these are post-run target components, not members of the 17-input inference contract. The process-error and subgroup results are consistent with unobserved operational outcomes limiting prediction, but do not establish a real-plant causal explanation. The held-out Machine Breakdown subgroup is particularly uncertain because its sample is small.

## Target and synthetic accounting audit

The revised experimental definition is `Operational_Efficiency_pct = 100 × OEE`, where `OEE = Availability × Performance × Quality`. The dataset's target mean/std are 77.092/6.226 pp (range 16.546–92.418); 0 records have Performance ≥ 1 and 0 targets are ≥100. No upper clipping is applied in this experiment.

Order completion is explicitly defined as **Completed iff `Good_Length_m >= Customer_Required_Length_m`**. `Length_Shortfall_m` remains the separate planned-versus-actual processing-length measure. Automated row-level checks cover geometry, actual/good/scrap length, scrap rate, shortfall, runtime, availability, quality, Performance bounds, OEE, efficiency, completion status, row/schema integrity, and categorical-level preservation.

| check | passed | max_abs_residual | tolerance |
| --- | --- | --- | --- |
| tube_geometry_mm | True | 0.0100000 | 0.0110000 |
| actual_length_m | True | 0.0050000 | 0.0051000 |
| good_plus_scrap_m | True | 0.0000000 | 0.0051000 |
| good_length_m | True | 0.0000000 | 0.0051000 |
| scrap_length_m | True | 0.0050000 | 0.0051000 |
| length_shortfall_m | True | 0.0000000 | 0.0000000 |
| run_time_min | True | 0.0100000 | 0.0101000 |
| availability | True | 0.0000005 | 0.0000010 |
| quality | True | 0.0000005 | 0.0000010 |
| performance_bounds | True | 0.0000000 | 0.0000000 |
| oee | True | 0.0000015 | 0.0000020 |
| operational_efficiency_pct | True | 0.0000000 | 0.0000000 |
| order_completion | True |  |  |

Overall accounting validation passed: **True**. Full check details are in `dataset_accounting_validation.json`. The prior revised generator audit records the nominal 12–22 m/min recipe-reference envelope and process ranges as **synthetic assumptions, not manufacturing evidence or Polyhose-specific operating limits**. Model 1 prediction errors on this revised synthetic distribution are listed in `model1_process_prediction_diagnostics.csv`.

## Leakage-controlled performance

Outer-CV family comparisons use the same five folds and training-only inner hyperparameter selection. Values are mean ± standard deviation across outer folds. Candidate models are all compared on identical rows/features within this revised dataset; scores from different dataset versions are not treated as paired measurements.

| model | fold_count | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std | full_training_inner_MAE | family_selected_for_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | 5 | 4.6744 | 0.1478 | 5.8065 | 0.1706 | 0.0877 | 0.0114 | 4.6996 | True |
| LightGBM | 5 | 4.7023 | 0.1415 | 5.8359 | 0.1613 | 0.0782 | 0.0222 | 4.7075 | False |
| HistGradientBoosting | 5 | 4.7080 | 0.1322 | 5.8507 | 0.1664 | 0.0736 | 0.0228 | 4.7180 | False |
| ExtraTrees | 5 | 4.7652 | 0.1395 | 5.9086 | 0.1745 | 0.0552 | 0.0238 | 4.7598 | False |
| RandomForest | 5 | 4.7709 | 0.1081 | 5.9134 | 0.1384 | 0.0534 | 0.0267 | 4.7383 | False |

### Feature-group ablation

All feature-group configurations use identical outer folds. "Predicted process only" uses the nine frozen Model 1 outputs; the selected-process subset is selected inside each training fold using inner-fold marginal MAE improvement.

| configuration | fold_count | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| customer_plus_inner_selected_process_RF | 5 | 4.7571 | 0.1183 | 5.9095 | 0.1504 | 0.0547 | 0.0241 |
| all_17_inference_inputs_RF | 5 | 4.7706 | 0.1110 | 5.9122 | 0.1300 | 0.0536 | 0.0307 |
| predicted_process_only_RF | 5 | 4.7801 | 0.1034 | 5.9283 | 0.1431 | 0.0488 | 0.0183 |
| customer_inputs_only_RF | 5 | 4.7848 | 0.1231 | 5.9570 | 0.1461 | 0.0394 | 0.0262 |
| training_fold_median | 5 | 4.8894 | 0.1564 | 6.1159 | 0.2094 | -0.0118 | 0.0096 |
| training_fold_mean | 5 | 4.9063 | 0.1301 | 6.0922 | 0.1800 | -0.0042 | 0.0051 |

### Held-out result and residuals

Training-only inner CV selected **Ridge** with parameters `{"alpha": 50.0}`. On the saved 1,000-row holdout, MAE=4.9357, RMSE=6.3981, R²=0.0986. Residual convention is prediction − actual: mean=0.1096, median=-0.3014, SD=6.4003, 5th–95th percentile=[-8.9049, 10.3963] pp; absolute-error median/p90=4.3053/9.5164 pp.

Residual records and subgroup metrics are in `residual_diagnostics.csv` and `subgroup_error_analysis.csv`. Subgroup estimates with fewer than 20 rows are marked unreliable and should not be over-interpreted. Summary below shows the largest reliable subgroup MAEs per grouping for the selected model.

Held-out feature-group ablation (same saved test rows):

| configuration | feature_count | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| customer_inputs_only_RF | 8 | 5.0972 | 6.5795 | 0.0468 |
| predicted_process_only_RF | 9 | 5.0558 | 6.5226 | 0.0632 |
| all_17_inference_inputs_RF | 17 | 5.0460 | 6.5067 | 0.0678 |
| customer_plus_inner_selected_process_RF | 16 | 5.0260 | 6.4820 | 0.0748 |
| training_fold_mean | 0 | 5.2362 | 6.7407 | -0.0005 |
| training_fold_median | 0 | 5.2391 | 6.7752 | -0.0107 |

| model | grouping | group | count | MAE | RMSE | mean_residual | reliable_at_minimum_n | minimum_n_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | Braid_Type | None | 146 | 5.8948 | 8.8054 | 0.7197 | True | 20 |
| Ridge | Braid_Type | Helical | 200 | 4.9506 | 6.0978 | -0.1860 | True | 20 |
| Ridge | Braid_Type | 1-Wire | 211 | 4.8181 | 5.8845 | -0.0021 | True | 20 |
| Ridge | Braid_Type | Textile | 281 | 4.7537 | 5.9091 | -0.0218 | True | 20 |
| Ridge | Braid_Type | 2-Wire | 162 | 4.5221 | 5.5889 | 0.2979 | True | 20 |
| Ridge | Fault_Code | Cooling Failure | 30 | 8.5844 | 9.6032 | 8.3620 | True | 20 |
| Ridge | Fault_Code | Dimension Variation | 67 | 6.8592 | 7.9685 | 6.7653 | True | 20 |
| Ridge | Fault_Code | Material Scorch | 27 | 5.2935 | 6.2837 | 4.9366 | True | 20 |
| Ridge | Fault_Code | Carrier Problem | 53 | 4.6814 | 5.7102 | 4.0279 | True | 20 |
| Ridge | Fault_Code | High Throughput Stress | 23 | 4.6485 | 6.0993 | 3.8499 | True | 20 |
| Ridge | Material_Type | PVC | 125 | 5.1908 | 7.8309 | 0.3325 | True | 20 |
| Ridge | Material_Type | PE | 219 | 5.0208 | 6.6856 | -0.3795 | True | 20 |
| Ridge | Material_Type | TPU | 196 | 4.9912 | 6.0356 | 0.6771 | True | 20 |
| Ridge | Material_Type | PA12 | 360 | 4.8588 | 6.0323 | 0.2118 | True | 20 |
| Ridge | Material_Type | PP | 100 | 4.5988 | 5.6934 | -0.5785 | True | 20 |
| Ridge | Tube_Type | Hydraulic | 194 | 5.1536 | 6.3484 | 0.2644 | True | 20 |
| Ridge | Tube_Type | Industrial | 185 | 5.0038 | 6.7081 | 0.5428 | True | 20 |
| Ridge | Tube_Type | PTFE | 192 | 4.8892 | 7.0614 | 0.3266 | True | 20 |
| Ridge | Tube_Type | Composite | 184 | 4.8562 | 5.8492 | -0.9955 | True | 20 |
| Ridge | Tube_Type | Pneumatic | 245 | 4.8081 | 6.0350 | 0.3196 | True | 20 |

The held-out Machine Breakdown subgroup has only 16 records; its Ridge MAE is 17.96 pp and it is explicitly below the minimum-n threshold, so this is a warning signal rather than a stable subgroup estimate.

## Importance and interpretation

Top transformed-feature importances for the selected training-only model:

| feature | importance | method |
| --- | --- | --- |
| Die_Temperature_C | 0.5369 | absolute_standardized_coefficient |
| Screw_RPM | 0.4996 | absolute_standardized_coefficient |
| Extrusion_Speed_kg_hr | 0.4928 | absolute_standardized_coefficient |
| Tube_Type_PTFE | 0.3899 | absolute_standardized_coefficient |
| Puller_Speed_m_min | 0.2876 | absolute_standardized_coefficient |
| Braid_Type_Helical | 0.2704 | absolute_standardized_coefficient |
| Braid_Type_None | 0.2495 | absolute_standardized_coefficient |
| Material_Grade_G3 | 0.2485 | absolute_standardized_coefficient |
| Tube_Type_Industrial | 0.2261 | absolute_standardized_coefficient |
| Line_Speed_m_min | 0.2247 | absolute_standardized_coefficient |
| Extruder_Temperature_C | 0.2165 | absolute_standardized_coefficient |
| Target_Tube_Wall_mm | 0.2139 | absolute_standardized_coefficient |
| Material_Grade_G1 | 0.1986 | absolute_standardized_coefficient |
| Material_Type_PA12 | 0.1957 | absolute_standardized_coefficient |
| Material_Type_PVC | 0.1857 | absolute_standardized_coefficient |

Importance is a model-specific predictive diagnostic—not a causal effect. `customer_input_sensitivity.csv` gives perturbation contrasts on test inputs, with tube geometry kept valid and categorical alternatives chosen from training levels. It is not a causal experiment.

## Findings and recommendation

- The fixed dataset hash and all 162 protected-file hashes match the pre-audit preservation manifest: **True**.
- The evaluated Model 2 relationship remains limited; MAE/RMSE and R² should be interpreted with fold variability and subgroup counts rather than R² alone.
- Numeric pairs with absolute correlation ≥0.9999 are catalogued in `near_duplicate_numeric_features.csv`; target-related pairs reflect accounting/target formulas and are not accepted as model inputs.
- The experiment uses a different upstream feature-generation population than the original revised-experiment runner: this audit freezes production Model 1 inference artifacts. Do not compare these numbers to earlier revised-experiment metrics as paired outcomes.
- All nine modeled process columns are overwritten with frozen Model 1 predictions before splitting; `process_input_source_validation.csv` reports their differences from recorded post-run values.
- Machine context and realized faults/downtime are not part of the deployed 17-feature contract, despite contributing to target variation. This supports residual uncertainty and fault-specific error.
- The generator's speed, material/process profiles, environmental variation, fault probabilities/severity, and nominal-rate assumptions remain synthetic until manufacturing engineering validates them. No Polyhose operating limits are inferred here.

**Recommendation: revise/validate generator assumptions and inference-time feature availability with manufacturing engineering before candidate training.** Keep production generator, dataset, models, and app unchanged; do not promote a candidate based on synthetic-only results.

## Files

See `README.md` for the rerun sequence. Machine-readable outputs include the dataset accounting validation, feature role/leakage audit, target/relationship diagnostics, ablation and nested-CV metrics, residual/subgroup outputs, feature importance, preservation verification, and diagnostic manifest.
