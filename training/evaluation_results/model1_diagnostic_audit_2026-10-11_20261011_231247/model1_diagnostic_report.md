# Controlled Model 1 Diagnostic Audit

## Scope, data and preservation

This isolated analysis uses the existing revised synthetic dataset (5,000 rows × 40 columns; SHA-256 `0db6c6d33f0a3d580b2f673cc3761b4c2551f48c1ca68d296303d9854a669b15`), fixed saved 4,000/1,000 row-index split and seed 42. It creates no dataset and does not modify the production dataset, generator, models, metadata, inference bundle, or Streamlit app. Pre-analysis and post-analysis preservation SHA-256 checks cover production data/code, models/metadata, the app and all existing evaluation experiment files; results are in `preservation_manifest.json` and `preservation_verification.json`.

Model 1's inference contract is the metadata/config-defined eight fields: categorical material, grade, tube type, braid type and numeric OD, wall, ID and required length. All are present in this dataset and none is a process target. Machine_ID, shift, ambient/humidity, planned processing/time, fault and other post-run columns are excluded at inference. The target fields evaluated are the nine process outputs from `training/config.py`.

## Saved Model 1 versus baselines

The current artifacts were trained against the production dataset, not this revised experimental dataset. Their revised-dataset evaluation therefore measures transfer to a different synthetic target population; it is not a reproduction of the original production test scores. The original metadata values are recorded alongside the revised-set results in `saved_metric_context.csv`, without treating them as paired.

| target | train_variance | train_min | train_max | saved_artifact_test_MAE | saved_artifact_test_RMSE | saved_artifact_test_R2 | selected_test_MAE | selected_test_RMSE | selected_test_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extruder_Temperature_C | 63.4647 | 199.5800 | 245.0000 | 3.7128 | 4.6968 | 0.6292 | 3.5430 | 4.5092 | 0.6582 |
| Die_Temperature_C | 75.5870 | 190.2900 | 235.0000 | 4.8043 | 5.9395 | 0.5124 | 4.6426 | 5.7614 | 0.5412 |
| Cooling_Tank_Temperature_C | 2.4520 | 15.0400 | 28.0000 | 1.0321 | 1.4326 | 0.1578 | 0.8904 | 1.2977 | 0.3089 |
| Melt_Pressure_bar | 73.0810 | 101.8000 | 161.6000 | 4.6811 | 5.8847 | 0.5117 | 4.5195 | 5.6889 | 0.5437 |
| Air_Pressure_bar | 0.9158 | 5.0100 | 11.3000 | 0.5354 | 0.6773 | 0.5023 | 0.5260 | 0.6597 | 0.5280 |
| Screw_RPM | 50.3580 | 51.3400 | 100.0000 | 3.4801 | 4.3828 | 0.6175 | 3.3822 | 4.2729 | 0.6365 |
| Line_Speed_m_min | 3.0830 | 8.8400 | 18.5100 | 1.8822 | 2.0766 | -0.4084 | 0.6223 | 0.7639 | 0.8094 |
| Puller_Speed_m_min | 3.6390 | 7.0000 | 19.0900 | 1.9006 | 2.1693 | -0.3509 | 0.8026 | 1.0220 | 0.7001 |
| Extrusion_Speed_kg_hr | 143.6860 | 64.2600 | 146.7600 | 5.5996 | 7.0506 | 0.6368 | 5.4389 | 6.8272 | 0.6594 |

Holdout comparison against training-only constant baselines and the inner-selected candidate:

| target | comparison | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| Extruder_Temperature_C | training_mean | 6.2765 | 7.7143 | -0.0004 |
| Extruder_Temperature_C | training_median | 6.2796 | 7.7150 | -0.0006 |
| Extruder_Temperature_C | Saved production Model1 artifact | 3.7128 | 4.6968 | 0.6292 |
| Extruder_Temperature_C | Training_selected | 3.5430 | 4.5092 | 0.6582 |
| Die_Temperature_C | training_mean | 6.8720 | 8.5074 | -0.0004 |
| Die_Temperature_C | training_median | 6.8638 | 8.5058 | -0.0000 |
| Die_Temperature_C | Saved production Model1 artifact | 4.8043 | 5.9395 | 0.5124 |
| Die_Temperature_C | Training_selected | 4.6426 | 5.7614 | 0.5412 |
| Cooling_Tank_Temperature_C | training_mean | 1.1068 | 1.5613 | -0.0003 |
| Cooling_Tank_Temperature_C | training_median | 1.0981 | 1.5667 | -0.0072 |
| Cooling_Tank_Temperature_C | Saved production Model1 artifact | 1.0321 | 1.4326 | 0.1578 |
| Cooling_Tank_Temperature_C | Training_selected | 0.8904 | 1.2977 | 0.3089 |
| Melt_Pressure_bar | training_mean | 6.7228 | 8.4225 | -0.0002 |
| Melt_Pressure_bar | training_median | 6.7257 | 8.4250 | -0.0008 |
| Melt_Pressure_bar | Saved production Model1 artifact | 4.6811 | 5.8847 | 0.5117 |
| Melt_Pressure_bar | Training_selected | 4.5195 | 5.6889 | 0.5437 |
| Air_Pressure_bar | training_mean | 0.7825 | 0.9609 | -0.0016 |
| Air_Pressure_bar | training_median | 0.7826 | 0.9611 | -0.0019 |
| Air_Pressure_bar | Saved production Model1 artifact | 0.5354 | 0.6773 | 0.5023 |
| Air_Pressure_bar | Training_selected | 0.5260 | 0.6597 | 0.5280 |
| Screw_RPM | training_mean | 5.7342 | 7.0917 | -0.0013 |
| Screw_RPM | training_median | 5.7371 | 7.0987 | -0.0033 |
| Screw_RPM | Saved production Model1 artifact | 3.4801 | 4.3828 | 0.6175 |
| Screw_RPM | Training_selected | 3.3822 | 4.2729 | 0.6365 |
| Line_Speed_m_min | training_mean | 1.4542 | 1.7499 | -0.0001 |
| Line_Speed_m_min | training_median | 1.4490 | 1.7601 | -0.0119 |
| Line_Speed_m_min | Saved production Model1 artifact | 1.8822 | 2.0766 | -0.4084 |
| Line_Speed_m_min | Training_selected | 0.6223 | 0.7639 | 0.8094 |
| Puller_Speed_m_min | training_mean | 1.5225 | 1.8673 | -0.0010 |
| Puller_Speed_m_min | training_median | 1.5222 | 1.8745 | -0.0087 |
| Puller_Speed_m_min | Saved production Model1 artifact | 1.9006 | 2.1693 | -0.3509 |
| Puller_Speed_m_min | Training_selected | 0.8026 | 1.0220 | 0.7001 |
| Extrusion_Speed_kg_hr | training_mean | 9.5073 | 11.7025 | -0.0007 |
| Extrusion_Speed_kg_hr | training_median | 9.5049 | 11.7003 | -0.0003 |
| Extrusion_Speed_kg_hr | Saved production Model1 artifact | 5.5996 | 7.0506 | 0.6368 |
| Extrusion_Speed_kg_hr | Training_selected | 5.4389 | 6.8272 | 0.6594 |

Candidate-family results use 5 fixed outer folds in the 4,000-row training split. Family/config selection occurs inside each outer training portion using three inner folds. The held-out 1,000 rows do not influence fitting or selection.

| target | evaluation_arm | fold_count | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Air_Pressure_bar | Ridge | 5 | 0.5150 | 0.0049 | 0.6449 | 0.0078 | 0.5453 | 0.0113 |
| Air_Pressure_bar | inner_selected | 5 | 0.5150 | 0.0049 | 0.6449 | 0.0078 | 0.5453 | 0.0113 |
| Air_Pressure_bar | Saved_Model1 | 5 | 0.5238 | 0.0086 | 0.6593 | 0.0067 | 0.5245 | 0.0191 |
| Air_Pressure_bar | HistGradientBoosting | 5 | 0.5243 | 0.0055 | 0.6569 | 0.0086 | 0.5283 | 0.0069 |
| Air_Pressure_bar | RandomForest | 5 | 0.5407 | 0.0076 | 0.6782 | 0.0102 | 0.4972 | 0.0076 |
| Air_Pressure_bar | ExtraTrees | 5 | 0.5468 | 0.0040 | 0.6850 | 0.0086 | 0.4871 | 0.0046 |
| Cooling_Tank_Temperature_C | Ridge | 5 | 0.8685 | 0.0090 | 1.2619 | 0.0177 | 0.3494 | 0.0132 |
| Cooling_Tank_Temperature_C | inner_selected | 5 | 0.8685 | 0.0090 | 1.2619 | 0.0177 | 0.3494 | 0.0132 |
| Cooling_Tank_Temperature_C | HistGradientBoosting | 5 | 0.8919 | 0.0069 | 1.2849 | 0.0214 | 0.3255 | 0.0129 |
| Cooling_Tank_Temperature_C | RandomForest | 5 | 0.9371 | 0.0073 | 1.3423 | 0.0227 | 0.2638 | 0.0200 |
| Cooling_Tank_Temperature_C | ExtraTrees | 5 | 0.9568 | 0.0163 | 1.3591 | 0.0327 | 0.2453 | 0.0285 |
| Cooling_Tank_Temperature_C | Saved_Model1 | 5 | 1.0051 | 0.0246 | 1.3762 | 0.0305 | 0.2259 | 0.0325 |
| Die_Temperature_C | Ridge | 5 | 4.6643 | 0.0872 | 5.7866 | 0.0969 | 0.5563 | 0.0107 |
| Die_Temperature_C | inner_selected | 5 | 4.6643 | 0.0872 | 5.7866 | 0.0969 | 0.5563 | 0.0107 |
| Die_Temperature_C | HistGradientBoosting | 5 | 4.7276 | 0.0724 | 5.8614 | 0.0943 | 0.5448 | 0.0060 |
| Die_Temperature_C | Saved_Model1 | 5 | 4.7596 | 0.0609 | 5.9224 | 0.0871 | 0.5351 | 0.0161 |
| Die_Temperature_C | RandomForest | 5 | 4.9459 | 0.0798 | 6.1405 | 0.0641 | 0.5002 | 0.0148 |
| Die_Temperature_C | ExtraTrees | 5 | 5.0631 | 0.1580 | 6.2781 | 0.1737 | 0.4779 | 0.0134 |
| Extruder_Temperature_C | Ridge | 5 | 3.6206 | 0.1477 | 4.6166 | 0.1894 | 0.6633 | 0.0218 |
| Extruder_Temperature_C | inner_selected | 5 | 3.6206 | 0.1477 | 4.6166 | 0.1894 | 0.6633 | 0.0218 |
| Extruder_Temperature_C | HistGradientBoosting | 5 | 3.6731 | 0.1450 | 4.6853 | 0.1811 | 0.6532 | 0.0204 |
| Extruder_Temperature_C | Saved_Model1 | 5 | 3.7146 | 0.1342 | 4.7305 | 0.1640 | 0.6466 | 0.0171 |
| Extruder_Temperature_C | RandomForest | 5 | 3.8541 | 0.1150 | 4.9081 | 0.1600 | 0.6195 | 0.0189 |
| Extruder_Temperature_C | ExtraTrees | 5 | 3.9530 | 0.1623 | 5.0432 | 0.2110 | 0.5982 | 0.0249 |
| Extrusion_Speed_kg_hr | Ridge | 5 | 5.4257 | 0.1331 | 6.8033 | 0.1624 | 0.6776 | 0.0084 |
| Extrusion_Speed_kg_hr | inner_selected | 5 | 5.4257 | 0.1331 | 6.8033 | 0.1624 | 0.6776 | 0.0084 |
| Extrusion_Speed_kg_hr | HistGradientBoosting | 5 | 5.6518 | 0.1315 | 7.0535 | 0.1519 | 0.6534 | 0.0080 |
| Extrusion_Speed_kg_hr | Saved_Model1 | 5 | 5.6770 | 0.1194 | 7.1170 | 0.1457 | 0.6472 | 0.0058 |
| Extrusion_Speed_kg_hr | ExtraTrees | 5 | 5.8746 | 0.1008 | 7.3179 | 0.1742 | 0.6269 | 0.0116 |
| Extrusion_Speed_kg_hr | RandomForest | 5 | 5.9615 | 0.1315 | 7.4495 | 0.1925 | 0.6133 | 0.0167 |
| Line_Speed_m_min | HistGradientBoosting | 5 | 0.5929 | 0.0152 | 0.7360 | 0.0205 | 0.8239 | 0.0109 |
| Line_Speed_m_min | inner_selected | 5 | 0.5929 | 0.0152 | 0.7360 | 0.0205 | 0.8239 | 0.0109 |
| Line_Speed_m_min | RandomForest | 5 | 0.6174 | 0.0137 | 0.7688 | 0.0201 | 0.8078 | 0.0122 |
| Line_Speed_m_min | Ridge | 5 | 0.6187 | 0.0237 | 0.7640 | 0.0232 | 0.8100 | 0.0147 |
| Line_Speed_m_min | ExtraTrees | 5 | 0.6226 | 0.0132 | 0.7751 | 0.0202 | 0.8046 | 0.0124 |
| Line_Speed_m_min | Saved_Model1 | 5 | 1.8836 | 0.0342 | 2.0761 | 0.0301 | -0.4015 | 0.0675 |
| Melt_Pressure_bar | Ridge | 5 | 4.4723 | 0.0590 | 5.6599 | 0.0493 | 0.5608 | 0.0156 |
| Melt_Pressure_bar | inner_selected | 5 | 4.4723 | 0.0590 | 5.6599 | 0.0493 | 0.5608 | 0.0156 |
| Melt_Pressure_bar | HistGradientBoosting | 5 | 4.5964 | 0.0736 | 5.7887 | 0.0726 | 0.5404 | 0.0234 |
| Melt_Pressure_bar | Saved_Model1 | 5 | 4.6509 | 0.0576 | 5.8635 | 0.0685 | 0.5287 | 0.0163 |
| Melt_Pressure_bar | RandomForest | 5 | 4.8136 | 0.1097 | 6.0534 | 0.1039 | 0.4973 | 0.0306 |
| Melt_Pressure_bar | ExtraTrees | 5 | 4.8646 | 0.1366 | 6.0938 | 0.0906 | 0.4907 | 0.0260 |
| Puller_Speed_m_min | inner_selected | 5 | 0.8105 | 0.0208 | 1.0334 | 0.0264 | 0.7059 | 0.0152 |
| Puller_Speed_m_min | HistGradientBoosting | 5 | 0.8105 | 0.0208 | 1.0311 | 0.0270 | 0.7071 | 0.0165 |
| Puller_Speed_m_min | Ridge | 5 | 0.8119 | 0.0190 | 1.0411 | 0.0214 | 0.7016 | 0.0117 |
| Puller_Speed_m_min | RandomForest | 5 | 0.8429 | 0.0255 | 1.0683 | 0.0328 | 0.6855 | 0.0220 |
| Puller_Speed_m_min | ExtraTrees | 5 | 0.8476 | 0.0216 | 1.0759 | 0.0314 | 0.6811 | 0.0201 |
| Puller_Speed_m_min | Saved_Model1 | 5 | 1.9127 | 0.0445 | 2.1857 | 0.0424 | -0.3147 | 0.0235 |
| Screw_RPM | Ridge | 5 | 3.3655 | 0.0285 | 4.2362 | 0.0462 | 0.6430 | 0.0136 |
| Screw_RPM | inner_selected | 5 | 3.3655 | 0.0285 | 4.2362 | 0.0462 | 0.6430 | 0.0136 |
| Screw_RPM | HistGradientBoosting | 5 | 3.4918 | 0.0725 | 4.4057 | 0.0803 | 0.6139 | 0.0166 |
| Screw_RPM | Saved_Model1 | 5 | 3.4940 | 0.0485 | 4.3994 | 0.0580 | 0.6151 | 0.0113 |
| Screw_RPM | ExtraTrees | 5 | 3.6680 | 0.0592 | 4.6074 | 0.0588 | 0.5779 | 0.0099 |
| Screw_RPM | RandomForest | 5 | 3.6867 | 0.1170 | 4.6401 | 0.0974 | 0.5717 | 0.0182 |

`residual_diagnostics.csv` contains held-out prediction/residual rows; baseline comparison records include residual bias, SD, quantiles and absolute-error quantiles.

## Why the frozen speed predictions are weak here

On the revised synthetic holdout, saved-model metrics:

| target | train_variance | train_min | train_max | saved_artifact_test_MAE | saved_artifact_test_RMSE | saved_artifact_test_R2 | selected_test_MAE | selected_test_RMSE | selected_test_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Line_Speed_m_min | 3.0830 | 8.8400 | 18.5100 | 1.8822 | 2.0766 | -0.4084 | 0.6223 | 0.7639 | 0.8094 |
| Puller_Speed_m_min | 3.6390 | 7.0000 | 19.0900 | 1.9006 | 2.1693 | -0.3509 | 0.8026 | 1.0220 | 0.7001 |

The two PNGs show actual-versus-predicted, residual-versus-actual, residual-versus-predicted and decile-mean collapse checks. Decile-level predictions and errors are also in `Line_Speed_m_min_decile_diagnostics.csv` and `Puller_Speed_m_min_decile_diagnostics.csv`.

The revised generator source includes synthetic assumptions, not manufacturing facts: nominal line speed is a function of material profile, grade, geometry, product difficulty and required length; realized speed adds a stochastic rate factor and machine-specific speed factor. Machine identity and that stochastic factor are not Model 1 inputs. Puller speed is generated from line speed plus braid offset/noise, with selected fault perturbations. However, the training-selected models achieve held-out R² 0.809 for line speed and 0.700 for puller speed, while the frozen production artifacts score −0.408 and −0.351 on this different synthetic dataset. The evidence therefore points to a substantial model/data-population mismatch in the frozen artifacts, not proof that the available customer inputs are intrinsically insufficient. Residual stochastic effects may remain; neither result establishes a real manufacturing cause. Synthetic nominal-speed bounds/process ranges must not be read as Polyhose limits.

See `feature_relationship_diagnostics.csv` for customer-input/target associations, category means, speed-rate correlations and line/puller difference distribution. Correlations and subgroup means are descriptive and do not prove causal effects.

## Downstream Model 2 impact

Each arm uses the same outer folds and fixed production-parameter Random Forest Model 2 estimator, with the 8 customer inputs and 9 process-valued columns only:

- **A — saved Model 1:** frozen production Model 1 prediction bundle on revised-dataset inputs.
- **B — nested training-only Model 1:** within each outer fold, Model 1 family/configs are selected using outer-training rows only; Model 1 features for Model 2 training are cross-fitted within outer training; outer-validation/test features come from Model 1 fitted only on the corresponding outer-training/full-training partition.
- **C — oracle:** recorded process targets are given to Model 2 for both its training and evaluation rows. This is **not deployable**, is only a diagnostic upper-bound-style comparison, and does not represent the app inference contract.

No outer validation/test process target is used to fit or select B's upstream models. Results:

| arm | deployable | outer_fold_count | outer_MAE_mean | outer_MAE_std | outer_RMSE_mean | outer_RMSE_std | outer_R2_mean | outer_R2_std | saved_test_MAE | saved_test_RMSE | saved_test_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_saved_Model1_inference_predictions | True | 5 | 4.7690 | 0.1106 | 5.9096 | 0.1307 | 0.0545 | 0.0286 | 5.0470 | 6.5094 | 0.0670 |
| B_outer_train_only_selected_Model1_OOF | True | 5 | 4.7685 | 0.1447 | 5.9315 | 0.1826 | 0.0479 | 0.0266 | 4.9921 | 6.4642 | 0.0799 |
| C_oracle_recorded_process_non_deployable | False | 5 | 3.0671 | 0.0862 | 4.0594 | 0.1676 | 0.5540 | 0.0234 | 3.1808 | 4.6513 | 0.5236 |

The B train-to-inference distribution differs by construction: Model 2 training rows use OOF process predictions, while validation/test rows use upstream fits on the full available training portion, paralleling the intended stacked-learning setup. The fixed downstream estimator makes arm changes attributable to feature-source differences within this audit; it does not establish that any alternative Model 1 should replace production.

## Answers and recommendation

1. **Reliability:** on this revised synthetic population the frozen artifacts are comparatively predictive for extruder/die temperatures, melt/air pressure, screw RPM and extrusion rate (R² about 0.50–0.64); cooling-tank temperature is weaker (R² 0.158), while the two speed artifact scores are negative. The training-selected candidates perform materially better on the held-out speed targets (line 0.809, puller 0.700). These labels describe this synthetic transfer test only.
2. **Weak speed targets:** the revised synthetic generator contains machine/stochastic speed effects absent from the customer feature contract, but candidate held-out results show the recorded customer/product features retain strong predictive signal. The frozen artifacts' negative R² is consistent with transfer/model mismatch; the exact causal contribution of generator noise versus model form is not isolated. Plots and subgroup summaries are synthetic diagnostics, not evidence of a factory cause.
3. **Algorithm change:** the training-selected candidates improve held-out line and puller speed against the saved artifacts, and aggregate fixed-fold results show whether gains repeat across folds. The choices use only training data, but these candidate-family settings are deliberately small, fixed comparisons—not a broad tuning search or an external validation.
4. **Downstream impact:** Model 1 arm B does not materially improve Model 2 over A in outer CV (MAE means 4.7690 vs 4.7685; RMSE 5.9096 vs 5.9315; R² 0.0545 vs 0.0479). On the untouched test, B is slightly better (MAE 4.9921 vs 5.0470; RMSE 6.4642 vs 6.5094; R² 0.0799 vs 0.0670), but this small test-only difference is not consistent outer-fold evidence of a material gain. C's oracle improvement is non-deployable.
5. **Limitations:** the revised target is generated under explicit synthetic equations, noise, hidden machine/fault context and unverified ranges. Any apparent error floor could reflect missing inputs, generator noise or mismatch between production-trained Model 1 artifacts and revised data; this audit cannot attribute it to actual plant behavior.
6. **Real-data validation:** validate recipe/setpoint rules and operational limits with process engineering; obtain time-aligned material/grade/geometry, machine identity, recipe, actual setpoints, speeds, line-stop/fault, scrap and good-output measurements; quantify sensor and operator variation; use time- or machine-held-out validation and calibration checks.
7. **App status:** keep the app labeled and used as a synthetic-data demo. The evidence-based next step is a governed real-production data pilot and time-aligned validation, not model promotion or score tuning.

## Automated checks and unresolved limits

The audit requires a 5,000 × 40 revised dataset and expected SHA-256, checks the saved split is disjoint 4,000/1,000, confirms all nine saved Model 1 pipelines and metadata feature-contract parity, keeps oracle fields segregated, and verifies preservation hashes before reporting. The separate validation command checks output completeness, required fold/arm counts, finite residuals, plot signatures and hashes; all 11 automated output checks passed and all 200 protected-file hashes matched. Syntax checks passed for the runner and both verification scripts. No validation check failed. This remains a synthetic diagnostic, not real manufacturing evidence.
