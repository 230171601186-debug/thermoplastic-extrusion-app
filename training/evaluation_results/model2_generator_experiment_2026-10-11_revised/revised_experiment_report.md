# Revised Isolated Model 2 Generator Experiment

## Scope and preservation

This revision is isolated in a new sibling directory. The production dataset, production generator, model artifacts, metadata, inference code, and Streamlit app were not changed, and no candidate was promoted.

Revised dataset SHA-256: `0db6c6d33f0a3d580b2f673cc3761b4c2551f48c1ca68d296303d9854a669b15`. The 5,000-row, 40-column schema, existing categorical labels, and fixed seed (42) are preserved. Re-generation was exact.

The existing experiment directory is retained unchanged. The same saved row-index train/test partition and leakage-controlled nested-stack procedure were used for this generated dataset; identical row positions do not make version-1 and version-2 rows paired observations.

## Diagnosed and revised logic

The prior generator centered realized line speed at nominal recipe speed with modest additive noise, then calculated `Performance` using `bounded(actual_rate / nominal_rate, 0, 1)`. This made values above nominal collapse to exactly 1.0; a high-throughput fault could also add speed after process generation. Those were the sources of the ceiling mass.

The revised process now draws realized rate as a continuous beta-distributed fraction of the synthetic nominal recipe reference, with a small synthetic machine factor. High-throughput stress changes screw RPM but no longer injects a line-speed increase beyond the recipe reference. `Performance` is calculated as realized line rate / nominal recipe rate without target clipping; OEE and efficiency follow directly from availability, performance, and quality.

The nominal 12–22 m/min recipe-reference envelope, beta distribution parameters, machine effects, inherited material/process profiles, fault probabilities, severity distributions, and all process bounds are **synthetic assumptions requiring manufacturing validation**. They are not Polyhose settings or validated operating limits. The nominal envelope was chosen only to keep the generated rate within the inherited experiment's synthetic line-speed range.

Order completion now means `Good_Length_m >= Customer_Required_Length_m`. This is an explicit synthetic fulfillment definition; `Length_Shortfall_m` remains `max(Planned_Processing_Length_m - Actual_Processing_Length_m, 0)`, a separate process-planning measure. The revised data has 4303 completed records and 0 completed records below required good output.

## Dataset-level changes (descriptive, not paired)

| dataset | target | mean | std | min | q25 | median | q75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_experiment_v1 | Performance | 0.9759 | 0.0371 | 0.7183 | 0.9608 | 1.0000 | 1.0000 | 1.0000 |
| existing_experiment_v1 | Operational_Efficiency_pct | 86.3542 | 6.1381 | 48.0361 | 82.4351 | 87.5807 | 91.2387 | 96.7830 |
| revised_experiment_v2 | Performance | 0.8677 | 0.0489 | 0.7112 | 0.8348 | 0.8713 | 0.9051 | 0.9811 |
| revised_experiment_v2 | Operational_Efficiency_pct | 77.0924 | 6.2260 | 16.5455 | 73.1151 | 77.6173 | 81.6555 | 92.4184 |

Performance's prior median/q75/max were 1.0000/1.0000/1.0000; revised values are 0.8713/0.9051/0.9811. No revised record is at or above 1.0.

Efficiency's prior median/max were 87.58/96.78%; revised median/max are 77.62/92.42%. The target distribution shifted downward as rate performance is now below nominal by construction. This is an intentional generator behavior change, not evidence of improved/worsened real factory efficiency.

## Leakage-controlled Model 2 results

Five-fold outer CV is reported as mean ± fold standard deviation. Candidate algorithm selection remains inside training folds; the saved test partition is not used to select the model.

| dataset | split | configuration | fold_count | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std | cross_dataset_scores_are_paired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_experiment_v1 | nested_outer_cv | training_mean | 5 | 4.9989 | 0.0957 | 6.1511 | 0.1876 | -0.0028 | 0.0033 | False |
| existing_experiment_v1 | nested_outer_cv | customer_only_fixed_RF | 5 | 4.7481 | 0.0989 | 5.8959 | 0.1741 | 0.0785 | 0.0193 | False |
| existing_experiment_v1 | nested_outer_cv | process_only_fixed_RF | 5 | 4.7712 | 0.1194 | 5.9144 | 0.2194 | 0.0726 | 0.0374 | False |
| existing_experiment_v1 | nested_outer_cv | full_stack_fixed_RF | 5 | 4.7190 | 0.1284 | 5.8381 | 0.2169 | 0.0964 | 0.0360 | False |
| existing_experiment_v1 | nested_outer_cv | inner_selected_full_stack | 5 | 4.7047 | 0.1458 | 5.8235 | 0.2287 | 0.1007 | 0.0438 | False |
| existing_experiment_v1 | saved_test | training_mean | 1 | 4.8706 |  | 6.0858 |  | -0.0000 |  | False |
| existing_experiment_v1 | saved_test | customer_only_fixed_RF | 1 | 4.5999 |  | 5.8918 |  | 0.0627 |  | False |
| existing_experiment_v1 | saved_test | process_only_fixed_RF | 1 | 4.5272 |  | 5.8335 |  | 0.0812 |  | False |
| existing_experiment_v1 | saved_test | full_stack_fixed_RF | 1 | 4.4766 |  | 5.7378 |  | 0.1111 |  | False |
| existing_experiment_v1 | saved_test | inner_selected_full_stack | 1 | 4.4458 |  | 5.7018 |  | 0.1222 |  | False |
| revised_experiment_v2 | nested_outer_cv | training_mean | 5 | 4.9063 | 0.1301 | 6.0922 | 0.1800 | -0.0042 | 0.0051 | False |
| revised_experiment_v2 | nested_outer_cv | customer_only_fixed_RF | 5 | 4.7848 | 0.1231 | 5.9570 | 0.1461 | 0.0394 | 0.0262 | False |
| revised_experiment_v2 | nested_outer_cv | process_only_fixed_RF | 5 | 4.8530 | 0.1436 | 6.0346 | 0.1732 | 0.0144 | 0.0274 | False |
| revised_experiment_v2 | nested_outer_cv | full_stack_fixed_RF | 5 | 4.7999 | 0.1460 | 5.9581 | 0.1732 | 0.0392 | 0.0275 | False |
| revised_experiment_v2 | nested_outer_cv | inner_selected_full_stack | 5 | 4.7752 | 0.1524 | 5.9369 | 0.1793 | 0.0461 | 0.0260 | False |
| revised_experiment_v2 | saved_test | training_mean | 1 | 5.2362 |  | 6.7407 |  | -0.0005 |  | False |
| revised_experiment_v2 | saved_test | customer_only_fixed_RF | 1 | 5.0972 |  | 6.5795 |  | 0.0468 |  | False |
| revised_experiment_v2 | saved_test | process_only_fixed_RF | 1 | 5.1422 |  | 6.6019 |  | 0.0403 |  | False |
| revised_experiment_v2 | saved_test | full_stack_fixed_RF | 1 | 5.0769 |  | 6.5384 |  | 0.0587 |  | False |
| revised_experiment_v2 | saved_test | inner_selected_full_stack | 1 | 5.0384 |  | 6.5000 |  | 0.0697 |  | False |

Revised fixed-RF full-stack CV: MAE 4.7999 ± 0.1460, RMSE 5.9581 ± 0.1732, R² 0.0392 ± 0.0275.

Revised training-only selected full-stack CV: MAE 4.7752 ± 0.1524, RMSE 5.9369 ± 0.1793, R² 0.0461 ± 0.0260. Held-out test: MAE 5.0384, RMSE 6.5000, R² 0.0697. Test residual mean +0.216 pp; 5th/95th percentiles -9.316/+10.886 pp.

Descriptively, the selected-model outer-CV scores moved from v1 to v2 by MAE +0.0705, RMSE +0.1134, and R² -0.0546; held-out values moved by MAE +0.5926, RMSE +0.7981, and R² -0.0525. Numerically these model scores are worse in v2, but because target populations differ these deltas are not paired estimates of a model regression.

Version-1 and version-2 model metrics are measured on different generated target populations. Their numerical differences are descriptive only, not paired gains/losses or evidence of generalization to manufacturing.

### Revised full-stack RF outer-CV residuals by training-derived target quartile

| target_range_training_quantiles | count | residual_mean | residual_std | residual_median | MAE |
| --- | --- | --- | --- | --- | --- |
| range_1 | 999 | 7.3605 | 4.0579 | 7.0120 | 7.4336 |
| range_2 | 999 | 1.4030 | 2.5249 | 1.5556 | 2.3707 |
| range_3 | 998 | -2.3667 | 2.4060 | -2.2941 | 2.7069 |
| range_4 | 1004 | -6.6765 | 2.6585 | -6.5275 | 6.6769 |

### Highest-error revised held-out fault groups, selected full-stack model

| value | count | MAE | RMSE | mean_residual |
| --- | --- | --- | --- | --- |
| Machine Breakdown | 16 | 17.9477 | 22.2632 | 17.9477 |
| Cooling Failure | 30 | 8.6719 | 9.8120 | 8.4724 |
| Dimension Variation | 67 | 6.9844 | 8.0608 | 6.7781 |
| Material Scorch | 27 | 5.5300 | 6.5175 | 5.1577 |
| High Throughput Stress | 23 | 5.3648 | 6.7165 | 4.4403 |
| Pressure Instability | 18 | 5.0685 | 6.8236 | 4.2772 |
| Carrier Problem | 53 | 4.7572 | 5.9633 | 4.0526 |
| Power Cut | 9 | 4.7559 | 5.0660 | 0.2677 |

All material/tube/grade/braid/machine/shift/fault subgroup metrics are in `test_generalization_by_group.csv`; full prediction residuals and feature importances are also saved separately.

## Accounting and automated validation

The generator and independent validation both checked tube geometry; actual length = line speed × runtime; good length + scrap length = actual length; scrap length = actual length × scrap rate; shortfall definition; runtime = planned time − downtime; availability = runtime / planned time; performance = realized rate / reconstructed nominal rate; quality = good length / actual length; OEE = availability × performance × quality; efficiency = OEE × 100; and the good-output completion rule.

| identity | max_abs_residual |
| --- | --- |
| actual_length | 0.0050 |
| good_plus_scrap | 0.0000 |
| scrap_length | 0.0050 |
| good_length | 0.0000 |
| length_shortfall | 0.0000 |
| runtime | 0.0100 |
| availability | 0.0000 |
| quality | 0.0000 |
| oee | 0.0000 |
| operational_efficiency | 0.0000 |

Automated checks passed: fixed-seed regeneration, schema and categorical-label preservation, accounting, order completion (0 inconsistent completed records), production dataset SHA-256, 40 production/model artifact hashes, and 32 files in the prior experiment directory.

## Recommendation

The revised synthetic mechanics resolve the prior Performance ceiling and ambiguous completion flag at the data-generation level, while preserving accounting and schema. However, the new rate-loss distribution and nominal-speed assumptions are not factory-validated. Keep production unchanged and do not promote a candidate; obtain manufacturing review of the assumptions before considering a further generator or model-training phase.
