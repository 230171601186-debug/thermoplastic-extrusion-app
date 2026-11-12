# Model 2 Generator Experiment

## Scope and Preservation

Baseline dataset SHA-256: `cbd42684bc475ca9a7ad4f5b4512962c785034f2aae7c4944f0804e624c20396`. Experimental dataset SHA-256: `37887ca7f2cbb3a1e48c8103c4d371fab1fe54c5c106fa91c7dea816a5c6055c`. Production dataset, model artifacts, metadata, generator, and inference code were not written.

The production baseline metrics are saved results, not retrained. Existing nested-stack summaries are taken from the controlled evaluation. The experiment uses the same 80/20 split configuration and saved row-index partition on a different generated dataset; scores across dataset versions are not directly comparable as paired observations.

## Generator Change

Performance is now modeled as realized line output rate divided by a synthetic nominal recipe speed during actual runtime. Runtime is therefore not embedded in the Performance denominator and then counted again by Availability. Efficiency remains derived only from OEE components. Faults and downtime remain risk-dependent and stochastic.

The nominal-speed formula, inherited material profiles, fault tables, and bounds are synthetic engineering-inspired assumptions, not factory-validated settings.

## Experimental Dataset Checks

Shape: [5000, 40]; missing cells: 0; duplicate IDs: 0; faulted rows: 1599; `Braid_Type='None'`: 776; geometry max residual: 0.0100 mm.

The inherited status rule uses total actual processing length versus planned processing length, not good output versus customer-required length. 279 of 4786 baseline completed records and 289 of 4818 experimental completed records have good output below customer-required length. No project documentation clarifies whether this is an intended fulfillment policy; treat it as an unresolved synthetic accounting assumption.

Accounting residuals and target summaries are in `experimental_dataset_audit.json`. All schema, category, geometry, bounds, and accounting assertions ran before model evaluation.

## Nested CV Results

| dataset_version | split | configuration | fold_count | MAE_mean | RMSE_mean | R2_mean | MAE_std | RMSE_std | R2_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiment | nested_outer_cv | training_mean | 5 | 4.998880977296874 | 6.151071578976141 | -0.0027683561254817136 | 0.09573445427329089 | 0.18762024836904231 | 0.0032950867319890198 |
| experiment | nested_outer_cv | customer_only_fixed_RF | 5 | 4.748076367699213 | 5.895921653228595 | 0.07848913634216312 | 0.0988829570825295 | 0.1740912053467596 | 0.0193095652912959 |
| experiment | nested_outer_cv | process_only_fixed_RF | 5 | 4.7712002948541805 | 5.914360634888222 | 0.07263353302049842 | 0.11937655701673841 | 0.21941226307994907 | 0.037387869942557515 |
| experiment | nested_outer_cv | full_stack_fixed_RF | 5 | 4.7190364654650825 | 5.838058965750032 | 0.09642502291190759 | 0.12837777835996714 | 0.21691581192731604 | 0.03595898214212813 |
| experiment | nested_outer_cv | inner_selected_full_stack | 5 | 4.704703640594901 | 5.8235459825651175 | 0.10073866279775848 | 0.1457848730225997 | 0.22874143437799124 | 0.04380546363784065 |

## Held-Out Test Results

| dataset_version | split | configuration | fold_count | MAE_mean | RMSE_mean | R2_mean | MAE_std | RMSE_std | R2_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiment | saved_test | training_mean | 1 | 4.8706306034 | 6.085815578181987 | -1.671999912278288e-05 |  |  |  |
| experiment | saved_test | customer_only_fixed_RF | 1 | 4.599935384653588 | 5.891780257272782 | 0.0627342052656739 |  |  |  |
| experiment | saved_test | process_only_fixed_RF | 1 | 4.527214291313244 | 5.8335373307844085 | 0.0811732105785416 |  |  |  |
| experiment | saved_test | full_stack_fixed_RF | 1 | 4.476626825665119 | 5.737757213338994 | 0.1110977172900743 |  |  |  |
| experiment | saved_test | inner_selected_full_stack | 1 | 4.445780326257324 | 5.701826321520456 | 0.1221957984611359 |  |  |  |

Baseline current-dataset nested CV results (separate dataset):

| split | target | configuration | fold_count | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std | MAE_percentage_points_mean | MAE_percentage_points_std | RMSE_percentage_points_mean | RMSE_percentage_points_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outer_cv | Operational_Efficiency_pct | training_mean | 5 | 7.159336612999999 | 0.1554400230156836 | 9.229311027870104 | 0.3179489084423157 | -0.0017108751162346 | 0.0008869608358077 |  |  |  |  |
| outer_cv | Operational_Efficiency_pct | customer_only | 5 | 6.4756403637299 | 0.1936122034308021 | 8.475575242200753 | 0.3422416678476868 | 0.1548979925662899 | 0.0367867007000559 |  |  |  |  |
| outer_cv | Operational_Efficiency_pct | process_only | 5 | 6.471657468856286 | 0.1841716601815232 | 8.482522225813645 | 0.3808257636213351 | 0.1540407074789356 | 0.0273470117021512 |  |  |  |  |
| outer_cv | Operational_Efficiency_pct | full_stack | 5 | 6.423436152427622 | 0.1893850914791188 | 8.38085335761167 | 0.3490383151335234 | 0.173877722282646 | 0.0321387132103704 |  |  |  |  |

Baseline existing fixed-RF held-out comparisons (separate dataset):

| split | model | MAE_percentage_points | RMSE_percentage_points | R2 | prediction_below_zero_count | prediction_above_100_count | prediction_min | prediction_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| saved_test_1000_rows | training_mean | 7.386826146000001 | 9.629305682399265 | -0.0001751300994923 | 0 | 0 | 73.35056942499999 | 73.35056942499999 |
| saved_test_1000_rows | customer_only_fixed_RF | 6.492930201066033 | 8.549346548931712 | 0.2115903208076837 | 0 | 0 | 56.11404313492066 | 83.93695553283771 |
| saved_test_1000_rows | process_only_fixed_RF | 6.478890636231596 | 8.676238958854398 | 0.1880129299877447 | 0 | 0 | 56.245117296874966 | 83.22161881496295 |
| saved_test_1000_rows | current_production_artifacts | 6.406045982181854 | 8.461828800497628 | 0.2276492491969358 | 0 | 0 | 51.80417127341271 | 83.5577418521909 |
| saved_test_1000_rows | corrected_candidate_full_stack_RF | 6.369941258599522 | 8.448139479917222 | 0.2301462046314215 | 0 | 0 | 53.29934531324408 | 82.21847911785281 |

The inner-selected alternative was chosen using training-only inner folds and was not selected by test results. Group generalization by material, grade, tube, braid, machine, shift, and fault is in `test_generalization_by_group.csv`; residual summaries and feature importances are separate CSVs.

## Residual and Feature-Importance Diagnostics

Across the full-stack fixed-RF outer-fold predictions, residual bias by training-derived target quartile is range_1: +7.30 pp, range_2: +0.72 pp, range_3: -3.32 pp, range_4: -5.24 pp. The inner-selected held-out residuals have mean -0.07 pp; 5th/95th percentiles -6.81/+10.53 pp. Machine Breakdown remains a difficult small test subgroup (n=12, MAE=19.80 percentage points).

The selected full-training XGBoost feature importances are led by Screw_RPM (0.128), Tube_Type_PTFE (0.062), Extrusion_Speed_kg_hr (0.061), Customer_Required_Length_m (0.041), Cooling_Tank_Temperature_C (0.040). These are model-specific predictive importance scores, not causal effects.

## Interpretation and Limitations

The fixed full-stack RF experimental nested CV is 0.096 R²; the training-only selected alternative is XGBoost. Do not interpret a cross-dataset score difference as measured improvement on the same population. Review MAE/RMSE, fold variance, fault-group errors, residuals, and model importances alongside R².

No R² target was imposed. Synthetic data does not validate real Polyhose operations. In the experimental target, the median and 75th percentile of Performance are both 1.0, showing substantial ceiling saturation; this is a modeling assumption to validate rather than evidence of actual process performance. The performance definition and profile constants require manufacturing-engineer review before any generator change could be considered for production.

## Recommendation

Revise the generator assumptions further before any candidate training. The experiment removes a potential availability/performance double count but the nominal-speed reference remains unvalidated, Performance is heavily ceiling-saturated, and order status does not consistently reflect good output against customer requirements. Retain the current production generator and artifacts unchanged until engineering review resolves these assumptions; do not promote this dataset or train production artifacts automatically.
