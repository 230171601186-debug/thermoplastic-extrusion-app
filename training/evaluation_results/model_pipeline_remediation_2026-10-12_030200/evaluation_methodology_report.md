# Leakage-safe evaluation methodology and results

## Data, split, and fixed protocol

- Dataset: `dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv`, 5,000 rows × 40 columns, SHA-256 `cbd42684bc475ca9a7ad4f5b4512962c785034f2aae7c4944f0804e624c20396`. It is synthetic.
- Existing saved split: seed 42, 4,000 training rows / 1,000 held-out test rows. Five outer folds operate only inside the 4,000-row training partition.
- No estimator selection or hyperparameter search occurred in this experiment. Fixed Random Forests were used for an independent, reproducible comparison: Model 1 `n_estimators=60`, `max_depth=20`, `min_samples_leaf=2`; downstream `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2`; seed 42.
- For each outer fold, each Model 1 training feature is an OOF prediction produced by a pipeline whose encoder and estimator were fit without that row. Outer validation upstream features are predicted using pipelines fit only on that outer training partition. Model 2/3 mean, customer-only, and full-stack arms use identical rows and folds. The saved 1,000-row test partition was evaluated once after CV.
- The deployable feature set is the eight customer fields plus nine predicted process parameters. Recorded process outcomes and post-run columns are prohibited. The oracle/recorded process values are **not** used in this experiment.
- Metrics from this fixed experiment are not interchanged with historical candidate/production metrics, which can use other model families, parameters, and evaluation methods.

The automated recomputation validator passed all 14 checks: required feature exclusions, row/fold coverage, upstream OOF disjointness and one prediction per OOF row, explicit outer-validation and saved-test partition exclusion, scrap fraction/percentage-point reporting, process target coverage and synthetic assertion bounds, and independent recomputation of saved fold metrics and CV summary. Details: `validation_results.json`.

## Model 1 process outputs (synthetic)

Values below are outer-fold mean ± sample SD; the saved-test scores are separately calculated on the fixed 1,000 rows. Mean/median are training-only baselines within each fold. A target is called “predictable” here only relative to this generated dataset and this fixed model.

| Process target | Outer CV MAE | Outer CV RMSE | Outer CV R² | Mean-baseline MAE | Median-baseline MAE | Saved-test MAE / RMSE / R² |
|---|---:|---:|---:|---:|---:|---:|
| Extruder temperature (°C) | 3.8369 ± 0.1735 | 4.8224 ± 0.2352 | 0.6133 ± 0.0164 | 6.2509 | 6.2510 | 3.8289 / 4.8492 / 0.6060 |
| Die temperature (°C) | 4.7984 ± 0.0745 | 5.9619 ± 0.1248 | 0.5069 ± 0.0199 | 6.8664 | 6.8630 | 4.8053 / 6.0616 / 0.4981 |
| Cooling-tank temperature (°C) | 0.9627 ± 0.0290 | 1.3952 ± 0.0460 | 0.2301 ± 0.0283 | 1.1422 | 1.1347 | 0.8916 / 1.2674 / 0.2840 |
| Melt pressure (bar) | 4.8737 ± 0.0778 | 6.1536 ± 0.1536 | 0.4740 ± 0.0153 | 6.8066 | 6.8030 | 4.8492 / 6.1209 / 0.4894 |
| Air pressure (bar) | 0.5425 ± 0.0103 | 0.6839 ± 0.0132 | 0.4809 ± 0.0197 | 0.7692 | 0.7692 | 0.5481 / 0.6862 / 0.4432 |
| Screw speed (rpm) | 3.7825 ± 0.0820 | 4.7777 ± 0.0985 | 0.5544 ± 0.0226 | 5.7777 | 5.7749 | 3.6476 / 4.6235 / 0.5677 |
| Line speed (m/min) | 0.8186 ± 0.0195 | 1.0296 ± 0.0307 | 0.7854 ± 0.0128 | 1.8278 | 1.8277 | 0.8087 / 1.0313 / 0.7724 |
| Puller speed (m/min) | 0.9681 ± 0.0251 | 1.2310 ± 0.0345 | 0.7200 ± 0.0131 | 1.9079 | 1.9080 | 0.9602 / 1.2129 / 0.7110 |
| Extrusion rate (kg/hr) | 5.9527 ± 0.1537 | 7.4440 ± 0.1459 | 0.6075 ± 0.0137 | 9.6288 | 9.6304 | 5.7584 / 7.2939 / 0.6222 |

All nine beat the fold-specific mean and median MAE baseline in this fixed synthetic evaluation. Cooling-tank temperature is weakest by R² (0.23); line/puller speed are comparatively better than the concern that motivated this task, but this is the fixed 60-tree RF experiment, not a re-evaluation of the app's fitted candidates. No score establishes real process-setting suitability. The process CSV includes fold and saved-test rows, and verifies none of the process predictions breached the existing synthetic assertion ranges.

## Matched Model 2 and Model 3 comparisons

Outer-CV values are mean ± sample SD over the same five outer folds / 4,000 rows. Test results are **not** CV averages.

| Target | Feature arm | CV MAE | CV RMSE | CV R² |
|---|---|---:|---:|---:|
| Efficiency (pp) | Training-fold target mean | 7.1593 ± 0.1554 | 9.2293 ± 0.3179 | -0.0017 ± 0.0009 |
| Efficiency (pp) | Customer-only RF | 6.4799 ± 0.1879 | 8.4788 ± 0.3282 | 0.1541 ± 0.0382 |
| Efficiency (pp) | Full 17-feature cross-fitted stack RF | 6.4156 ± 0.2659 | 8.4137 ± 0.4325 | 0.1674 ± 0.0489 |
| Scrap (fraction) | Training-fold target mean | 0.027127 ± 0.000351 (2.7127 ± 0.0351 pp) | 0.032759 ± 0.000480 (3.2759 ± 0.0480 pp) | -0.0016 ± 0.0012 |
| Scrap (fraction) | Customer-only RF | 0.022537 ± 0.000483 (2.2537 ± 0.0483 pp) | 0.027777 ± 0.000511 (2.7777 ± 0.0511 pp) | 0.2792 ± 0.0355 |
| Scrap (fraction) | Full 17-feature cross-fitted stack RF | 0.021952 ± 0.000606 (2.1952 ± 0.0606 pp) | 0.026978 ± 0.000658 (2.6978 ± 0.0658 pp) | 0.3198 ± 0.0431 |

| Target | Feature arm | Saved-test MAE | Saved-test RMSE | Saved-test R² |
|---|---|---:|---:|---:|
| Efficiency (pp) | Training mean | 7.3868 | 9.6293 | -0.0002 |
| Efficiency (pp) | Customer-only RF | 6.4910 | 8.5492 | 0.2116 |
| Efficiency (pp) | Full stack RF | 6.4094 | 8.4333 | 0.2329 |
| Scrap (fraction) | Training mean | 0.027102 (2.7102 pp) | 0.032669 (3.2669 pp) | -0.0000 |
| Scrap (fraction) | Customer-only RF | 0.021346 (2.1346 pp) | 0.026497 (2.6497 pp) | 0.3421 |
| Scrap (fraction) | Full stack RF | 0.021102 (2.1102 pp) | 0.026523 (2.6523 pp) | 0.3408 |

### Interpretation

- **Model 2:** Adding the nine predicted process fields reduces average CV MAE by 0.0644 percentage point (about 1.0% relative to customer-only MAE) and RMSE by 0.0651 pp. The full stack beats customer-only MAE/RMSE in 2 of 5 folds, while R² improves in 3 of 5. On the saved test, MAE improves by 0.0815 pp and RMSE by 0.1159 pp. This is a small, mixed incremental gain—not evidence that Model 1 replacement alone will materially solve efficiency prediction.
- **Model 3:** Full-stack MAE/RMSE are lower than customer-only in each of the five folds, by about 0.0585 pp MAE and 0.0799 pp RMSE on average (both in percentage-point units). On saved test, MAE improves only 0.0243 pp, RMSE is 0.0026 pp worse, and R² decreases by 0.0013. Thus the apparent CV incremental gain is modest and does not dominate held-out differences.
- All full-stack error distributions remain broad. Outer-CV residual 5th–95th quantiles are -12.30 to +14.24 pp for efficiency, and -0.05196 to +0.03463 fraction for scrap. The efficiency residual-by-actual-band table shows systematic regression toward the center: the lowest actual quartile has positive mean residual around +8.7 to +10.0 pp per fold; the highest has negative mean residual around -7.7 to -8.4 pp. This is visible in the supplied plots.
- Reported scrap targets can be discussed in fraction or percentage points; fraction MAE × 100 equals percentage-point MAE. Percentage points are not relative percent error.

The evidence supports weak-to-moderate available-input signal in this synthetic target, with a small added contribution from process predictions. It does not isolate a causal reason, prove a particular target-generation flaw, or validate performance by material/tube subgroup. No group-held-out material/tube generalization claim was made.

## Residual and artifacts

- `downstream_residual_diagnostics.csv`: bias, median, SD, 5th/95th residual quantiles, median absolute error, and 90th-percentile absolute error by target/arm/split.
- `downstream_residual_by_actual_band.csv`: outer-fold full-stack residual/error summaries by target quartiles defined from each training fold.
- `model1_residual_diagnostics.csv`: Model 1 outer-fold residual bias, spread, quantiles, target support and absolute-error summaries by target/fold. The held-out Model 1 row-level predictions were not retained by the runner; saved-test scalar metrics are in `model1_process_metrics.csv`.
- `operational_efficiency_pct_outer_cv_residuals.png` and `scrap_rate_outer_cv_residuals.png`: outer-CV actual-versus-predicted and residual-versus-predicted plots.
- `downstream_predictions_residuals.csv`, `downstream_fold_metrics.csv`, and `downstream_cv_summary.csv`: auditable row-level predictions and matched metrics.
- `model1_process_metrics.csv` and `model1_outer_predictions.csv`: process metrics/predictions and mean/median baselines.

## Limitations and next evidence

This protocol checks row-excluded upstream fitting and training-only preprocessing but uses shuffled row folds. It does not represent temporal, machine, shift, product-family, material-lot, or site holdout. Scores are conditional on the synthetic target definitions. Candidate model selection and production models have different fixed parameters and selection paths; this run does not select or promote them. No feature-importance ranking is offered as a causal explanation.

**Disposition:** no production model change is justified by this study. The decisive next step for a defensible reliability claim is governed historical production data with well-defined efficiency/scrap accounting and timestamps plus machine/line identities, then temporal and machine/product-group holdout evaluation.
