# Model 2 Oracle Comparison: Bottleneck Analysis

## Executive conclusion

The three-arm comparison is valid for this **revised synthetic dataset** and the specified fixed Random Forest Model 2. It shows a large, consistent gap between using recorded process outcomes (oracle C) and either deployable process-prediction path (A or B). Replacing the saved Model 1 bundle with the tested training-only-selected bundle (B) does not close that gap: B is effectively tied with A in average outer-fold MAE and is worse on average in RMSE and R².

Thus, process-measurement/prediction loss is important in this diagnostic, but **simply changing the tested Model 1 approach is not supported as the main practical fix**. The revised synthetic target also depends on generated downtime/fault and scrap/quality outcomes absent from the intended 17-feature inference contract. Model-family changes were already tested in the companion Model 2 audit and give only modest gains. Because the target mechanics and hidden-event distributions are synthetic assumptions, the single recommended next action is to **request suitable historical production data for validation before further model or generator changes**.

## Source discovery and validation

The requested dated Model 1 directory exists at `model1_diagnostic_audit_2026-10-11_20261011_231247/`. Discovery found that the listed summary, fold metrics, prediction rows, runner, diagnostic manifest, and validation results all exist there. The manifest binds the run to the revised dataset SHA-256 `0db6c6d33f0a3d580b2f673cc3761b4c2551f48c1ca68d296303d9854a669b15`; the successful run's validation file says all 11 source checks passed and its preservation manifest verified 200/200 protected hashes. No timestamped alternatives matching the requested Model 2 diagnostic directory were found; the available companion diagnostic is `model2_diagnostic_audit_2026-10-11/`.

This analysis independently recomputed each outer-fold and saved-test metric from `model2_downstream_predictions.csv`. Maximum absolute difference from the stored metrics was less than `9e-16`. For each arm:

- Outer CV evaluates the same 4,000 training-partition rows once, in the same five 800-row validation folds.
- The separate saved-test evaluation uses the same 1,000 held-out rows.
- Actual target values and fold assignments match across arms.
- Outer-fold means and sample standard deviations are kept separate from saved-test metrics.
- All arms use the same Model 2 estimator configuration: `RandomForestRegressor` with 200 trees, depth 20, minimum leaf size 2, seed 42 (`n_jobs=1` in this audit).

### Methodology by path

- **A — saved Model 1 predictions:** frozen production Model 1 inference pipelines generate the nine process-valued Model 2 inputs. These artifacts were trained before this comparison and are not reselected on the evaluated outer folds.
- **B — selected Model 1 predictions:** within each outer fold, process-model family/configuration selection uses only that outer training partition. Model 2 training features are cross-fitted; each OOF configuration is selected inside its OOF fitting portion. The outer validation process predictions come from upstream models fitted only on outer training. The held-out test arm similarly uses training-only selection/cross-fitting and training-fitted upstream models.
- **C — oracle:** the eight customer inputs plus the nine **recorded actual process targets** are supplied to Model 2 for training and evaluation. This is explicitly non-deployable and diagnostic only; actual post-run processes are not available to the deployed pipeline.

The oracle values are kept in a separate arm. The deployable A/B frames are formed from predicted process columns and the customer inputs. Model 2 fitting loops over the same arm-specific rows with the same estimator. Source-level review and row-level prediction checks found no use of outer-validation process targets in B's upstream model fit/selection and no oracle rows in A/B. The comparison does not prove that the oracle is a mathematical upper bound for every possible model or feature set.

The companion report's metrics are contextual only. Its feature-group RF ablation uses the same revised target and saved process bundle, but it is a separate audit run and is not substituted for the validated A/B/C predictions here. Its nested family comparison uses all 17 inference inputs with different candidate models; that comparison is discussed separately below, not mixed into this three-arm table.

## Three-path results

MAE and RMSE are percentage points of `Operational_Efficiency_pct`; lower is better. R² is higher-is-better. Outer-fold variability is the sample standard deviation over five folds.

| Path | Deployable? | Outer rows / folds | MAE mean ± SD | RMSE mean ± SD | R² mean ± SD | Saved-test rows | Saved-test MAE / RMSE / R² | Folds better than A (MAE / RMSE / R²) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — saved Model 1 inference predictions | Yes | 4,000 / 5 (800 each) | 4.7690 ± 0.1106 | 5.9096 ± 0.1307 | 0.0545 ± 0.0286 | 1,000 | 5.0470 / 6.5094 / 0.0670 | baseline |
| B — outer-training-selected Model 1 OOF | Yes | 4,000 / 5 (800 each) | 4.7685 ± 0.1447 | 5.9315 ± 0.1826 | 0.0479 ± 0.0266 | 1,000 | 4.9921 / 6.4642 / 0.0799 | 3/5 / 1/5 / 1/5 |
| C — recorded-process oracle | **No** | 4,000 / 5 (800 each) | 3.0671 ± 0.0862 | 4.0594 ± 0.1676 | 0.5540 ± 0.0234 | 1,000 | 3.1808 / 4.6513 / 0.5236 | 5/5 / 5/5 / 5/5 |

### Practical magnitude

- **B versus A, outer CV:** MAE improves by just **0.0005 pp** (about 0.01% of A's MAE); RMSE worsens by **0.0219 pp**; R² decreases by **0.0066**. Direction is inconsistent by fold: B lowers MAE in 3/5 folds, RMSE in 1/5, and improves R² in 1/5. This is not a material, consistent downstream gain.
- **B versus A, saved test:** B is slightly better by **0.0549 pp MAE**, **0.0453 pp RMSE**, and **0.0129 R²**. This one held-out result is small relative to errors around 5–6 pp and does not outweigh the inconsistent outer-fold evidence.
- **C versus A, outer CV:** oracle MAE is lower by **1.7019 pp** (35.7% of A's MAE), RMSE lower by **1.8502 pp** (31.3%), and R² higher by **0.4995**; all three directions improve in all five folds. The held-out gap is also large: MAE lower by **1.8661 pp**, RMSE lower by **1.8582 pp**, R² higher by **0.4566**.
- Even the oracle's outer-fold R² is about **0.554**, leaving substantial target variation unexplained by this RF using actual process values and customer inputs. It is a strong diagnostic contrast, not a deployable score or guaranteed ceiling.

## What this says about the bottleneck

### Saved Model 1 errors matter, but the tested replacement did not fix Model 2

Oracle C's consistent improvement shows that the recorded process values carry predictive information for this synthetic target that is lost when replaced by predictions. In that limited sense, process-prediction error is an important bottleneck.

However, B is not an effective demonstrated remedy: despite training-only selection and cross-fitting, it is effectively tied with A on mean MAE and slightly worse in mean RMSE/R². The companion Model 1 diagnostic found strong improvement in the held-out synthetic speed targets for training-selected process models (line speed R² 0.809, puller speed R² 0.700 versus frozen artifacts at −0.408 and −0.351), but that did not translate into material Model 2 outer-fold improvement. Therefore, those individual Model 1 speed gains do not establish that Model 2's main bottleneck is solved by changing Model 1.

### Available predicted features have limited incremental signal

The companion revised-data Model 2 diagnostic's same-fold RF ablation found:

- Customer-only: MAE 4.7848, RMSE 5.9570, R² 0.0394.
- Predicted-process-only: MAE 4.7801, RMSE 5.9283, R² 0.0488.
- All 17 customer plus predicted-process inputs: MAE 4.7706, RMSE 5.9122, R² 0.0536.
- Customer plus inner-selected subset of predicted process inputs: MAE 4.7571, RMSE 5.9095, R² 0.0547.

These are modest differences against the fold-mean baseline (MAE 4.9063; R² −0.0042). Together with A/B, they suggest the *predicted* process values add little usable information under the current synthetic task and estimator. They do not show that customer features have no relationship to efficiency.

### Model family/training setup is not the leading evidenced cause

The companion Model 2 audit compared multiple families under its leakage-controlled folds using the 17-input stack. Ridge was best with MAE 4.6744 ± 0.1478, RMSE 5.8065 ± 0.1706, R² 0.0877 ± 0.0114; RF scored MAE 4.7709 ± 0.1081, RMSE 5.9134 ± 0.1384, R² 0.0534 ± 0.0267. Ridge is a modest improvement, but still explains little of the synthetic target variation. This makes model family a secondary contributor worth noting—not evidence that extensive tuning is the main remedy. These separate algorithm results are not directly interchangeable with oracle C, whose three paths intentionally hold the estimator fixed.

## Synthetic target construction and information gap

### Verified implementation facts

The revised generator computes:

1. `Actual_Processing_Length_m = Line_Speed_m_min × Run_Time_min` (rounded to the stored length precision).
2. `Scrap_Length_m = Actual_Processing_Length_m × Scrap_Rate`; `Good_Length_m = Actual_Processing_Length_m − Scrap_Length_m`.
3. `Availability = Run_Time_min / Planned_Production_Time_min`.
4. `Performance = realized Line_Speed_m_min / nominal_line_speed_for_spec(spec)`.
5. `Quality = Good_Length_m / Actual_Processing_Length_m`.
6. `OEE = Availability × Performance × Quality`.
7. `Operational_Efficiency_pct = 100 × OEE`.

The synthetic fault, risk, operating, and scrap functions use generated context and random variation. For example, fault selection depends on a generated risk score, shift and machine context; downtime depends on the selected fault and generated severity; scrap includes process deviations, material/geometry context, a fault penalty, and random noise. The intended model input contract includes customer fields and nine process predictions, but excludes `Machine_ID`, `Shift`, fault code, downtime, planned production time, runtime, scrap rate, and output accounting fields.

The companion accounting validator passed the identities, including OEE and target derivation, and validated completion as `Good_Length_m >= Customer_Required_Length_m`. This verifies internal synthetic accounting consistency, not factual validity of assumptions or metrics.

### Empirical patterns in this synthetic sample

The target has mean 77.09%, standard deviation 6.23 pp, range 16.55–92.42%; it has 4,934 distinct values in 5,000 rows. No performance values are at or above 1 and no efficiency values are at or above 100%.

Across all 5,000 generated records, target correlations are Availability **+0.588**, Performance **+0.688**, Quality **+0.622**, Downtime **−0.584**, and Scrap_Rate **−0.622**. These component associations partly follow from the formulas and correlated generated variables; they are not independent causal estimates. Availability averages 0.986 (median 1.0); downtime averages 6.51 minutes (median 0, q75 9.90); `No Fault` is 70.48% of rows. Performance and quality both vary, and target variation is not reducible to downtime alone.

### Plausible explanation, not a causal finding

The oracle provides observed line speed, temperatures, pressure, RPM, puller speed, and extrusion rate, so Model 2 can recover synthetic process-to-performance/quality relationships more directly. The deployed arms only have estimated versions of those process values. In addition, availability, quality, and efficiency are affected by generated latent conditions and noise. Since several key event/accounting quantities are intentionally not inputs, weak deployable R² is compatible with missing information in this synthetic setup.

This evidence cannot establish how much of the remaining error is from particular generator assumptions, measurement error, Model 2 representation, missing event inputs, or real process behaviour. In particular, it cannot validate the synthetic fault probabilities, nominal-rate formula, severity distributions, process ranges, or fault impacts as Polyhose-specific facts or limits.

## One recommendation

**Request suitable historical production data for validation.**

Obtain consented, governed, time-aligned records with customer/recipe inputs, actual process measurements, machine and shift, planned production time, run time/downtime and reason codes, accepted output, scrap/rework, and order requirement/status. Validate accounting definitions and feature availability with manufacturing owners, then use machine/time-held-out evaluation. This is the most justified next action because the current target is wholly synthetic, the oracle gap demonstrates dependence on actual outcomes that deployable estimates do not reproduce, and model-family changes on synthetic data remain modest. The evidence is not strong enough to decide whether the generator or the deployment feature design should be revised to match real operation.

## Limitations

- All cited datasets and outcomes are synthetic; no real factory data, limits, or causal relationships were validated.
- A/B/C evaluate one fixed RF setup; family-level results are from the separate companion audit.
- C is non-deployable and compares actual post-run process values; it is not a production-performance estimate and not a guaranteed model ceiling.
- Outer-fold standard deviations describe only five shuffled folds of this synthetic dataset.
- A uses frozen production-trained Model 1 artifacts; B retrains/selects process models on the revised synthetic training rows. Their training feature distributions differ by design.
- The report does not test alternative target definitions, feature designs, time/machine splits, or fault-specific model strategies.

Reproducible source checks, row alignment, metric recalculation, and calculation details are recorded in `oracle_methodology_validation.json`. Original large predictions and reports remain in their source audit directories and are referenced in `diagnostic_manifest.json`.
