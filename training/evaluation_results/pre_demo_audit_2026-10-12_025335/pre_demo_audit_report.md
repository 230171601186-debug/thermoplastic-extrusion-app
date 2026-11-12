# Polyhose Predictive Analytics — Pre-Demo Quality Audit

**Scope:** read-only audit of the existing Streamlit demonstration app, model artifacts, training and inference contracts, saved synthetic evaluation, dependencies, and tests. No production or app source was changed, and no model was promoted.

## Executive result

**The app runs and produces an end-to-end prediction for a valid default input.** Streamlit 1.58.0 rendered the app locally, loaded the candidate bundle, and returned nine process values, operational efficiency, and scrap percentage. The disclaimer correctly states that the models use synthetic data and are not validated Polyhose factory settings or production recommendations. Both repository test modules passed: 18 tests total.

**The app is suitable only as a clearly labelled synthetic proof-of-concept demo—not as a production planner or source of operating setpoints.** The main pre-demo risks are model provenance (the app actually loads isolated candidate artifacts, not the `models/` production artifacts), weak-to-moderate predictive performance on synthetic data, and legacy cross-validation methodology that is not fully leakage-controlled. Factory-level accuracy and process safety have not been established.

## What is implemented

- Eight customer-facing inputs are present: material type, grade, tube type, braid type, OD, wall, ID and required length.
- `"None"` is an explicit braid category and the default selection. The CSV loader preserves it rather than treating it as missing.
- Positive numeric values, non-finite/missing values, extra/missing fields and tube geometry `|ID - (OD - 2 × wall)| < 0.25 mm` are validated in the inference bundle. Unknown nonblank categories are accepted by the fitted encoder's `ignore` policy and warned about.
- The two downstream models and all nine upstream Model 1 pipelines consume exactly the eight customer inputs plus the nine process predictions (17 ordered features); post-run outcomes are excluded. The inference loader checks manifest agreement, schema, category vocabularies and that the Model 2/3 manifests point to identical Model 1 files.
- Efficiency is formatted as percent; scrap is stored as a fraction and multiplied by 100 for display. Nine process outputs have unit labels and deterministic save/reload predictions matched in the bundle audit.
- The app currently calls the process values **“Recommended Process Parameters”** even though its banner disclaims operational validity. Avoid showing those values as approved setpoints during the demo.

## Active inference path and artifact authority

The entry point imports `CandidateInferenceBundle` from the controlled-stacking bundle audit. Its default loader follows manifests in `training/evaluation_results/controlled_stacking/model2_candidate/` and `model3_candidate/`, then loads candidate Model 1 and downstream candidate pickles. The production artifacts under `models/process/`, `models/efficiency/`, and `models/quality/` are not the app's active inference path.

Both candidate manifests identify the production CSV (`dataset_sha256=cbd42684bc475ca9a7ad4f5b4512962c785034f2aae7c4944f0804e624c20396`), a fixed 4,000/1,000 split, and seed 42. These are still **isolated candidates**, despite the fact that the UI actively serves them. This makes candidate-vs-production status a release/governance issue; do not describe the app artifacts as production-approved.

The application catches exceptions during loading/prediction and presents generic safe messages; `InferenceValidationError` details are surfaced for validated inputs. This protects users from raw exception text, but the app module does not add explicit diagnostic logging for generic failures.

## Model quality evidence

The available dataset has 5,000 rows and 40 columns; all evidence below is synthetic. The existing saved 80/20 split contains 4,000 training and 1,000 test rows (seed 42). “Saved-test” rows below are not cross-validation averages. Production-artifact results and candidate-bundle results are reported separately.

### Model 1 — process parameter predictions

The saved training metrics compare Random Forest, XGBoost and LightGBM with training-only CV. The selected algorithm is taken from `models/model_metadata.json` (selected using training CV), and its held-out test metrics are shown below; the test set was not used to choose the algorithm. For every process target, train-only mean and median test baselines have R² approximately zero.

| Target | Selected saved model | MAE | RMSE | R² | Mean baseline MAE / RMSE | Median baseline MAE / RMSE |
|---|---|---:|---:|---:|---:|---:|
| Extruder temperature (°C) | LightGBM | 3.7175 | 4.7352 | 0.6243 | 6.2070 / 7.7293 | 6.1951 / 7.7257 |
| Die temperature (°C) | XGBoost | 4.7580 | 5.9909 | 0.5097 | 6.9842 / 8.5568 | 6.9727 / 8.5571 |
| Cooling tank temperature (°C) | Random Forest | 0.8873 | 1.2648 | 0.2869 | 1.0785 / 1.4980 | 1.0758 / 1.5040 |
| Melt pressure (bar) | LightGBM | 4.7644 | 5.9648 | 0.5152 | 6.9027 / 8.5678 | 6.8968 / 8.5665 |
| Air pressure (bar) | Random Forest | 0.5475 | 0.6846 | 0.4459 | 0.7447 / 0.9201 | 0.7450 / 0.9203 |
| Screw speed (RPM) | LightGBM | 3.5112 | 4.4317 | 0.6028 | 5.5993 / 7.0345 | 5.6090 / 7.0456 |
| Line speed (m/min) | XGBoost | 0.7766 | 0.9921 | 0.7894 | 1.7731 / 2.1657 | 1.7726 / 2.1646 |
| Puller speed (m/min) | LightGBM | 0.9470 | 1.1958 | 0.7191 | 1.8461 / 2.2581 | 1.8456 / 2.2570 |
| Extrusion rate (kg/hr) | LightGBM | 5.6326 | 7.0501 | 0.6471 | 9.6177 / 11.8677 | 9.6178 / 11.8673 |

These metrics use the selected production artifacts recorded in `models/model_metadata.json` and the saved fixed test split. The app does **not** use these production Model 1 pickles; it uses the Model 1 candidate files referenced by the candidate manifests. The table indicates learnable structure in the synthetic population, not physical setpoint accuracy. Separate Model 1 diagnostics on the revised synthetic dataset are a transfer test against a different target population and cannot be interpreted as paired evidence.

### Model 2 — operational efficiency

Target MAE/RMSE are percentage points.

| Artifact/evaluation | MAE | RMSE | R² |
|---|---:|---:|---:|
| Saved production Model 2, held-out synthetic test | 6.4060 | 8.4618 | 0.2276 |
| Isolated corrected candidate Model 2, same held-out rows | 6.3699 | 8.4481 | 0.2301 |
| Corrected candidate nested outer CV, 5 folds | 6.4234 ± 0.1894 | 8.3809 ± 0.3490 | 0.1739 ± 0.0321 |
| Training-mean baseline, held-out test | 7.3868 | — | approximately 0 |
| Training-mean baseline, nested outer CV | 7.1593 ± 0.1554 | 9.2293 ± 0.3179 | −0.0017 ± 0.0009 |

Candidate change versus saved artifact on the same test rows is small: MAE improves 0.0361 pp, RMSE improves 0.0137 pp, and R² improves 0.0025. These differences do not establish a meaningful model improvement. The candidate uses its own leakage-controlled nested evaluation, which is the more defensible cross-validation estimate; the legacy `training/results/model2_metrics.csv` CV results should not be compared as if they were methodologically identical.

### Model 3 — scrap rate

The target is stored as a fraction. Parenthesized MAE values are percentage points after multiplying by 100.

| Artifact/evaluation | MAE | RMSE | R² |
|---|---:|---:|---:|
| Saved production Model 3, held-out synthetic test | 0.02044 (2.0436 pp) | 0.02665 | 0.3347 |
| Isolated corrected candidate Model 3, same held-out rows | 0.02095 (2.0945 pp) | 0.02641 | 0.3464 |
| Corrected candidate nested outer CV, 5 folds | 0.02188 ± 0.00052 (2.1876 ± 0.0520 pp) | 0.02693 ± 0.00057 | 0.3222 ± 0.0375 |
| Training-mean baseline, held-out test | 0.02710 (2.7102 pp) | 0.03267 | approximately 0 |

The candidate has slightly lower RMSE and higher R² than the saved production artifact but higher MAE on the same test rows. This is a mixed result, not an unambiguous promotion case.

## Leakage and train/inference consistency

**Inference schema parity is good:** Model 2 and Model 3 receive the same ordered 17 input fields in training manifests and inference. The active inference bundle checks those contracts, the paired upstream pipeline file hashes, and output shape/finiteness/bounds. Saved candidate predictions were reproducible on reload.

**Legacy CV needs qualification:**

- Model 2 computes Model 1 out-of-fold features once for the entire training split and then cross-validates Model 2 on that matrix. For an outer Model 2 validation fold, Model 1 training used to produce some outer-training rows' features can include other outer-validation rows. This is not fully nested upstream/downstream evaluation.
- Model 3 constructs its training stack by predicting the training rows with Model 1 pipelines fitted on those same rows, then runs downstream CV. This directly makes training-fold process features in-sample with respect to Model 1.
- Held-out test process predictions are generated from Model 1 pipelines fitted on training rows, so the saved test result is distinct from the contaminated legacy CV procedure. It remains a single synthetic random split, not temporal/site/machine generalization.
- Corrected controlled-stacking nested-CV results and candidate artifacts exist. Use those for cautious synthetic comparisons and report legacy CV separately.

No post-production target columns are included in the app's predictor frame. The oracle analysis is not used or surfaced by the app.

## Application behavior, dependencies and limitations

### Verified behavior

- Exactly eight input widgets; the default geometry (OD 10, wall 2, ID 6) passes validation.
- `Braid_Type="None"` renders and predicts successfully.
- A valid click produced finite, bounded process, efficiency and scrap predictions. Repeated serialized inference was deterministic in the existing bundle audit.
- Missing/blank/non-finite fields, unsupported feature columns, invalid geometry and invalid model outputs are rejected by the candidate inference tests.
- `requirements-app.txt` installs `requirements/ml_requirements.txt` via relative include and requires Streamlit `>=1.58.0`; the selected environment has Streamlit 1.58.0 and required ML modules available for execution.

### Remaining application risks

- Finite numeric values are checked for positivity and geometry but are not compared with training support. There is no confidence interval or explicit numeric out-of-distribution warning. Do not invent company limits; derive any support messaging from observed training coverage and validate it with process owners.
- An unseen category is accepted and warns that the encoder ignores it; predictions for such combinations are not validated and can be weak.
- The inference checks individual process values against synthetic generator bounds only. Those are not approved machine limits and do not enforce consistency among all nine predicted values.
- Unit labels and percentage conversion are correct in inspected paths. The labels “Recommended Process Parameters” and live efficiency/scrap values can still invite overinterpretation despite the disclaimer.
- Only helper-level UI tests exist; the bundle suite does not cover actual user-facing error-state rendering or deployment packaging.
- No root README, deployment configuration, or CI test workflow was discovered in the workspace. The app ran locally, but deployment compatibility, clean-machine dependency installation, browser accessibility, and hosted runtime were not tested.

## Prioritized issues and recommended order

See [`prioritized_issues.csv`](./prioritized_issues.csv) for evidence, impact and confirmation tests.

1. **Clarify active artifact status before the demo.** State visibly that the app is using isolated synthetic candidate artifacts and show their provenance; do not call them production-approved.
2. **Keep usage explicitly non-operational.** The synthetic test metrics are modest and no real factory data has validated accuracy, calibration or physical acceptability.
3. **Use corrected nested evaluation for model claims.** Clearly retire/qualify legacy CV numbers and keep test-set evaluation separate.
4. **Add training-support messaging and UI integration tests.** Do not invent physical ranges or promise confidence.
5. **Validate joint process plausibility and process constraints with authorized manufacturing evidence** before moving beyond a proof-of-concept.

**Recommendation:** preserve the current app as a synthetic demo only. For an internship presentation, lead with the synthetic-data disclaimer and the artifact-status caveat, and avoid presenting outputs as approved operating settings. The next evidence-based improvement is to obtain governed historical production data with product/machine/context, recorded settings, downtime/faults, produced/scrap quantities and timestamps, then design a temporal or machine/site holdout with responsible process-owner review. Do not automatically promote the candidate bundle.

## Integrity and test status

`artifact_integrity_manifest.json` records per-file SHA-256 values for the dataset, generator/training code, production models/metadata, app, dependencies and all existing evaluation-result artifacts. `preservation_verification.json` reports the post-audit comparison. The integrity utility and all newly created audit files are excluded from the protected baseline.

The app and inference suites passed (18/18 tests) and a valid end-to-end browser submission succeeded. The standalone `pytest` command was unavailable because pytest is not installed; no package was added. See [`test_coverage_summary.md`](./test_coverage_summary.md) for exact commands and scope.
