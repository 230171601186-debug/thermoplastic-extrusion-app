# Test and runtime coverage

## Commands and outcomes

| Check | Result | Evidence |
|---|---|---|
| `python -m unittest -v app.tests.test_streamlit_app` | PASS | 2 tests passed |
| `python -m unittest -v training.evaluation_results.controlled_stacking.bundle_audit.tests.test_candidate_inference` | PASS | 16 tests passed; includes missing/invalid input, geometry, `"None"` braid, unseen categories, bounds, bundle consistency and read-only checks |
| Streamlit version | PASS | Streamlit 1.58.0, satisfying `requirements-app.txt` (`>=1.58.0`) |
| Live Streamlit smoke | PASS | App rendered at localhost; the form loaded the candidate bundle, `"None"` was selected, and a valid submission displayed nine process outputs, efficiency and scrap percentage |
| Python syntax | PASS | Compiled 37 Python files under `app/` and `training/`; zero syntax failures |
| Dataset split/schema/metric checks | PASS | 5,000 × 40 dataset; 4,000/1,000 saved split; candidate bundle audit reproduces the Model 2/3 saved-test metrics |
| `pytest` runner attempt | NOT AVAILABLE | `python -m pytest` failed because `pytest` is not installed in the selected Python environment. No dependency was installed; both repository test modules ran successfully with `unittest`. |

The smoke test exercised one default valid input path; it is not a broad browser, deployment, accessibility, load, or security test. No factory data or operational credentials were used.

## Existing test scope

- `app/tests/test_streamlit_app.py`: only the UI-to-feature-name mapping and scrap fraction-to-percent conversion.
- `training/evaluation_results/controlled_stacking/bundle_audit/tests/test_candidate_inference.py`: 16 tests exercise the serialized candidate inference bundle, customer input validation, geometry constraints, categorical `"None"`/unknown category handling, output bounds/shapes, save/reload consistency and preservation of production artifacts.

No test currently exercises the full UI error states, model provenance/status display, unsupported numeric extrapolation warnings, or joint physical consistency of all nine outputs. Candidate inference tests do not establish manufacturing validity.

The first integrity-manifest capture attempt exposed that root `requirements.txt` is an empty directory, not a file. The integrity utility was corrected to hash the actual `training/ml_requirements.txt` dependency manifest; the subsequent baseline capture succeeded.

## Metric context

Scores below are synthetic dataset results and must not be presented as factory-validated accuracy.

| Model | Current saved-test result | Corrected candidate result | Simple baseline/context |
|---|---|---|---|
| Model 1 process targets | Selected saved-test R² spans 0.2869 (cooling temperature) to 0.7894 (line speed); see the nine per-target MAE/RMSE/R² rows in the audit report. | Revised-dataset Model 1 diagnostic artifacts are transfer results on a different synthetic dataset and are not paired with these scores. | Train-only mean/median baselines have approximately zero R² by construction; per-target baseline MAE/RMSE are listed in the main report. |
| Model 2 efficiency | Saved production artifact: MAE 6.4060 pp, RMSE 8.4618 pp, R² 0.2276. | Candidate: MAE 6.3699 pp, RMSE 8.4481 pp, R² 0.2301; corrected nested-CV MAE 6.4234 ± 0.1894 pp, R² 0.1739 ± 0.0321. | Training-mean saved-test MAE 7.3868 pp; corrected nested-CV training-mean MAE 7.1593 ± 0.1554 pp. |
| Model 3 scrap rate | Saved production artifact: MAE 0.02044 fraction (2.0436 pp), RMSE 0.02665 fraction, R² 0.3347. | Candidate: MAE 0.02095 fraction (2.0945 pp), RMSE 0.02641 fraction, R² 0.3464; corrected nested-CV MAE 0.02188 ± 0.00052 fraction, R² 0.3222 ± 0.0375. | Training-mean saved-test MAE 0.02710 fraction (2.7102 pp), R² approximately 0. |
