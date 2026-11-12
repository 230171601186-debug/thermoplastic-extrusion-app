# Test and validation results

## Passed

| Check | Command / method | Result |
|---|---|---|
| Streamlit helper unit tests | `python -m unittest -v app.tests.test_streamlit_app` | PASS — 2 tests |
| Candidate inference tests | `python -m unittest -v training.evaluation_results.controlled_stacking.bundle_audit.tests.test_candidate_inference` | PASS — 16 tests |
| Isolated pipeline contract tests | `python -m unittest discover -s training/evaluation_results/model_pipeline_remediation_2026-10-12_030200 -p test_pipeline_contract.py -v` | PASS — 4 tests, including explicit outer/test partition exclusion |
| Experiment output validation | `python training/evaluation_results/model_pipeline_remediation_2026-10-12_030200/validate_experiment.py` | PASS — all 14 recomputation, leakage, schema, coverage and bound checks |
| Python syntax | AST parse of all `.py` files under `app/` and `training/` (including the isolated scripts) | PASS — 42 files, 0 syntax errors |
| Streamlit startup | `python -m streamlit run app\streamlit_app.py --server.headless true --server.port 8765` and `curl.exe --fail http://127.0.0.1:8765/_stcore/health` | PASS — HTTP health body `ok`; server then stopped |
| Prediction contract smoke | Loaded `CandidateInferenceBundle`; standard valid input returned nine process outputs plus efficiency and scrap with finite values and documented units | PASS |
| Preservation | `python training/evaluation_results/model_pipeline_remediation_2026-10-12_030200/integrity.py verify` | See `preservation_verification.json`; must remain PASS after all report creation |

## Failed/transient checks and resolution

- Initial invocation of the experiment test by dotted module name failed because the isolated directory contains hyphens and is not an importable Python package. The prescribed `unittest discover -s ... -p test_pipeline_contract.py` invocation passed all three tests.
- The first validator execution found a validator implementation issue: NumPy booleans were not JSON serializable, and its expected OOF-audit count omitted the separate saved-test upstream fit. The isolated validator was corrected, rerun, and all checks passed (216 expected audit rows).
- The first health-check attempt used PowerShell `Invoke-WebRequest` and was not usable in this noninteractive environment. `curl.exe` succeeded against the same health endpoint (`ok`).
- The first ad hoc syntax command had a quoting error, not a source syntax error. Replaced with an AST parse over all 42 project Python sources; zero parse errors.

After adding explicit outer-fold/test exclusion evidence to the isolated audit, the fixed experiment was rerun. The final isolated tests and 14-check validator passed.

These resolved attempts are disclosed for reproducibility; they do not represent unresolved project test failures.

## Environment

Python 3.13.14; NumPy 2.4.6; pandas 3.0.3; scikit-learn 1.9.0; joblib 1.5.3. Project environment was already configured. `pytest` was not used; all available required suites ran with the standard-library `unittest` runner.

## Remaining limitations

No real-data validation, time-based split, machine/site holdout, operational boundary approval, or production model promotion was performed. None is implied by the passing software tests.
