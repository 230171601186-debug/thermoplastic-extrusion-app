# Thermoplastic Extrusion Predictive Analytics

A Streamlit-based decision-support prototype for exploring thermoplastic extrusion process estimates from product and order requirements. The project combines a staged machine-learning pipeline, model evaluation artifacts, and an interactive prediction interface.

> **Important scope statement:** The checked-in dataset is synthetic. The current models are a proof of concept and have **not** been validated against Polyhose production records, live machine telemetry, or a controlled factory trial. Outputs are estimates for demonstration and research only—not approved machine setpoints, quality acceptance criteria, or production instructions.

## What the application does

The customer-facing workflow accepts eight planning inputs:

- Material type and material grade
- Tube type and braid type
- Target tube outside diameter, wall thickness, and inside diameter
- Customer-required length

The candidate inference workflow then estimates:

1. Nine process parameters (temperatures, pressures, speeds, and extrusion rate)
2. Operational efficiency (reported on a 0–100 percentage-point scale)
3. Scrap rate (internally represented as a fraction and displayed as a percentage)

The app validates input categories, finite numeric values, tube geometry, prediction schema, units, and output ranges before displaying a result. It also warns when numeric inputs fall outside the checked-in synthetic training-data ranges; these are extrapolation warnings, not factory operating limits. A CSV report can be downloaded for each successful prediction. The Production intelligence workspace adds session-scoped prediction history with CSV export, a what-if comparison for alternative order lengths using the latest successful product profile, and descriptive downtime analysis from an uploaded CSV. Valid downtime uploads are summarized by machine, fault code, and day; missing, malformed, negative, or non-finite values are rejected. Scenario outputs are model comparisons, not causal estimates or approved production plans.

## Architecture

```text
Customer / order requirements
          |
          v
Input validation and category checks
          |
          v
Model 1: process parameter estimates
          |
          +------------------------------+
          |                              |
          v                              v
Model 2: operational efficiency    Model 3: scrap-rate estimate
          |                              |
          +---------------+--------------+
                          |
                          v
                Output contract checks
                          |
                          v
             Streamlit results + CSV report
```

The training code is organized under `training/`; the Streamlit entry point is `app/streamlit_app.py`; the source dataset is under `dataset/`; and model artifacts and evaluation evidence are under `models/` and `training/evaluation_results/`.

## Run locally

Use Python 3.13 in a virtual environment. The serialized candidate models were built with Python 3.13.14; the ML dependencies are pinned to the package versions recorded in their manifests.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

### macOS / Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

The app loads its candidate inference bundle from repository-relative paths. Run it from the repository root so Streamlit, imports, and data paths resolve consistently.

## Deploy a preview to Streamlit Community Cloud

Deploy the feature branch as a preview first; do not merge it into `main` until the hosted app has been checked.

1. In Streamlit Community Cloud, create a new app and select this repository.
2. Select branch `feature/production-intelligence-upgrades`.
3. Set the app entry point to `app/streamlit_app.py`.
4. Open **Advanced settings** and select Python **3.13** to match the serialized candidate-model environment.
5. Deploy. The root `requirements.txt` delegates to the pinned runtime dependencies in `requirements-app.txt`.

After the first deployment, verify the app starts without errors, submit a valid input, confirm invalid tube geometry is rejected, inspect the model-limitations panel, download a CSV report, and review the application logs. A successful GitHub Actions run is not a substitute for this hosted smoke test.

**Deployment status:** A Streamlit Community Cloud preview was previously observed rendering an earlier feature-branch UI. The new `feature/production-intelligence-upgrades` branch still requires a fresh deployment or branch update and hosted end-to-end verification. A working preview is not production approval.

## Dataset quality gate

Every training dataset load now runs a non-destructive structural quality check before model fitting. It checks required model columns, missing required values, blank categorical values, non-numeric and non-finite numeric values, and reports duplicate rows without silently deleting them. A failed check stops training; the loader does not impute, rewrite, or generate replacement records.

The reusable checks live in `training/data_quality.py`, with regression tests in `training/tests/test_data_quality.py`. Run them with:

```bash
python -m pytest training/tests -q
```

## Model training

Training dependencies are listed in `training/ml_requirements.txt`. The training scripts are:

- `training/model1_process_parameters.py`
- `training/model2_efficiency.py`
- `training/model3_quality.py`

Training and evaluation can be computationally expensive. Do not retrain production-facing artifacts or replace the candidate inference bundle without reviewing the experiment protocol, leakage controls, held-out metrics, artifact compatibility, and app tests.

## Validation

The app includes a **Model validation & limitations** panel sourced from the exact downstream candidate evaluation artifacts. It reports five-fold outer-CV and fixed 1,000-row held-out metrics for the efficiency and scrap-rate candidates. These scores are shown as evidence, not as a production-readiness badge.

Run the application tests with:

```bash
python -m pip install pytest
python -m pytest app/tests training/evaluation_results/controlled_stacking/bundle_audit/tests -q
```

Run the full ML training stack only in a separate environment:

```bash
python -m pip install -r training/ml_requirements.txt
python training/model1_process_parameters.py
python training/model2_efficiency.py
python training/model3_quality.py
```

Model 3 training now uses out-of-fold Model 1 predictions for downstream training rows, matching the leakage-control approach already used by Model 2. Held-out test rows are still transformed by Model 1 pipelines fitted only on the training split.

The root `requirements.txt` delegates to `requirements-app.txt` for common cloud deployment discovery. The ML dependencies are pinned to the package versions recorded in the serialized model manifests. GitHub Actions tests the app and candidate-inference contracts on Python 3.12 and 3.13, with the training smoke test on Python 3.13. Training scripts should be retrained only after reviewing the saved split, leakage controls, held-out metrics, and artifact compatibility.

## Project report

See [`docs/internship-project-report.md`](docs/internship-project-report.md) for the project overview, system workflow, model evaluation results, implemented features, limitations, and recommended next steps. The report distinguishes implemented prototype capabilities from future factory-integration work.

## Model improvement roadmap

See [`docs/model-improvement-roadmap.md`](docs/model-improvement-roadmap.md) for the current controlled-evaluation metrics, feature gaps, production-data requirements, and a staged plan for planning-time efficiency forecasting, live production nowcasting, and downtime-risk modeling.

## Data and model governance

Before any real operational pilot:

1. Replace synthetic records with an approved, representative, quality-controlled production extract.
2. Confirm the data dictionary, units, timestamps, machine identifiers, process context, and target definitions with process engineers.
3. Define a time-aware and/or machine-aware validation design that reflects deployment conditions; prevent leakage across repeated production runs.
4. Compare against simple engineering baselines and report MAE/RMSE/R² per target, plus calibration and residual diagnostics.
5. Review out-of-domain inputs and prediction uncertainty; do not silently extrapolate beyond the training distribution.
6. Obtain process-engineering and quality approval before exposing any prediction as a recommended operating value.
7. Monitor drift, performance, and data quality after deployment.

## Repository map

| Path | Purpose |
|---|---|
| `app/streamlit_app.py` | Streamlit user interface and prediction workflow |
| `app/tests/` | App contract and validation tests |
| `docs/internship-project-report.md` | Submission-oriented technical project report |
| `dataset/` | Synthetic modeling dataset |
| `training/` | Data quality gates, preprocessing, training, evaluation, and experiment records |
| `models/` | Serialized pipelines, split metadata, and model metadata |
| `requirements-app.txt` | Runtime dependencies |
| `training/ml_requirements.txt` | ML and training dependencies |

## Current status

This repository is an actively developed prototype. A successful local run demonstrates software and model integration; it does not establish real-world predictive validity. Keep this distinction visible in demos, reports, and any stakeholder presentation.
