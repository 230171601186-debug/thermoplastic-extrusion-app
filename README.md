# Thermoplastic Extrusion Intelligence

A Streamlit-based predictive analytics proof of concept for thermoplastic extrusion. The application translates customer product and order specifications into estimated process parameters, operational efficiency, and scrap rate.

> **Scope and safety:** This repository currently uses a synthetic 5,000-row dataset. The candidate models have not been validated against Polyhose production records, live Faclon telemetry, or controlled factory trials. Predictions are research/demo estimates—not machine settings, process instructions, quality acceptance decisions, or production advice.

## Product capabilities
- Customer-facing configuration for material, grade, tube type, braid type, tube geometry, and required length.
- Nine estimated process parameters covering temperatures, pressures, screw/line/puller speeds, and extrusion rate.
- Candidate operational-efficiency and scrap-rate estimates.
- Input geometry and category validation before inference.
- Output schema, unit, finiteness, and target-scale validation before display.
- Unit-separated charts and downloadable CSV prediction snapshots.
- Cached model loading and automated application/inference tests.

## Architecture
```text
.
├── app/
│   ├── streamlit_app.py
│   └── tests/
├── dataset/
│   └── Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv
├── models/
│   ├── process/
│   ├── efficiency/
│   ├── quality/
│   ├── model_metadata.json
│   └── data_split.json
├── training/
│   ├── config.py
│   ├── preprocessing.py
│   ├── model1_process_parameters.py
│   ├── model2_efficiency.py
│   ├── model3_quality.py
│   ├── evaluate.py
│   └── evaluation_results/
└── requirements-app.txt
```

The inference bundle loads nine upstream process estimators and two downstream estimators. The downstream candidate models consume the process estimates alongside the eight customer inputs. Candidate manifests and audit artefacts record feature contracts and evaluation evidence.

## Run locally (Windows PowerShell)
From the repository root:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```
Open the local URL printed by Streamlit (normally `http://localhost:8501`).

If PowerShell blocks virtual-environment activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

## Run tests
Install the training/test dependencies listed in `training/ml_requirements.txt`, then run from the repository root:
```powershell
python -m unittest discover -s app/tests -p "test_*.py" -v
```
The application tests cover input mapping and validation, inference output contracts, unit conversions, report generation, model caching, and Streamlit interactions.

## Model evaluation: interpret carefully
The checked-in `models/model_metadata.json` records held-out test metrics. For regression, MAE and RMSE are in each target's native unit; R² measures explained variance relative to the test-set baseline. A low or negative R² means the model may be only marginally better than predicting the test-set mean (or worse).

Current saved downstream test metrics:

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| Operational efficiency | 6.406 percentage points | 8.462 percentage points | 0.228 |
| Scrap rate | 0.0204 fraction (about 2.04 percentage points) | 0.0266 fraction | 0.335 |

These values describe performance on a held-out split of synthetic data only. They do not establish generalisation to real manufacturing conditions. In particular, the efficiency model's R² is weak and should be treated as a major research limitation, not hidden behind a polished dashboard.

## Data and model governance
- Do not commit credentials, API keys, `.env` files, or Streamlit secrets.
- Do not upload confidential company data or personally identifiable information without explicit authorization.
- Preserve the checked-in model artifacts and manifests when changing application presentation.
- Any retraining must use reproducible splits, training-only preprocessing, leakage checks, cross-validation, and a final untouched test set.
- Do not promote a candidate model based only on training score or a single favourable split.
- A production claim requires representative plant data, documented sensor/label definitions, temporal validation, drift monitoring, operator review, and controlled acceptance testing.

## Current status
**Academic / engineering proof of concept.** The application is structured for further experimentation and UI development, but the current synthetic-data models are not production-ready.