# Thermoplastic Extrusion Predictive Analytics
## Internship Project Technical Report

**Organization:** Polyhose India Pvt. Ltd.  
**Project area:** Predictive analytics for thermoplastic extrusion  
**Application:** Streamlit machine-learning decision-support prototype  
**Report date:** 13 November 2026

## 1. Abstract

This project develops a machine-learning prototype to estimate selected thermoplastic extrusion process parameters, operational efficiency, and scrap rate from product specifications and customer order requirements. The solution combines a staged ML inference pipeline with a Streamlit dashboard for input validation, prediction review, model evaluation, scenario comparison, and descriptive downtime analysis.

The current implementation uses a **synthetic 5,000-row dataset**. It has not been validated against Polyhose production records, live machine telemetry, or a controlled factory trial. It is therefore a research and demonstration prototype, not a production-validated system or a source of approved machine settings.

## 2. Problem statement and objectives

Thermoplastic extrusion involves interacting product and process variables. This project explores whether product specifications and order requirements can support data-driven estimates of process parameters and downstream production indicators.

The objectives are to:
1. Develop a structured dataset and staged ML workflow.
2. Estimate process parameters, operational efficiency, and scrap rate.
3. Validate user inputs and model outputs before displaying estimates.
4. Build a professional dashboard for planning, process review, and model governance.
5. Compare alternative order-length scenarios while holding other inputs constant.
6. Analyze approved historical downtime records descriptively.
7. Document performance, limitations, and requirements for factory-data validation.

## 3. Dataset and governance

The checked-in dataset is `dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv`, containing 5,000 synthetic records. It was created for model development and application testing; it is not an extract of actual Polyhose production history.

The training quality gate checks required columns, missing values, blank categorical fields, non-numeric and non-finite numeric values, and duplicate rows. Invalid datasets are rejected rather than silently modified.

Synthetic data is useful for testing the software pipeline, but cannot establish that learned relationships represent actual factory behavior. Production validation requires approved plant records, confirmed data definitions and units, timestamps, machine identifiers, and process-engineering review.

## 4. System architecture

The candidate inference pipeline has three stages:

- **Model 1 – Process parameter estimation:** predicts extruder temperature, die temperature, cooling-tank temperature, melt pressure, air pressure, screw speed, line speed, puller speed, and extrusion rate.
- **Model 2 – Operational efficiency estimation:** predicts operational efficiency from customer inputs and estimated process information.
- **Model 3 – Scrap-rate estimation:** predicts scrap rate from customer inputs and estimated process information.

The Streamlit application checks input categories, finite numeric values, tube geometry, output fields, units, and prediction ranges before displaying a result. It also flags numeric inputs outside the synthetic training-data range.

## 5. Application modules

### Production planning
Accepts material type, material grade, tube type, braid type, tube OD, wall thickness, tube ID, and customer-required length. It validates the specification, generates nine process estimates plus efficiency and scrap estimates, and exports a CSV prediction report.

### Process intelligence
Displays the latest successful prediction in an outcome overview, process breakdown, and order-profile view. Charts are separated by engineering unit to avoid misleading comparisons.

### Production intelligence
- Stores the latest 100 successful predictions in the active Streamlit session and supports CSV export.
- Compares an alternative order length with the latest successful input profile, keeping other inputs fixed.
- Displays changes in estimated efficiency and scrap rate as model-output differences, not causal effects.
- Accepts downtime CSV files with `Timestamp`, `Machine_ID`, `Downtime_min`, and `Fault_Code`.
- Validates records and summarizes downtime by machine, fault code, and date.

Prediction history is session-scoped, not persistent shared storage. Downtime analysis is descriptive only; no machine-failure prediction or root-cause model is implemented.

### Model governance
Displays five-fold outer cross-validation and fixed held-out test metrics for the downstream candidate models, with explicit limitations and synthetic-data warnings.

## 6. Model evaluation

The downstream candidates were evaluated using five-fold outer cross-validation and a fixed 1,000-row held-out synthetic test split.

| Target | Evaluation | R² | MAE |
|---|---|---:|---:|
| Operational efficiency | 5-fold outer CV | 0.174 | 6.42 percentage points |
| Operational efficiency | Held-out test | 0.230 | 6.37 percentage points |
| Scrap rate | 5-fold outer CV | 0.322 | 2.19 percentage points |
| Scrap rate | Held-out test | 0.346 | 2.09 percentage points |

The models explain only a limited portion of the variation in the synthetic held-out data. The efficiency model is particularly weak. The scrap-rate model performs better but is not a dependable production quality predictor. Full-stack improvements over customer-only baselines are small, so the current results do not justify claims of strong predictive performance.

Model 1 performance also varies by target. Saved held-out R² values range approximately from 0.29 for cooling-tank temperature to 0.79 for line speed. Each process target requires separate review.

## 7. Software validation

The repository includes automated checks for customer-input mapping, geometry validation, prediction output schema and units, safe inference-error handling, report generation, prediction history, downtime data validation, dataset quality, candidate inference contracts, and application startup.

The upgrade branch is `feature/production-intelligence-upgrades`. Commit `7e699fb5555917e1134dc1332b709cd10de25fa3` had a successful GitHub Actions run before the latest scenario-comparison and downtime-trend changes. Those latest changes require a fresh CI run and hosted end-to-end verification before being described as fully verified.

## 8. Limitations

1. The dataset is synthetic and is not verified Polyhose production history.
2. Efficiency and scrap-rate models have modest predictive performance.
3. No live IoT or machine telemetry integration is implemented.
4. Prediction history is limited to the active application session.
5. Downtime analysis requires an uploaded CSV and is descriptive only.
6. Scenario comparison changes order length only; it does not prove causality or optimize machine settings.
7. No factory trial, shadow-mode deployment, production acceptance test, or process-engineering approval has been completed.

## 9. Recommended future work

1. Obtain approved, representative production records and confirm data definitions, units, machine IDs, timestamps, and target calculations.
2. Implement time-aware and machine-aware validation to prevent leakage across production runs.
3. Compare candidate models against engineering and historical-average baselines; report MAE, RMSE, R², residual diagnostics, and performance by machine and product family.
4. Add persistent prediction history, access controls, and auditability.
5. Integrate validated downtime and production records through an approved data pipeline.
6. Monitor data drift, model error, missing telemetry, and out-of-domain inputs.
7. Run the system in shadow mode and obtain process-engineering and quality approval before any operational use.

## 10. Conclusion

The project delivers an end-to-end prototype combining staged process and production-indicator estimates, a Streamlit dashboard, input/output validation, model-governance views, session-based prediction history, order-length what-if comparison, and descriptive downtime analysis.

The work demonstrates the software and analytical workflow required for further development. However, the current models use synthetic data and have modest predictive performance. The next major improvement is obtaining approved production data and evaluating the models under machine-aware and time-aware conditions—not simply increasing model complexity. The system should currently be presented as a research and decision-support prototype, not a production-ready industrial AI system.
