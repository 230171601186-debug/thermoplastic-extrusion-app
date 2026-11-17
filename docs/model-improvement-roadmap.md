# Model Improvement Roadmap

## Current evidence

The application uses a synthetic 5,000-row dataset. The figures below come from the controlled candidate evaluation artifacts and the fixed 1,000-row held-out split. They are not Polyhose production results.

| Target | Five-fold outer-CV MAE | Outer-CV R² | Held-out MAE | Held-out R² |
|---|---:|---:|---:|---:|
| Operational efficiency | 6.42 percentage points | 0.174 | 6.37 percentage points | 0.230 |
| Scrap rate | 2.19 percentage points | 0.322 | 2.09 percentage points | 0.346 |

The full-stack candidates improve only modestly over customer-only baselines on the held-out split:

- Efficiency MAE changes from 6.49 to 6.37 percentage points.
- Scrap-rate MAE changes from 2.13 to 2.09 percentage points.

The R² values show that substantial outcome variation remains unexplained. Hyperparameter tuning alone is unlikely to solve this because the current planning-time input set does not contain much of the operational context that drives efficiency and scrap.

## Why efficiency prediction is constrained

The current customer-facing model receives product and order requirements, then estimates process parameters before estimating efficiency. The dataset also contains machine, shift, ambient, planned-production, downtime, and run-time fields, but most are intentionally excluded from the current UI contract.

For planning-time prediction, the most useful next features to validate with process engineers are:

- Machine ID and machine family, with enough historical runs per machine
- Shift and planned production window
- Ambient temperature and humidity, where relevant to the process
- Planned processing length and planned production time
- Material-grade / tube-type / braid-type history on each machine
- Changeover and setup context, maintenance state, and recent machine condition
- Historical machine-level availability and quality summaries computed only from prior runs

Do not use actual downtime, actual run time, actual processing length, scrap length, good length, final fault codes, or other post-run outcomes in a planning-time model. They would leak information that is unavailable when the prediction is made.

## Split the problem into three products

### 1. Planning-time efficiency forecast

Predict expected efficiency before a production run. Use order, material, machine assignment, shift, planned schedule, and historical machine/product context. Evaluate on later time periods and unseen or separately held-out machines.

### 2. Live production nowcast

Estimate current OEE/efficiency during a run from timestamped telemetry and production counts. Inputs can include live process values, actual run time, downtime events, produced length, accepted length, and rejects—but only after each value becomes available. Use time-respecting validation and define prediction latency.

### 3. Downtime risk and cause analysis

Build a separate task using timestamped downtime events, duration, machine, fault code, maintenance records, and pre-event sensor history. Define the prediction horizon (for example, risk in the next 30 or 60 minutes) before training. Do not treat a post-event fault code as a predictor of that same event.

## Data required for a credible pilot

Request a governed production extract with:

- Unique production-run ID, machine ID, shift, start/end timestamps, product specification, material lot, and planned schedule
- Time-stamped sensor readings with units, sampling frequency, missing-data flags, and calibration context
- Downtime start/end, reason/fault code, planned vs unplanned status, and maintenance work-order linkage
- Actual produced quantity/length, accepted quantity/length, scrap quantity/length, and inspection/defect outcomes
- Documented definitions for OEE, availability, performance, quality, operational efficiency, scrap rate, and downtime

Keep raw records immutable. Resolve duplicate events, clock drift, missing intervals, machine changes, and inconsistent units before building features.

## Evaluation and release gates

1. Agree target definitions and when each feature becomes available with production, quality, and maintenance stakeholders.
2. Establish simple baselines (historical machine mean, product-family mean, and engineering rules).
3. Use chronological validation; add machine-held-out evaluation where generalization to new machines matters.
4. Report MAE/RMSE/R² by machine, material, product family, shift, and time period—not just one aggregate score.
5. Inspect residuals, subgroup error, calibration, drift, and out-of-domain inputs.
6. Run in read-only shadow mode and compare predictions with actual outcomes before any operational use.
7. Require process-engineering and quality approval before any output can be considered a setpoint or production recommendation.

## Current release status

The current application is a software and modeling proof of concept on synthetic data. It is suitable for demonstrating input validation, model integration, reporting, and evaluation workflow. It is not evidence of validated factory performance and must not be used to set machine parameters or make production decisions.
