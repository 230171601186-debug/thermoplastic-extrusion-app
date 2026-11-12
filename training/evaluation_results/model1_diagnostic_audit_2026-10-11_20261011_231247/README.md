# Controlled Model 1 Diagnostic

Isolated, read-only analysis of the revised synthetic dataset. Production assets are protected by the pre-analysis SHA-256 inventory in `preservation_manifest.json`.

## Reproduction commands

From the repository root with the configured Python 3.13 environment:

```powershell
& 'C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe' 'training\evaluation_results\model1_diagnostic_audit_2026-10-11_20261011_231247\run_model1_diagnostic.py'
& 'C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe' 'training\evaluation_results\model1_diagnostic_audit_2026-10-11_20261011_231247\verify_preservation.py'
& 'C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe' 'training\evaluation_results\model1_diagnostic_audit_2026-10-11_20261011_231247\validate_diagnostic_outputs.py'
```

The audit reads the revised 5,000 x 40 dataset and saved `models/data_split.json`; it does not generate data. Model 1 candidates compare one fixed setting per requested family over five outer folds; family selection uses three inner folds. Model 2 impact uses the same outer folds, nested three-fold Model 1 cross-fitting for training rows, fixed Model 2 Random Forest hyperparameters, and three upstream feature arms. The oracle arm uses recorded process outcomes and is diagnostic only, never deployable.

## Key deliverables

- `model1_diagnostic_report.md`: findings, limits, and recommendations.
- `model1_metrics.csv`, `baseline_comparison.csv`, `fold_level_results.csv`, `model1_fold_aggregate.csv`, `residual_diagnostics.csv`.
- `feature_relationship_diagnostics.csv`, line/puller diagnostic PNGs, and per-decile files.
- `model2_downstream_impact.csv`, `model2_downstream_summary.csv`, and row predictions.
- `preservation_manifest.json`, `preservation_verification.json`, and `diagnostic_manifest.json`.
- `validation_results.json`: fast, rerunnable checks for output completeness, row/fold counts, finite metrics, plot files and preserved hashes.

All results concern synthetic data and do not validate Polyhose machine behaviour, operating limits, or expected factory performance.
