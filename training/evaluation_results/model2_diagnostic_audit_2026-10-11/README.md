# Model 2 Diagnostic Audit

Read-only audit of the revised experimental dataset using the saved production Model 1 inference pipelines. The production dataset, generator, model artifacts, metadata, inference contract, and app are protected by the hash manifest.

## Reproduce

From the repository root, using the configured Python 3.13 environment:

```powershell
& 'C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe' 'training\evaluation_results\model2_diagnostic_audit_2026-10-11\run_model2_diagnostic_audit.py'
& 'C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe' 'training\evaluation_results\model2_diagnostic_audit_2026-10-11\finalize_model2_diagnostic_audit.py'
```

The first command runs nested CV and held-out diagnostics; the second verifies production preservation, validates dataset identities, re-fits only the selected training-only model to extract feature importance, and writes this report. Both commands require the repository's configured dependencies and the reviewed revised dataset. `verify_preservation.py` can be run independently to re-check protected hashes.

## Key outputs

- `model2_diagnostic_report.md`: findings, assumptions, and recommendation.
- `dataset_accounting_validation.json`: row-level identity and schema checks.
- `aggregate_model_comparison.csv`, `fold_level_model_comparison.csv`: nested CV and descriptive holdout model metrics.
- `feature_group_ablation_results.csv`, `feature_group_ablation_aggregate.csv`: identical-fold feature-group comparisons.
- `residual_diagnostics.csv`, `subgroup_error_analysis.csv`: holdout residuals and subgroup metrics.
- `selected_model_feature_importance.csv`, `customer_input_sensitivity.csv`: model-specific diagnostics.
- `near_duplicate_numeric_features.csv`: numeric pairs with absolute Pearson correlation ≥0.9999.
- `production_preservation_verification.json`: post-run protected-file SHA-256 verification.

All numerical findings describe synthetic data and are not evidence of factory performance or validated operating limits.
