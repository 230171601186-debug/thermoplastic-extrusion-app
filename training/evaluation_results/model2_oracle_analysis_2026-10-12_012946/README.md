# Model 2 Oracle Analysis

Read-only analysis of the existing validated Model 1 diagnostic run, its three Model 2 feature paths, and the companion revised-dataset Model 2 audit. No models, data, generators, metadata, inference contracts, or application code are changed. No experiment is rerun.

## Reproduction

From the repository root, with the configured Python environment:

```powershell
& 'C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe' 'training\evaluation_results\model2_oracle_analysis_2026-10-12_012946\run_oracle_analysis.py'
```

The script reads the saved summary/fold/prediction CSVs and manifests, verifies identical rows and targets across arms, recomputes fold and test metrics from predictions, checks the source run's validation results and dataset hash, and records methodology checks. It does not fit any model or generate data.

## Outputs

- `oracle_analysis_report.md`: methodology, result interpretation, synthetic accounting trace, recommendation, and limitations.
- `oracle_comparison_summary.csv`: three paths with CV means/standard deviations, held-out metrics, row/fold counts, deployability, and directional fold consistency.
- `oracle_methodology_validation.json`: row-alignment and metric-recalculation checks, code-path checks, fold consistency, and limitations.
- `diagnostic_manifest.json`: SHA-256 hashes and relative paths for read-only source artifacts.
- `run_oracle_analysis.py`: reproduction/validation script.

Large prediction tables and prior diagnostic results remain in their existing source directories:

- `../model1_diagnostic_audit_2026-10-11_20261011_231247/`
- `../model2_diagnostic_audit_2026-10-11/`
- `../model2_generator_experiment_2026-10-11_revised/`

Every result concerns synthetic data; the oracle arm uses post-run process targets and is not deployable.
