# Isolated model-pipeline remediation experiment

This directory records artifact lineage, audits the active eight-input inference contract, and evaluates fixed Model 1/2/3 pipelines with leakage-safe nested upstream cross-fitting. It is a synthetic-data investigation only. Nothing in this directory is promoted or wired into production.

## Reproduction

From the project root (`C:\Users\khale\OneDrive\Pictures\Desktop\app`) in the configured Python 3.13.14 environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python training\evaluation_results\model_pipeline_remediation_2026-10-12_030200\run_pipeline_experiment.py
python training\evaluation_results\model_pipeline_remediation_2026-10-12_030200\validate_experiment.py
python training\evaluation_results\model_pipeline_remediation_2026-10-12_030200\create_residual_diagnostics.py
python -m unittest discover -s training\evaluation_results\model_pipeline_remediation_2026-10-12_030200 -p test_pipeline_contract.py -v
python training\evaluation_results\model_pipeline_remediation_2026-10-12_030200\integrity.py verify
```

The experiment uses the existing dataset and `models/data_split.json`, seed 42, 4,000 training rows and 1,000 held-out rows, five outer folds and four inner upstream OOF folds. Estimator families/parameters are fixed in advance and recorded in `experiment_manifest.json`; no search or test-based selection occurs. The runner writes row-level predictions and metric tables. The residual diagnostics script creates the process/downstream residual summaries, quartile table, and PNG figures from those predictions.

To refresh the original protected-file baseline before an entirely new run, first preserve the existing manifest elsewhere; then run `integrity.py capture`. Do not recapture after analysis as a way to hide a preservation mismatch.

## Contents

- `pipeline_lineage_report.md`: active app artifact identity and training lineage.
- `inference_contract_audit.md`: customer input and output contract, checks, and unsupported-domain behavior.
- `evaluation_methodology_report.md`: fixed split/fold protocol and matched Model 1/2/3 results.
- `prioritized_remediation_plan.csv`: classified findings, priority, proposed correction, tests, and safety.
- `test_results.md`: exact commands and outcomes, including resolved transient command/validator issues.
- `artifact_integrity_manifest.json`, `preservation_verification.json`: pre-work hashes and post-work verification for protected production and prior experiment files.
- `experiment_manifest.json`, `validation_results.json`: dataset/model parameters and executable checks.
- CSVs and PNGs: saved predictions, fold metrics, residual summaries, and diagnostic plots.
- `run_pipeline_experiment.py`, `validate_experiment.py`, `create_residual_diagnostics.py`, `integrity.py`, `test_pipeline_contract.py`: re-runnable isolated tooling.

## Preservation and limitations

The hash scope includes app, dataset, model files/metadata, training source and prior evaluation artifacts, and dependency manifests, excluding only this new experiment directory. Run the `verify` command after all project actions. All model/data findings remain synthetic; use governed real production history and agreed label definitions before any production inference or performance claim.
