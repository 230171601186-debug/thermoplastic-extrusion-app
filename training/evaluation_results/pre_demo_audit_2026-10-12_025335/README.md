# Pre-demo quality audit

Read-only audit of the existing synthetic-data Streamlit demonstration app. No production dataset, generator, model, metadata, inference code, or app source was changed. The audit outputs and integrity-check utility are isolated in this directory.

## Reproduction

Run from the project root in PowerShell. The commands below use Python 3.13.14, which was the configured interpreter for this audit; use the project's configured Python executable if it differs.

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
& "C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe" -m unittest -v app.tests.test_streamlit_app
& "C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe" -m unittest -v training.evaluation_results.controlled_stacking.bundle_audit.tests.test_candidate_inference
& "C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe" -m streamlit --version
```

For a local browser smoke test (stop the server with Ctrl+C after checking it):

```powershell
& "C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe" -m streamlit run app/streamlit_app.py --server.headless true --server.port 8765 --browser.gatherUsageStats false
```

The captured integrity manifest covers the production dataset and generator/training code, production models and metadata, app files, dependency manifests, and every file present in `training/evaluation_results` before this audit directory was written. Verify after reproducing checks:

```powershell
& "C:\Users\khale\AppData\Local\Microsoft\WindowsApps\python3.13.exe" training/evaluation_results/pre_demo_audit_2026-10-12_025335/run_integrity_check.py verify
```

`run_integrity_check.py capture` intentionally replaces the baseline manifest and should only be used when deliberately establishing a new baseline.

## Deliverables

- `pre_demo_audit_report.md` — findings, evidence, priorities and recommendation.
- `prioritized_issues.csv` — issue register.
- `test_coverage_summary.md` — test and runtime results.
- `artifact_integrity_manifest.json` — pre-report protected-file SHA-256 hashes.
- `preservation_verification.json` — post-audit verification.
- `diagnostic_manifest.json` — scope, methods, evidence paths and unresolved limitations.
