"""Create a pre-analysis SHA-256 inventory without changing protected files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_OUT = Path(__file__).resolve().parent
OUT = BASE_OUT
if OUT.exists() and any(OUT.iterdir()):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = BASE_OUT.with_name(f"{BASE_OUT.name}_{stamp}")
    suffix = 1
    while OUT.exists():
        OUT = BASE_OUT.with_name(f"{BASE_OUT.name}_{stamp}_{suffix}")
        suffix += 1
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_cache(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def main() -> None:
    paths: set[Path] = set()
    paths.update((ROOT / "models").rglob("*"))
    paths.update((ROOT / "app").rglob("*"))
    paths.update((ROOT / "training/evaluation_results").rglob("*"))
    explicit = [
        ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv",
        ROOT / "training/generate_synthetic_dataset.py",
        ROOT / "training/model1_process_parameters.py",
        ROOT / "training/model2_efficiency.py",
        ROOT / "training/config.py",
        ROOT / "training/preprocessing.py",
        ROOT / "training/evaluate.py",
    ]
    paths.update(explicit)
    protected = {}
    missing = []
    for path in sorted(paths):
        if not path.is_file() or is_cache(path):
            continue
        if OUT in path.parents:
            continue
        if not path.exists():
            missing.append(path.relative_to(ROOT).as_posix())
            continue
        protected[path.relative_to(ROOT).as_posix()] = sha256(path)

    dataset_path = ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
    manifest = {
        "captured_at_local": datetime.now().astimezone().isoformat(),
        "workspace": str(ROOT),
        "output_directory": OUT.relative_to(ROOT).as_posix(),
        "production_dataset_sha256": sha256(dataset_path),
        "protected_file_count": len(protected),
        "protected_files_sha256": protected,
        "missing_expected_paths": missing,
        "scope": [
            "production dataset and generator",
            "models directory including metadata, split, and production artifacts",
            "training Model 1/2 source, configuration, preprocessing and metrics",
            "Streamlit app files",
            "all pre-existing evaluation_results files excluding Python caches and this audit directory",
        ],
        "existing_experiment_file_counts": {
            relative: sum(
                1 for path in (ROOT / relative).rglob("*")
                if path.is_file() and not is_cache(path)
            )
            for relative in (
                "training/evaluation_results/model2_generator_experiment",
                "training/evaluation_results/model2_generator_experiment_2026-10-11",
                "training/evaluation_results/model2_generator_experiment_2026-10-11_revised",
                "training/evaluation_results/model2_diagnostic_audit_2026-10-11",
            )
        },
    }
    output = OUT / "preservation_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_directory": str(OUT),
        "manifest": str(output),
        "production_dataset_sha256": manifest["production_dataset_sha256"],
        "protected_file_count": len(protected),
        "missing_expected_paths": missing,
    }, indent=2))


if __name__ == "__main__":
    main()
