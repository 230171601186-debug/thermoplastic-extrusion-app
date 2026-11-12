from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUDIT_DIR.parents[2]
EVALUATION_RESULTS = PROJECT_ROOT / "training" / "evaluation_results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_files() -> list[Path]:
    expected = [
        PROJECT_ROOT / "dataset" / "Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv",
        PROJECT_ROOT / "training" / "generate_synthetic_dataset.py",
        PROJECT_ROOT / "training" / "config.py",
        PROJECT_ROOT / "training" / "preprocessing.py",
        PROJECT_ROOT / "training" / "evaluate.py",
        PROJECT_ROOT / "training" / "model1_process_parameters.py",
        PROJECT_ROOT / "training" / "model2_efficiency.py",
        PROJECT_ROOT / "training" / "model3_quality.py",
        PROJECT_ROOT / "models" / "model_metadata.json",
        PROJECT_ROOT / "models" / "data_split.json",
        PROJECT_ROOT / "training" / "ml_requirements.txt",
        PROJECT_ROOT / "requirements-app.txt",
    ]
    missing = [path.relative_to(PROJECT_ROOT).as_posix() for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required protected files are missing: {missing}")

    files = set(expected)
    for directory in (
        PROJECT_ROOT / "app",
        PROJECT_ROOT / "models",
        EVALUATION_RESULTS,
    ):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if directory == EVALUATION_RESULTS:
                audit_relative = AUDIT_DIR.relative_to(EVALUATION_RESULTS)
                try:
                    relative = path.relative_to(EVALUATION_RESULTS)
                except ValueError:
                    continue
                if relative == audit_relative or audit_relative in relative.parents:
                    continue
            files.add(path)

    return sorted(files, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def snapshot() -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
        for path in protected_files()
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture() -> None:
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "scope": [
            "production dataset, generator, training/configuration code",
            "production models and metadata",
            "Streamlit app and its local tests",
            "all files in existing training/evaluation_results directories",
            "root dependency manifests",
        ],
        "excluded": [
            "this new audit directory and files created for this audit",
            "runtime state outside the project",
        ],
        "file_count": 0,
        "sha256_by_relative_path": snapshot(),
    }
    payload["file_count"] = len(payload["sha256_by_relative_path"])
    write_json(AUDIT_DIR / "artifact_integrity_manifest.json", payload)
    print(f"Captured SHA-256 for {payload['file_count']} protected files.")


def verify() -> None:
    manifest_path = AUDIT_DIR / "artifact_integrity_manifest.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256_by_relative_path"]
    after = snapshot()
    before_paths = set(before)
    after_paths = set(after)
    changed = sorted(path for path in before_paths & after_paths if before[path] != after[path])
    missing = sorted(before_paths - after_paths)
    added = sorted(after_paths - before_paths)
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "protected_files_before": len(before),
        "protected_files_after": len(after),
        "changed": changed,
        "missing": missing,
        "added": added,
        "passed": not changed and not missing and not added,
    }
    write_json(AUDIT_DIR / "preservation_verification.json", result)
    print(
        f"Preservation verification {'PASSED' if result['passed'] else 'FAILED'}: "
        f"{len(before)} before, {len(after)} after, "
        f"{len(changed)} changed, {len(missing)} missing, {len(added)} added."
    )
    if not result["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture or verify protected project SHA-256 hashes.")
    parser.add_argument("action", choices=("capture", "verify"))
    action = parser.parse_args().action
    if action == "capture":
        capture()
    else:
        verify()


if __name__ == "__main__":
    main()
