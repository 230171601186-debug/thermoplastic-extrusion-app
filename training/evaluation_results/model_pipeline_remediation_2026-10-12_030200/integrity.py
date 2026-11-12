from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_files() -> list[Path]:
    files: set[Path] = set()
    for directory in (ROOT / "app", ROOT / "dataset", ROOT / "models", ROOT / "training"):
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            try:
                path.relative_to(EXPERIMENT_DIR)
            except ValueError:
                files.add(path)
    for path in (ROOT / "requirements-app.txt", ROOT / "training" / "ml_requirements.txt"):
        if path.is_file():
            files.add(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def snapshot() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in protected_files()
    }


def capture() -> None:
    hashes = snapshot()
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "file_count": len(hashes),
        "scope": [
            "all app and dataset files",
            "all model artifacts and metadata",
            "all training source and pre-existing experiment outputs",
            "root app and ML dependency manifests",
        ],
        "excluded": [
            "the new model_pipeline_remediation experiment directory",
            "runtime files outside the project",
        ],
        "sha256_by_relative_path": hashes,
    }
    (EXPERIMENT_DIR / "artifact_integrity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Captured hashes for {len(hashes)} protected files.")


def verify() -> None:
    baseline_path = EXPERIMENT_DIR / "artifact_integrity_manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["sha256_by_relative_path"]
    current = snapshot()
    changed = sorted(
        path for path in baseline.keys() & current.keys() if baseline[path] != current[path]
    )
    missing = sorted(baseline.keys() - current.keys())
    added = sorted(current.keys() - baseline.keys())
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "protected_file_count_before": len(baseline),
        "protected_file_count_after": len(current),
        "changed": changed,
        "missing": missing,
        "added": added,
        "passed": not changed and not missing and not added,
    }
    (EXPERIMENT_DIR / "preservation_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Preservation {'PASSED' if result['passed'] else 'FAILED'}: "
        f"{len(baseline)} before, {len(current)} after, "
        f"{len(changed)} changed, {len(missing)} missing, {len(added)} added."
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "verify"))
    action = parser.parse_args().action
    capture() if action == "capture" else verify()
