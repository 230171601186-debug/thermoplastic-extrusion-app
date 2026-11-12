"""Verify files captured in the pre-audit preservation manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(
        (OUT / "production_preservation_manifest.json").read_text(encoding="utf-8-sig")
    )
    mismatches = []
    for relative, expected in manifest["file_sha256"].items():
        path = ROOT / Path(relative)
        if not path.is_file():
            mismatches.append({"path": relative, "issue": "missing"})
        elif sha256(path) != expected:
            mismatches.append(
                {
                    "path": relative,
                    "expected": expected,
                    "actual": sha256(path),
                }
            )
    result = {
        "verified": not mismatches,
        "verified_file_count": len(manifest["file_sha256"]) - len(mismatches),
        "expected_file_count": len(manifest["file_sha256"]),
        "production_dataset_sha256_before": manifest["dataset_sha256"],
        "production_dataset_sha256_after": sha256(
            ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
        ),
        "existing_experiment_directory_file_counts": manifest[
            "experiment_directory_file_counts"
        ],
        "mismatches": mismatches,
    }
    (OUT / "production_preservation_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
