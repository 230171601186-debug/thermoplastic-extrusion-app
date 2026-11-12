"""Verify protected production and pre-existing experiment files are unchanged."""

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
        (OUT / "preservation_manifest.json").read_text(encoding="utf-8")
    )
    mismatches = []
    for relative_path, expected in manifest["protected_files_sha256"].items():
        path = ROOT / Path(relative_path)
        if not path.is_file():
            mismatches.append({"path": relative_path, "issue": "missing"})
            continue
        actual = sha256(path)
        if actual != expected:
            mismatches.append({
                "path": relative_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
            })

    production_dataset = ROOT / "dataset/Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"
    result = {
        "verified": not mismatches,
        "verified_file_count": len(manifest["protected_files_sha256"]) - len(mismatches),
        "expected_file_count": len(manifest["protected_files_sha256"]),
        "production_dataset_sha256_before": manifest["production_dataset_sha256"],
        "production_dataset_sha256_after": sha256(production_dataset),
        "mismatches": mismatches,
    }
    result["verified"] = (
        result["verified"]
        and result["production_dataset_sha256_before"]
        == result["production_dataset_sha256_after"]
    )
    (OUT / "preservation_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
