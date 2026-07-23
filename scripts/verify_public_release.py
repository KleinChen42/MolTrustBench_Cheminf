#!/usr/bin/env python3
"""Verify the distributed compact MolTrustBench public case-study package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
MANIFEST = DATA / "PUBLIC_DATA_MANIFEST.json"


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        require(path.is_file(), f"missing: {entry['path']}", errors)
        if path.is_file():
            require(sha256(path) == entry["sha256"], f"hash mismatch: {entry['path']}", errors)

    chemistry = read_rows("Table_J2_chemistry_overlap_summary.csv")
    require(len(chemistry) == 1, "chemistry summary must contain one row", errors)
    if chemistry:
        row = chemistry[0]
        expected = {
            "benchmark_accepted_n": "1978",
            "public_unique_parent_n": "2194629",
            "strict_identity_overlap_n": "346",
            "connectivity_identity_overlap_n": "360",
            "standardized_parent_overlap_n": "373",
            "scaffold_exposed_n": "1167",
            "scaffold_unexposed_n": "810",
            "scaffold_not_applicable_n": "1",
            "outcome_values_accessed": "False",
        }
        for key, value in expected.items():
            require(row.get(key) == value, f"unexpected {key}: {row.get(key)!r}", errors)
        require(
            int(row["scaffold_exposed_n"]) + int(row["scaffold_unexposed_n"]) + int(row["scaffold_not_applicable_n"]) == int(row["benchmark_accepted_n"]),
            "scaffold counts do not partition the accepted benchmark panel",
            errors,
        )
        transform_hash = row["transform_hash"]
    else:
        transform_hash = ""

    for name in ("Table_J2_public_nn_summary.csv", "Table_J2_train_nn_summary.csv"):
        rows = read_rows(name)
        require(len(rows) == 1, f"{name} must contain one row", errors)
        if rows:
            require(rows[0].get("outcome_values_accessed") == "False", f"{name} must remain outcome-free", errors)
            require(rows[0].get("transform_hash") == transform_hash, f"{name} transform hash mismatch", errors)

    anchors = read_rows("Table_J2_anchor_bracketing_sensitivity.csv")
    require([row.get("release_id") for row in anchors] == ["CHEMBL30", "CHEMBL32", "CHEMBL33"], "anchor rows must be CHEMBL30/32/33", errors)
    require(all(row.get("benchmark_accepted_n") == "1978" for row in anchors), "anchor benchmark-panel size mismatch", errors)
    require(all(row.get("transform_hash") == transform_hash for row in anchors), "anchor transform hash mismatch", errors)
    require(all(row.get("outcome_values_accessed") == "False" for row in anchors), "anchor sensitivity must remain outcome-free", errors)

    fingerprints = read_rows("Table_J2_fingerprint_sensitivity.csv")
    require(len(fingerprints) == 8, "fingerprint grid must contain eight declared configurations", errors)
    require({row.get("radius") for row in fingerprints} == {"2", "3"}, "fingerprint radius grid mismatch", errors)
    require({row.get("n_bits") for row in fingerprints} == {"1024", "2048"}, "fingerprint bit-length grid mismatch", errors)
    require({row.get("use_chirality") for row in fingerprints} == {"False", "True"}, "fingerprint chirality grid mismatch", errors)
    require(all(row.get("outcome_values_accessed") == "False" for row in fingerprints), "fingerprint sensitivity must remain outcome-free", errors)

    receipt = {
        "protocol": "moltrustbench-public-compact-verification-v1",
        "status": "pass" if not errors else "fail",
        "checked_files": len(manifest["files"]),
        "errors": errors,
        "interpretation_boundary": "public-source observability under declared release and chemistry contracts; no training-membership, memorization, leakage, or causal-performance inference",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
