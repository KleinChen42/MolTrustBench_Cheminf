"""Build auditable first-observed identity indexes from declared ChEMBL release panels.

This clean-room module never treats a partial historical panel as proof of a
global ChEMBL first-observable release. Every output is explicitly scoped to
the supplied, validated release panel; a future complete-archive certificate is
required before a global claim can be made.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .temporal import ReleaseRecord, validate_release_registry


class ReleaseIndexContractError(ValueError):
    """Raised when a release-panel first-observation contract is incomplete."""


IDENTITY_COLUMNS = (
    ("strict", "strict_key"),
    ("connectivity", "connectivity_key"),
    ("standardized_parent", "standardized_parent_key"),
)
REQUIRED_TRANSFORM_COLUMNS = frozenset(
    {
        "source_id",
        "row_id",
        "role",
        "status",
        "strict_key",
        "connectivity_key",
        "standardized_parent_key",
        "transform_hash",
        "rdkit_version",
    }
)
REQUIRED_RAW_MANIFEST_FIELDS = frozenset(
    {
        "release_id",
        "public_release_date",
        "source_url",
        "raw_sha256",
        "raw_bytes",
        "chemreps_data_rows",
        "input_kind",
    }
)


@dataclass(frozen=True)
class ReleasePanelInput:
    """One official raw release plus its locked-transform records."""

    release_id: str
    transformed_path: Path
    raw_manifest_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_manifest(path: Path, release: ReleaseRecord) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReleaseIndexContractError(f"cannot read raw release manifest: {path}") from exc
    missing = sorted(REQUIRED_RAW_MANIFEST_FIELDS - set(payload))
    if missing:
        raise ReleaseIndexContractError(f"raw release manifest {path} missing: {', '.join(missing)}")
    if str(payload["release_id"]) != release.release_id:
        raise ReleaseIndexContractError(
            f"raw release manifest {path} has release_id {payload['release_id']!r}, expected {release.release_id!r}"
        )
    if release.doi and str(payload.get("release_doi") or "") != release.doi:
        raise ReleaseIndexContractError(f"raw release manifest {path} has a DOI inconsistent with registry")
    if str(payload["public_release_date"]) != release.release_date.isoformat():
        raise ReleaseIndexContractError(f"raw release manifest {path} has a date inconsistent with registry")
    if release.release_timestamp is not None:
        if str(payload.get("public_release_timestamp") or "") != release.release_timestamp.isoformat():
            raise ReleaseIndexContractError(
                f"raw release manifest {path} has a timestamp inconsistent with registry"
            )
    raw_hash = str(payload["raw_sha256"])
    if not (raw_hash.startswith("sha256:") or len(raw_hash) == 64):
        raise ReleaseIndexContractError(f"raw release manifest {path} has an invalid raw_sha256")
    if int(payload["raw_bytes"]) < 1:
        raise ReleaseIndexContractError(f"raw release manifest {path} has a non-positive raw_bytes value")
    if int(payload["chemreps_data_rows"]) < 1:
        raise ReleaseIndexContractError(f"raw release manifest {path} has a non-positive chemreps_data_rows value")
    if str(payload["input_kind"]) != "official_chemreps_gzip":
        raise ReleaseIndexContractError(f"raw release manifest {path} must describe official_chemreps_gzip input")
    return payload


def _accepted_release_rows(
    path: Path, release: ReleaseRecord
) -> tuple[list[dict[str, str]], str, str, dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_TRANSFORM_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ReleaseIndexContractError(f"transformed release {path} missing: {', '.join(missing)}")
        all_rows = [dict(row) for row in reader]
    rows = [row for row in all_rows if str(row.get("status", "")) == "accepted"]
    if not rows:
        raise ReleaseIndexContractError(f"transformed release {path} has no accepted rows")
    bad_roles = sorted({row["role"] for row in rows if row["role"] != "public"})
    if bad_roles:
        raise ReleaseIndexContractError(
            f"transformed release {path} has accepted non-public role(s): {', '.join(bad_roles)}"
        )
    bad_sources = sorted({row["source_id"] for row in rows if row["source_id"] != release.release_id})
    if bad_sources:
        raise ReleaseIndexContractError(
            f"transformed release {path} has source IDs inconsistent with {release.release_id}: {', '.join(bad_sources)}"
        )
    transform_hashes = {row["transform_hash"] for row in rows if row["transform_hash"]}
    rdkit_versions = {row["rdkit_version"] for row in rows if row["rdkit_version"]}
    if len(transform_hashes) != 1:
        raise ReleaseIndexContractError(f"transformed release {path} must have exactly one non-empty transform hash")
    if len(rdkit_versions) != 1:
        raise ReleaseIndexContractError(f"transformed release {path} must have exactly one non-empty RDKit version")
    for _, identity_column in IDENTITY_COLUMNS:
        if any(not row[identity_column] for row in rows):
            raise ReleaseIndexContractError(f"accepted row in {path} lacks {identity_column}")
    rejection_reasons = Counter(
        str(row.get("reason_code") or "missing_rejection_reason")
        for row in all_rows
        if str(row.get("status", "")) != "accepted"
    )
    accounting: dict[str, object] = {
        "input_n": len(all_rows),
        "accepted_n": len(rows),
        "rejected_n": len(all_rows) - len(rows),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    return rows, next(iter(transform_hashes)), next(iter(rdkit_versions)), accounting

def build_first_observed_index(
    panel_inputs: Sequence[ReleasePanelInput], release_registry: Iterable[ReleaseRecord]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build deterministic per-identity first-observation records.

    The inputs may cover any declared subset of the verified release registry.
    Therefore this result is always limited to the declared panel; no caller can
    promote it into a global historical first-seen statement through this function.
    """

    releases = {record.release_id: record for record in validate_release_registry(release_registry)}
    if len(panel_inputs) < 2:
        raise ReleaseIndexContractError("first-observed indexing requires at least two releases")
    input_ids = [item.release_id for item in panel_inputs]
    if len(input_ids) != len(set(input_ids)):
        raise ReleaseIndexContractError("release panel has duplicate release IDs")
    unknown = sorted(set(input_ids) - set(releases))
    if unknown:
        raise ReleaseIndexContractError(f"release panel IDs are absent from verified registry: {', '.join(unknown)}")

    ordered_inputs = sorted(panel_inputs, key=lambda item: (releases[item.release_id].ordering_timestamp, item.release_id))
    all_hashes: set[str] = set()
    all_rdkit_versions: set[str] = set()
    release_lineage: list[dict[str, object]] = []
    first: dict[tuple[str, str], dict[str, object]] = {}

    for item in ordered_inputs:
        release = releases[item.release_id]
        raw_manifest = _load_raw_manifest(item.raw_manifest_path, release)
        rows, transform_hash, rdkit_version, accounting = _accepted_release_rows(item.transformed_path, release)
        all_hashes.add(transform_hash)
        all_rdkit_versions.add(rdkit_version)
        release_lineage.append(
            {
                "release_id": release.release_id,
                "release_date": release.release_date.isoformat(),
                "release_timestamp": release.release_timestamp.isoformat() if release.release_timestamp else "",
                "date_evidence_kind": release.date_evidence_kind,
                "release_doi": release.doi,
                "raw_manifest_path": item.raw_manifest_path.as_posix(),
                "raw_manifest_sha256": sha256_file(item.raw_manifest_path),
                "raw_sha256": str(raw_manifest["raw_sha256"]),
                "raw_bytes": int(raw_manifest["raw_bytes"]),
                "chemreps_data_rows": int(raw_manifest["chemreps_data_rows"]),
                "transformed_path": item.transformed_path.as_posix(),
                "transformed_sha256": sha256_file(item.transformed_path),
                "transform_accounting": accounting,
            }
        )
        for row in rows:
            for identity_level, identity_column in IDENTITY_COLUMNS:
                identity_key = row[identity_column]
                first.setdefault(
                    (identity_level, identity_key),
                    {
                        "identity_level": identity_level,
                        "identity_key": identity_key,
                        "first_observed_release": release.release_id,
                        "first_observed_date": release.release_date.isoformat(),
                        "first_observed_timestamp": release.ordering_timestamp.isoformat(),
                        "first_observed_source_id": row["source_id"],
                        "first_observed_row_id": row["row_id"],
                    },
                )

    if len(all_hashes) != 1:
        raise ReleaseIndexContractError("all transformed release inputs must share exactly one transform hash")
    if len(all_rdkit_versions) != 1:
        raise ReleaseIndexContractError("all transformed release inputs must share exactly one RDKit version")

    records = sorted(
        first.values(),
        key=lambda row: (
            str(row["identity_level"]),
            str(row["identity_key"]),
            str(row["first_observed_release"]),
            str(row["first_observed_row_id"]),
        ),
    )
    manifest: dict[str, object] = {
        "first_observation_scope": "first_observed_in_declared_release_panel",
        "global_chEMBL_first_observable_claim_permitted": False,
        "global_claim_blocker": (
            "This panel has not been separately certified as a complete, contiguous official ChEMBL archive "
            "from the chosen origin through its cutoff."
        ),
        "release_panel_n": len(ordered_inputs),
        "release_panel_ids": [item.release_id for item in ordered_inputs],
        "release_lineage": release_lineage,
        "identity_record_n": len(records),
        "identity_record_n_by_level": {
            level: sum(1 for row in records if row["identity_level"] == level)
            for level, _ in IDENTITY_COLUMNS
        },
        "transform_hash": next(iter(all_hashes)),
        "rdkit_version": next(iter(all_rdkit_versions)),
        "outcome_values_accessed": False,
    }
    return records, manifest


def write_first_observed_index(
    records: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    *,
    output: Path,
    manifest_path: Path,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "identity_level",
        "identity_key",
        "first_observed_release",
        "first_observed_date",
        "first_observed_timestamp",
        "first_observed_source_id",
        "first_observed_row_id",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})
    payload = dict(manifest)
    payload.update({"output_path": output.as_posix(), "output_sha256": sha256_file(output)})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_and_write_first_observed_index_sqlite(
    panel_inputs: Sequence[ReleasePanelInput],
    release_registry: Iterable[ReleaseRecord],
    *,
    output: Path,
    manifest_path: Path,
    work_db: Path,
) -> dict[str, object]:
    """Build a panel-scoped index with bounded Python memory.

    The SQLite database is an execution workspace, not an evidentiary input.
    It retains the earliest row encountered under the verified release ordering
    and input-row ordering, matching the in-memory algorithm's ``setdefault``
    behavior without materializing all historical identity keys in Python.
    """

    releases = {record.release_id: record for record in validate_release_registry(release_registry)}
    if len(panel_inputs) < 2:
        raise ReleaseIndexContractError("first-observed indexing requires at least two releases")
    input_ids = [item.release_id for item in panel_inputs]
    if len(input_ids) != len(set(input_ids)):
        raise ReleaseIndexContractError("release panel has duplicate release IDs")
    unknown = sorted(set(input_ids) - set(releases))
    if unknown:
        raise ReleaseIndexContractError(f"release panel IDs are absent from verified registry: {', '.join(unknown)}")
    if output.exists() or manifest_path.exists() or work_db.exists():
        raise ReleaseIndexContractError("SQLite first-observed output, manifest, or work database already exists")

    ordered_inputs = sorted(panel_inputs, key=lambda item: (releases[item.release_id].ordering_timestamp, item.release_id))
    work_db.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(work_db)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TABLE first_observed (
                identity_level TEXT NOT NULL,
                identity_key TEXT NOT NULL,
                first_observed_release TEXT NOT NULL,
                first_observed_date TEXT NOT NULL,
                first_observed_timestamp TEXT NOT NULL,
                first_observed_source_id TEXT NOT NULL,
                first_observed_row_id TEXT NOT NULL,
                PRIMARY KEY (identity_level, identity_key)
            ) WITHOUT ROWID
            """
        )
        all_hashes: set[str] = set()
        all_rdkit_versions: set[str] = set()
        release_lineage: list[dict[str, object]] = []
        insert_sql = (
            "INSERT OR IGNORE INTO first_observed "
            "(identity_level, identity_key, first_observed_release, first_observed_date, "
            "first_observed_timestamp, first_observed_source_id, first_observed_row_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        for item in ordered_inputs:
            release = releases[item.release_id]
            raw_manifest = _load_raw_manifest(item.raw_manifest_path, release)
            with item.transformed_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                missing = sorted(REQUIRED_TRANSFORM_COLUMNS - set(reader.fieldnames or ()))
                if missing:
                    raise ReleaseIndexContractError(f"transformed release {item.transformed_path} missing: {', '.join(missing)}")
                input_n = accepted_n = 0
                rejection_reasons: Counter[str] = Counter()
                release_hashes: set[str] = set()
                release_rdkit_versions: set[str] = set()
                batch: list[tuple[str, str, str, str, str, str, str]] = []
                for row in reader:
                    input_n += 1
                    if str(row.get("status") or "") != "accepted":
                        rejection_reasons[str(row.get("reason_code") or "missing_rejection_reason")] += 1
                        continue
                    accepted_n += 1
                    if row.get("role") != "public":
                        raise ReleaseIndexContractError(f"transformed release {item.transformed_path} has accepted non-public role")
                    if row.get("source_id") != release.release_id:
                        raise ReleaseIndexContractError(f"transformed release {item.transformed_path} has source ID inconsistent with {release.release_id}")
                    transform_hash = str(row.get("transform_hash") or "")
                    rdkit_version = str(row.get("rdkit_version") or "")
                    if not transform_hash or not rdkit_version:
                        raise ReleaseIndexContractError(f"accepted row in {item.transformed_path} lacks transform or RDKit version")
                    release_hashes.add(transform_hash)
                    release_rdkit_versions.add(rdkit_version)
                    for identity_level, identity_column in IDENTITY_COLUMNS:
                        identity_key = str(row.get(identity_column) or "")
                        if not identity_key:
                            raise ReleaseIndexContractError(f"accepted row in {item.transformed_path} lacks {identity_column}")
                        batch.append(
                            (
                                identity_level,
                                identity_key,
                                release.release_id,
                                release.release_date.isoformat(),
                                release.ordering_timestamp.isoformat(),
                                str(row["source_id"]),
                                str(row["row_id"]),
                            )
                        )
                    if len(batch) >= 50000:
                        connection.executemany(insert_sql, batch)
                        batch.clear()
                if batch:
                    connection.executemany(insert_sql, batch)
                if accepted_n < 1:
                    raise ReleaseIndexContractError(f"transformed release {item.transformed_path} has no accepted rows")
                if len(release_hashes) != 1 or len(release_rdkit_versions) != 1:
                    raise ReleaseIndexContractError(f"transformed release {item.transformed_path} must have exactly one transform hash and RDKit version")
                all_hashes.update(release_hashes)
                all_rdkit_versions.update(release_rdkit_versions)
                connection.commit()
            release_lineage.append(
                {
                    "release_id": release.release_id,
                    "release_date": release.release_date.isoformat(),
                    "release_timestamp": release.release_timestamp.isoformat() if release.release_timestamp else "",
                    "date_evidence_kind": release.date_evidence_kind,
                    "release_doi": release.doi,
                    "raw_manifest_path": item.raw_manifest_path.as_posix(),
                    "raw_manifest_sha256": sha256_file(item.raw_manifest_path),
                    "raw_sha256": str(raw_manifest["raw_sha256"]),
                    "raw_bytes": int(raw_manifest["raw_bytes"]),
                    "chemreps_data_rows": int(raw_manifest["chemreps_data_rows"]),
                    "transformed_path": item.transformed_path.as_posix(),
                    "transformed_sha256": sha256_file(item.transformed_path),
                    "transform_accounting": {
                        "input_n": input_n,
                        "accepted_n": accepted_n,
                        "rejected_n": input_n - accepted_n,
                        "rejection_reasons": dict(sorted(rejection_reasons.items())),
                    },
                }
            )
        if len(all_hashes) != 1:
            raise ReleaseIndexContractError("all transformed release inputs must share exactly one transform hash")
        if len(all_rdkit_versions) != 1:
            raise ReleaseIndexContractError("all transformed release inputs must share exactly one RDKit version")

        counts = {level: 0 for level, _ in IDENTITY_COLUMNS}
        for level, count in connection.execute("SELECT identity_level, COUNT(*) FROM first_observed GROUP BY identity_level"):
            counts[str(level)] = int(count)
        identity_record_n = sum(counts.values())
        if identity_record_n < 1:
            raise ReleaseIndexContractError("SQLite first-observed index contains no identity records")
        temporary_output = output.with_name(output.name + ".partial")
        if temporary_output.exists():
            raise ReleaseIndexContractError("SQLite first-observed temporary output already exists")
        fieldnames = (
            "identity_level", "identity_key", "first_observed_release", "first_observed_date",
            "first_observed_timestamp", "first_observed_source_id", "first_observed_row_id",
        )
        with temporary_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in connection.execute(
                "SELECT identity_level, identity_key, first_observed_release, first_observed_date, "
                "first_observed_timestamp, first_observed_source_id, first_observed_row_id "
                "FROM first_observed ORDER BY identity_level, identity_key, first_observed_release, first_observed_row_id"
            ):
                writer.writerow(dict(zip(fieldnames, row, strict=True)))
        os.replace(temporary_output, output)
    finally:
        connection.close()

    manifest: dict[str, object] = {
        "first_observation_scope": "first_observed_in_declared_release_panel",
        "global_chEMBL_first_observable_claim_permitted": False,
        "global_claim_blocker": (
            "This panel has not been separately certified as a complete, contiguous official ChEMBL archive "
            "from the chosen origin through its cutoff."
        ),
        "release_panel_n": len(ordered_inputs),
        "release_panel_ids": [item.release_id for item in ordered_inputs],
        "release_lineage": release_lineage,
        "identity_record_n": identity_record_n,
        "identity_record_n_by_level": counts,
        "transform_hash": next(iter(all_hashes)),
        "rdkit_version": next(iter(all_rdkit_versions)),
        "outcome_values_accessed": False,
        "execution_backend": "sqlite_streaming_outcome_free",
        "sqlite_version": sqlite3.sqlite_version,
    }
    manifest.update({"output_path": output.as_posix(), "output_sha256": sha256_file(output)})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest