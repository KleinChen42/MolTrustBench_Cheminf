"""Outcome-blind chemistry and overlap audit for J2.

This module deliberately separates chemical identity, scaffold status, and
nearest-neighbour similarity. It cannot read targets, predictions, losses, or
evaluation metrics. Every derived record carries the shared transform hash so a
benchmark and a public-library index cannot be silently compared under different
standardisation policies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Literal, Sequence

from .transform import ChemicalTransformConfig, IdentityRecord, TransformContractError, transform_smiles


IdentityLevel = Literal["strict", "connectivity", "standardized_parent"]
ScaffoldStatus = Literal["exposed", "unexposed", "indeterminate"]


@dataclass(frozen=True)
class MoleculeInput:
    """One source-row identity input; outcome values are intentionally absent."""

    source_id: str
    row_id: str
    smiles: str
    role: Literal["benchmark", "public"]


@dataclass(frozen=True)
class StandardizedRecord:
    source_id: str
    row_id: str
    role: Literal["benchmark", "public"]
    status: Literal["accepted", "rejected"]
    reason_code: str | None
    identity: IdentityRecord | None

    @property
    def identity_key(self) -> str | None:
        return None if self.identity is None else self.identity.strict_key


def _identity_is_complete(identity: IdentityRecord | None) -> bool:
    """Accepted audit records require every declared identity field."""

    if identity is None:
        return False
    return all(
        str(value or "").strip()
        for value in (
            identity.strict_key,
            identity.connectivity_key,
            identity.standardized_parent_key,
            identity.standardized_parent_smiles,
            identity.transform_hash,
            identity.rdkit_version,
        )
    )


@dataclass(frozen=True)
class IdentityOverlap:
    benchmark_source_id: str
    benchmark_row_id: str
    identity_level: IdentityLevel
    overlap: bool
    matched_public_row_ids: tuple[str, ...]
    transform_hash: str


@dataclass(frozen=True)
class ScaffoldAuditRecord:
    benchmark_source_id: str
    benchmark_row_id: str
    scaffold_status: ScaffoldStatus
    scaffold_smiles: str | None
    matched_public_row_ids: tuple[str, ...]
    reason_code: str | None
    transform_hash: str


@dataclass(frozen=True)
class NearestNeighborRecord:
    benchmark_source_id: str
    benchmark_row_id: str
    nearest_public_row_id: str | None
    nearest_similarity: float | None
    eligible_public_n: int
    method: Literal["bulk", "bruteforce"]
    transform_hash: str


def standardize_records(
    records: Iterable[MoleculeInput], config: ChemicalTransformConfig
) -> tuple[StandardizedRecord, ...]:
    """Run the common transform and retain every rejection with a reason code."""

    outcomes: list[StandardizedRecord] = []
    for record in records:
        try:
            identity = transform_smiles(record.smiles, config)
        except TransformContractError as exc:
            outcomes.append(
                StandardizedRecord(
                    source_id=record.source_id,
                    row_id=record.row_id,
                    role=record.role,
                    status="rejected",
                    reason_code=str(exc),
                    identity=None,
                )
            )
            continue
        if not _identity_is_complete(identity):
            outcomes.append(
                StandardizedRecord(
                    source_id=record.source_id,
                    row_id=record.row_id,
                    role=record.role,
                    status="rejected",
                    reason_code="incomplete_identity_after_standardization",
                    identity=None,
                )
            )
            continue
        outcomes.append(
            StandardizedRecord(
                source_id=record.source_id,
                row_id=record.row_id,
                role=record.role,
                status="accepted",
                reason_code=None,
                identity=identity,
            )
        )
    return tuple(outcomes)


def _accepted(records: Iterable[StandardizedRecord], role: str) -> tuple[StandardizedRecord, ...]:
    return tuple(record for record in records if record.status == "accepted" and record.role == role)


def _identity_key(record: StandardizedRecord, level: IdentityLevel) -> str:
    if record.identity is None:
        raise TransformContractError("identity lookup requested for a rejected record")
    if level == "strict":
        return record.identity.strict_key
    if level == "connectivity":
        return record.identity.connectivity_key
    return record.identity.standardized_parent_key


def audit_identity_overlap(
    records: Iterable[StandardizedRecord], level: IdentityLevel
) -> tuple[IdentityOverlap, ...]:
    """Compare identity keys only; labels and model outputs never enter this path."""

    accepted = tuple(records)
    benchmark = _accepted(accepted, "benchmark")
    public = _accepted(accepted, "public")
    public_index: dict[str, list[str]] = {}
    for row in public:
        public_index.setdefault(_identity_key(row, level), []).append(row.row_id)

    results: list[IdentityOverlap] = []
    for row in benchmark:
        matches = tuple(sorted(public_index.get(_identity_key(row, level), [])))
        assert row.identity is not None
        results.append(
            IdentityOverlap(
                benchmark_source_id=row.source_id,
                benchmark_row_id=row.row_id,
                identity_level=level,
                overlap=bool(matches),
                matched_public_row_ids=matches,
                transform_hash=row.identity.transform_hash,
            )
        )
    return tuple(results)


def identity_equivalent(left: StandardizedRecord, right: StandardizedRecord) -> bool:
    """Exclude all declared identity-equivalent public candidates from NN search."""

    if left.identity is None or right.identity is None:
        return False
    return any(
        (
            left.identity.strict_key == right.identity.strict_key,
            left.identity.connectivity_key == right.identity.connectivity_key,
            left.identity.standardized_parent_key == right.identity.standardized_parent_key,
        )
    )


def _scaffold_smiles(record: StandardizedRecord) -> tuple[str | None, str | None]:
    if record.identity is None:
        return None, "transform_rejected"
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as exc:  # pragma: no cover - chemistry runtime only
        raise TransformContractError("RDKit is required for scaffold extraction") from exc

    molecule = Chem.MolFromSmiles(record.identity.standardized_parent_smiles)
    if molecule is None:
        return None, "standardized_parent_unparseable"
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None, "empty_bemis_murcko_scaffold"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True), None


def audit_scaffold_overlap(records: Iterable[StandardizedRecord]) -> tuple[ScaffoldAuditRecord, ...]:
    """Emit exposed/unexposed/indeterminate scaffold status, never a forced binary."""

    accepted = tuple(records)
    benchmark = _accepted(accepted, "benchmark")
    public = _accepted(accepted, "public")
    public_index: dict[str, list[str]] = {}
    for row in public:
        scaffold, reason = _scaffold_smiles(row)
        if scaffold is not None and reason is None:
            public_index.setdefault(scaffold, []).append(row.row_id)

    result: list[ScaffoldAuditRecord] = []
    for row in benchmark:
        scaffold, reason = _scaffold_smiles(row)
        assert row.identity is not None
        if scaffold is None:
            result.append(
                ScaffoldAuditRecord(
                    benchmark_source_id=row.source_id,
                    benchmark_row_id=row.row_id,
                    scaffold_status="indeterminate",
                    scaffold_smiles=None,
                    matched_public_row_ids=(),
                    reason_code=reason,
                    transform_hash=row.identity.transform_hash,
                )
            )
            continue
        matches = tuple(sorted(public_index.get(scaffold, [])))
        result.append(
            ScaffoldAuditRecord(
                benchmark_source_id=row.source_id,
                benchmark_row_id=row.row_id,
                scaffold_status="exposed" if matches else "unexposed",
                scaffold_smiles=scaffold,
                matched_public_row_ids=matches,
                reason_code=None,
                transform_hash=row.identity.transform_hash,
            )
        )
    return tuple(result)


def _fingerprint(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - chemistry runtime only
        raise TransformContractError("RDKit is required for nearest-neighbor exposure") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise TransformContractError("standardized_parent_unparseable")
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)


def _eligible_public(benchmark: StandardizedRecord, public: Sequence[StandardizedRecord]) -> tuple[StandardizedRecord, ...]:
    return tuple(row for row in public if not identity_equivalent(benchmark, row))


def nearest_neighbor_bruteforce(
    benchmark: StandardizedRecord, public: Sequence[StandardizedRecord]
) -> NearestNeighborRecord:
    """Reference implementation used to validate the vectorised/bulk path."""

    if benchmark.identity is None:
        raise TransformContractError("nearest-neighbor requested for a rejected benchmark record")
    eligible = _eligible_public(benchmark, public)
    if not eligible:
        return NearestNeighborRecord(
            benchmark.source_id, benchmark.row_id, None, None, 0, "bruteforce", benchmark.identity.transform_hash
        )
    try:
        from rdkit import DataStructs
    except Exception as exc:  # pragma: no cover - chemistry runtime only
        raise TransformContractError("RDKit is required for nearest-neighbor exposure") from exc
    query = _fingerprint(benchmark.identity.standardized_parent_smiles)
    scored = [
        (DataStructs.TanimotoSimilarity(query, _fingerprint(row.identity.standardized_parent_smiles)), row.row_id)
        for row in eligible
        if row.identity is not None
    ]
    similarity, row_id = max(scored, key=lambda item: (item[0], item[1]))
    return NearestNeighborRecord(
        benchmark.source_id, benchmark.row_id, row_id, float(similarity), len(eligible), "bruteforce", benchmark.identity.transform_hash
    )


def nearest_neighbor_bulk(
    benchmark: StandardizedRecord, public: Sequence[StandardizedRecord]
) -> NearestNeighborRecord:
    """Vectorised implementation that must agree with :func:`nearest_neighbor_bruteforce`."""

    if benchmark.identity is None:
        raise TransformContractError("nearest-neighbor requested for a rejected benchmark record")
    eligible = _eligible_public(benchmark, public)
    if not eligible:
        return NearestNeighborRecord(
            benchmark.source_id, benchmark.row_id, None, None, 0, "bulk", benchmark.identity.transform_hash
        )
    try:
        from rdkit import DataStructs
    except Exception as exc:  # pragma: no cover - chemistry runtime only
        raise TransformContractError("RDKit is required for nearest-neighbor exposure") from exc
    query = _fingerprint(benchmark.identity.standardized_parent_smiles)
    fingerprints = [_fingerprint(row.identity.standardized_parent_smiles) for row in eligible if row.identity is not None]
    scores = DataStructs.BulkTanimotoSimilarity(query, fingerprints)
    ranked = max(zip(scores, (row.row_id for row in eligible)), key=lambda item: (item[0], item[1]))
    return NearestNeighborRecord(
        benchmark.source_id,
        benchmark.row_id,
        ranked[1],
        float(ranked[0]),
        len(eligible),
        "bulk",
        benchmark.identity.transform_hash,
    )


def stable_json_hash(payload: object) -> str:
    """Hash machine-readable audit records after sorting keys and compacting JSON."""

    encoded = json.dumps(payload, default=lambda item: asdict(item), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
