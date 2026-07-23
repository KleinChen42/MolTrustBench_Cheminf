"""Shared chemical-transformation contract for public-source overlap audits.

The configuration hash, rather than a filename or environment default, binds
benchmark and public-library transformations. Production transformation refuses
to fall back to a non-RDKit parser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class TransformContractError(ValueError):
    """Raised for an incomplete or incompatible chemical transformation."""


@dataclass(frozen=True)
class ChemicalTransformConfig:
    policy_id: str
    policy_version: str
    rdkit_version_constraint: str
    strict_isomeric_smiles: bool
    standardized_parent_isomeric_smiles: bool
    remove_fragments: bool
    neutralize_charge: bool
    canonicalize_tautomer: bool
    tautomer_max_tautomers: int
    tautomer_max_transforms: int
    tautomer_canonicalization_failure_policy: str
    require_reparseable_standardized_parent: bool
    retain_isotopes: bool
    metal_policy: str
    invalid_smiles_policy: str
    duplicate_collapse_policy: str
    fixture_fallback_only: bool

    @property
    def transform_hash(self) -> str:
        serialized = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ChemicalTransformConfig":
        fields = {
            "policy_id": payload.get("policy_id"),
            "policy_version": payload.get("policy_version"),
            "rdkit_version_constraint": payload.get("rdkit_version_constraint"),
            "strict_isomeric_smiles": payload.get("strict_isomeric_smiles"),
            "standardized_parent_isomeric_smiles": payload.get("standardized_parent_isomeric_smiles"),
            "remove_fragments": payload.get("remove_fragments"),
            "neutralize_charge": payload.get("neutralize_charge"),
            "canonicalize_tautomer": payload.get("canonicalize_tautomer"),
            "tautomer_max_tautomers": payload.get("tautomer_max_tautomers"),
            "tautomer_max_transforms": payload.get("tautomer_max_transforms"),
            "tautomer_canonicalization_failure_policy": payload.get("tautomer_canonicalization_failure_policy"),
            "require_reparseable_standardized_parent": payload.get("require_reparseable_standardized_parent"),
            "retain_isotopes": payload.get("retain_isotopes"),
            "metal_policy": payload.get("metal_policy"),
            "invalid_smiles_policy": payload.get("invalid_smiles_policy"),
            "duplicate_collapse_policy": payload.get("duplicate_collapse_policy"),
            "fixture_fallback_only": payload.get("fixture_fallback_only"),
        }
        missing = [name for name, value in fields.items() if value is None or value == ""]
        if missing:
            raise TransformContractError(f"chemical transform config missing: {', '.join(missing)}")
        if fields["invalid_smiles_policy"] != "reject_with_reason_code":
            raise TransformContractError("invalid SMILES must be rejected with a reason code")
        if fields["duplicate_collapse_policy"] != "one_record_per_declared_identity_key":
            raise TransformContractError("duplicate collapse must be declared-key based")
        if fields["metal_policy"] not in {"retain_and_flag", "reject_with_reason_code"}:
            raise TransformContractError("unsupported metal policy")
        for field_name in ("tautomer_max_tautomers", "tautomer_max_transforms"):
            value = fields[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise TransformContractError(f"{field_name} must be a positive integer")
        if fields["tautomer_canonicalization_failure_policy"] != "reject_with_reason_code":
            raise TransformContractError("tautomer canonicalization failures must be rejected with a reason code")
        if fields["require_reparseable_standardized_parent"] is not True:
            raise TransformContractError("standardized-parent SMILES must be reparsed before acceptance")
        return cls(**fields)  # type: ignore[arg-type]


@dataclass(frozen=True)
class IdentityRecord:
    raw_smiles: str
    strict_key: str
    connectivity_key: str
    standardized_parent_key: str
    standardized_parent_smiles: str
    transform_hash: str
    rdkit_version: str
    isotope_present: bool = False
    metal_present: bool = False


def _require_identity_value(value: object, reason_code: str) -> str:
    """Return a nonempty declared identity field or reject the row.

    RDKit can emit an empty InChIKey for a chemically parsed record without
    raising an exception. Such a record is not usable for a declared
    identity-level audit and must never enter the accepted population.
    """

    rendered = str(value or "").strip()
    if not rendered:
        raise TransformContractError(reason_code)
    return rendered


def load_transform_config(path: str | Path) -> ChemicalTransformConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ChemicalTransformConfig.from_mapping(payload)


def _canonicalize_tautomer(normalized: Any, config: ChemicalTransformConfig, rdkit_standardize: Any) -> Any:
    """Canonicalize one molecule or produce a row-level rejection reason."""

    try:
        enumerator = rdkit_standardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(config.tautomer_max_tautomers)
        enumerator.SetMaxTransforms(config.tautomer_max_transforms)
        return enumerator.Canonicalize(normalized)
    except Exception as exc:  # RDKit can raise an invariant violation for individual molecules.
        raise TransformContractError("tautomer_canonicalization_failure") from exc


def _standardized_parent(molecule: Any, config: ChemicalTransformConfig, rdkit_standardize: Any) -> Any:
    """Apply declared parent-standardization stages with row-level failures."""

    try:
        normalized = rdkit_standardize.Cleanup(molecule)
    except Exception as exc:
        raise TransformContractError("cleanup_failure") from exc
    if config.remove_fragments:
        try:
            normalized = rdkit_standardize.FragmentParent(normalized)
        except Exception as exc:
            raise TransformContractError("fragment_parent_failure") from exc
    if config.neutralize_charge:
        try:
            normalized = rdkit_standardize.Uncharger().uncharge(normalized)
        except Exception as exc:
            raise TransformContractError("uncharging_failure") from exc
    if config.canonicalize_tautomer:
        normalized = _canonicalize_tautomer(normalized, config, rdkit_standardize)
    return normalized


def _contains_metal(molecule: Any) -> bool:
    """Return whether a molecule contains a retained metal atom."""
    non_metals = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
    return any(atom.GetAtomicNum() not in non_metals for atom in molecule.GetAtoms())

def _contains_isotope(molecule: Any) -> bool:
    return any(atom.GetIsotope() != 0 for atom in molecule.GetAtoms())

def _apply_isotope_policy(molecule: Any, config: ChemicalTransformConfig, chem: Any) -> Any:
    """Apply the declared isotope policy before result-bearing identities."""
    if config.retain_isotopes:
        return molecule
    policy_molecule = chem.Mol(molecule)
    for atom in policy_molecule.GetAtoms():
        atom.SetIsotope(0)
    return policy_molecule

def transform_smiles(raw_smiles: str, config: ChemicalTransformConfig) -> IdentityRecord:
    """Create strict, connectivity, and standardized-parent identities.

    RDKit is intentionally imported lazily: a fixture-only fallback is not a
    legal production path for a declared audit result.
    """

    try:
        import rdkit
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception as exc:  # pragma: no cover - depends on chemistry runtime
        raise TransformContractError("RDKit is required for production chemical transformation") from exc

    molecule = Chem.MolFromSmiles(str(raw_smiles))
    if molecule is None:
        raise TransformContractError("invalid_smiles")
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise TransformContractError("sanitize_failure") from exc

    isotope_present = _contains_isotope(molecule)
    metal_present = _contains_metal(molecule)
    if metal_present and config.metal_policy == "reject_with_reason_code":
        raise TransformContractError("metal_rejected_by_policy")
    molecule = _apply_isotope_policy(molecule, config, Chem)

    strict_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=config.strict_isomeric_smiles)
    strict_key = _require_identity_value(
        Chem.MolToInchiKey(molecule), "strict_identity_missing"
    )
    connectivity_key = strict_key.split("-", 1)[0]
    _require_identity_value(connectivity_key, "connectivity_identity_missing")

    normalized = _standardized_parent(molecule, config, rdMolStandardize)
    try:
        parent_smiles = Chem.MolToSmiles(
            normalized,
            canonical=True,
            isomericSmiles=config.standardized_parent_isomeric_smiles,
        )
        parent_key = _require_identity_value(
            Chem.MolToInchiKey(normalized), "standardized_parent_identity_missing"
        )
    except TransformContractError:
        raise
    except Exception as exc:
        raise TransformContractError("parent_identity_failure") from exc
    try:
        reparsed_parent = Chem.MolFromSmiles(parent_smiles)
        if reparsed_parent is None:
            raise ValueError("canonical parent SMILES did not reparse")
        Chem.SanitizeMol(reparsed_parent)
    except Exception as exc:
        raise TransformContractError("standardized_parent_unparseable") from exc
    return IdentityRecord(
        raw_smiles=str(raw_smiles),
        strict_key=strict_key,
        connectivity_key=connectivity_key,
        standardized_parent_key=parent_key,
        standardized_parent_smiles=_require_identity_value(
            parent_smiles, "standardized_parent_smiles_missing"
        ),
        transform_hash=config.transform_hash,
        rdkit_version=str(rdkit.__version__),
        isotope_present=isotope_present,
        metal_present=metal_present,
    )
