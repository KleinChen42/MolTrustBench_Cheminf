"""Runtime locks for declared audit artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import yaml


class RuntimeLockError(RuntimeError):
    """Raised when a final-generation runtime differs from its frozen lock."""


@dataclass(frozen=True)
class RuntimeObservation:
    python: str
    implementation: str
    packages: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "python": self.python,
            "implementation": self.implementation,
            "packages": dict(sorted(self.packages.items())),
        }


def installed_rdkit_version() -> str:
    try:
        import rdkit
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise RuntimeLockError("RDKit is required by the final transform lock") from exc
    return str(rdkit.__version__)


def _installed_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeLockError(f"required runtime package is absent: {name}") from exc


def observe_runtime(package_names: tuple[str, ...]) -> RuntimeObservation:
    return RuntimeObservation(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        packages={name: _installed_package_version(name) for name in package_names},
    )


def validate_rdkit_runtime(expected: str, observed: str) -> None:
    if not expected:
        raise RuntimeLockError("expected_rdkit_version is required for final generation")
    if observed != expected:
        raise RuntimeLockError(f"RDKit runtime mismatch: expected {expected}, observed {observed}")


def _load_lock(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw) if Path(path).suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise RuntimeLockError("runtime lock must be a mapping")
    return payload


def require_runtime_lock(config_path: str | Path) -> dict[str, object]:
    """Fail unless the full declared result-bearing runtime is exact.

    A legacy transform YAML is accepted only for the RDKit-only contract. New
    release candidates must use a JSON/YAML runtime lock containing Python and
    every result-bearing package.
    """

    payload = _load_lock(config_path)
    if "packages" not in payload:
        expected = str(payload.get("expected_rdkit_version") or "")
        observed = installed_rdkit_version()
        validate_rdkit_runtime(expected, observed)
        return {"python": platform.python_version(), "implementation": platform.python_implementation(), "packages": {"rdkit": observed}}
    expected_python = str(payload.get("python") or "")
    expected_implementation = str(payload.get("implementation") or "CPython")
    packages = payload.get("packages")
    if not expected_python or not isinstance(packages, dict) or not packages:
        raise RuntimeLockError("full runtime lock requires python and nonempty packages")
    observed = observe_runtime(tuple(str(name) for name in packages))
    errors: list[str] = []
    if observed.python != expected_python:
        errors.append(f"Python expected {expected_python}, observed {observed.python}")
    if observed.implementation != expected_implementation:
        errors.append(f"implementation expected {expected_implementation}, observed {observed.implementation}")
    for name, expected in packages.items():
        actual = observed.packages[str(name)]
        if actual != str(expected):
            errors.append(f"{name} expected {expected}, observed {actual}")
    if errors:
        raise RuntimeLockError("runtime lock mismatch: " + "; ".join(errors))
    return observed.as_dict()
