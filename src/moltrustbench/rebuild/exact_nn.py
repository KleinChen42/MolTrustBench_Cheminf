"""Exact, identity-excluded public nearest-neighbour audit for J2.

This module operates only on locked standardized molecule records.  It does
not accept labels, targets, predictions, or metrics.  The optional CUDA path
uses an exact 2048-bit Morgan/Tanimoto kernel: it is not an approximate vector
index.  Every benchmark query masks strict, connectivity, *and* standardized
parent identity matches before the score is computed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
FINGERPRINT_WORDS = FINGERPRINT_BITS // 64
IDENTITY_COLUMNS = ("strict_key", "connectivity_key", "standardized_parent_key")
REQUIRED_COLUMNS = (
    "source_id",
    "row_id",
    "role",
    "status",
    "strict_key",
    "connectivity_key",
    "standardized_parent_key",
    "standardized_parent_smiles",
    "transform_hash",
    "rdkit_version",
)


@dataclass(frozen=True)
class NNInputRow:
    """An accepted, outcome-free transformed input row."""

    source_id: str
    row_id: str
    role: str
    strict_key: str
    connectivity_key: str
    standardized_parent_key: str
    standardized_parent_smiles: str
    transform_hash: str
    rdkit_version: str


@dataclass(frozen=True)
class NNResult:
    """One exact, non-identical public nearest-neighbour result."""

    benchmark_source_id: str
    benchmark_row_id: str
    nearest_public_row_id: str | None
    nearest_similarity: float | None
    eligible_public_n: int
    excluded_identity_equivalent_n: int
    backend: str
    transform_hash: str


def sha256_file(path: Path) -> str:
    """Return the content digest used by NN manifests."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: object) -> str:
    """Hash a JSON-serialisable parameter or metadata record."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_accepted_rows(path: Path, expected_role: str) -> list[NNInputRow]:
    """Load accepted transformed records and reject mixed role inputs."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
        rows = [
            NNInputRow(
                **{
                    field: row[field]
                    for field in REQUIRED_COLUMNS
                    if field != "status"
                }
            )
            for row in reader
            if row["status"] == "accepted"
        ]
    roles = {row.role for row in rows}
    if roles != {expected_role}:
        raise ValueError(f"{path} must contain accepted role {expected_role!r}, found {sorted(roles)}")
    if not rows:
        raise ValueError(f"{path} has no accepted {expected_role} rows")
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError(f"{path} has duplicate accepted row_id values")
    return rows


def assert_common_transform(benchmark: Sequence[NNInputRow], public: Sequence[NNInputRow]) -> tuple[str, str]:
    """Refuse a comparison unless both inputs share one transform/runtime."""

    benchmark_hashes = {row.transform_hash for row in benchmark}
    public_hashes = {row.transform_hash for row in public}
    benchmark_versions = {row.rdkit_version for row in benchmark}
    public_versions = {row.rdkit_version for row in public}
    if len(benchmark_hashes) != 1 or benchmark_hashes != public_hashes:
        raise ValueError("benchmark and public inputs do not share exactly one transform hash")
    if len(benchmark_versions) != 1 or benchmark_versions != public_versions:
        raise ValueError("benchmark and public inputs do not share exactly one RDKit version")
    return next(iter(benchmark_hashes)), next(iter(benchmark_versions))


def fingerprint_spec() -> dict[str, object]:
    """Return the frozen public-NN fingerprint contract."""

    return {
        "implementation": "RDKit Morgan bit vector",
        "radius": FINGERPRINT_RADIUS,
        "n_bits": FINGERPRINT_BITS,
        "metric": "Tanimoto",
        "packed_word_dtype": "uint64_little_endian",
    }


def fingerprint_words(smiles: str) -> np.ndarray:
    """Create one packed Morgan fingerprint under the J2 fixed contract."""

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - chemistry runtime only
        raise RuntimeError("RDKit is required for J2 nearest-neighbour fingerprinting") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("accepted standardized-parent SMILES is not parseable")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
    )
    bits = np.zeros((FINGERPRINT_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    return np.ascontiguousarray(np.packbits(bits, bitorder="little").view("<u8"))


def build_fingerprint_matrix(rows: Sequence[NNInputRow]) -> np.ndarray:
    """Return an N x 32 packed fingerprint matrix in the supplied row order."""

    matrix = np.empty((len(rows), FINGERPRINT_WORDS), dtype=np.uint64)
    for index, row in enumerate(rows):
        matrix[index] = fingerprint_words(row.standardized_parent_smiles)
    return matrix


def write_fingerprint_cache(rows: Sequence[NNInputRow], matrix: np.ndarray, prefix: Path) -> dict[str, object]:
    """Persist reusable, hash-bound fingerprint material without outcomes."""

    if matrix.shape != (len(rows), FINGERPRINT_WORDS):
        raise ValueError("fingerprint matrix shape does not match the supplied rows")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = prefix.with_name(prefix.name + "_fingerprints.npy")
    metadata_path = prefix.with_name(prefix.name + "_fingerprint_rows.csv")
    manifest_path = prefix.with_name(prefix.name + "_fingerprint_manifest.json")
    np.save(matrix_path, np.ascontiguousarray(matrix, dtype=np.uint64), allow_pickle=False)
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_id",
                "row_id",
                "role",
                "strict_key",
                "connectivity_key",
                "standardized_parent_key",
                "transform_hash",
                "rdkit_version",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_id": row.source_id,
                    "row_id": row.row_id,
                    "role": row.role,
                    "strict_key": row.strict_key,
                    "connectivity_key": row.connectivity_key,
                    "standardized_parent_key": row.standardized_parent_key,
                    "transform_hash": row.transform_hash,
                    "rdkit_version": row.rdkit_version,
                }
            )
    payload: dict[str, object] = {
        "fingerprint_spec": fingerprint_spec(),
        "fingerprint_spec_hash": stable_json_hash(fingerprint_spec()),
        "row_n": len(rows),
        "matrix_shape": list(matrix.shape),
        "matrix_path": matrix_path.as_posix(),
        "matrix_sha256": sha256_file(matrix_path),
        "metadata_path": metadata_path.as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
        "transform_hash": rows[0].transform_hash if rows else None,
        "rdkit_version": rows[0].rdkit_version if rows else None,
        "outcome_values_accessed": False,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def public_rows_for_gpu(public: Sequence[NNInputRow]) -> list[NNInputRow]:
    """Use a stable descending row-id order so equal scores have one tie rule."""

    return sorted(public, key=lambda row: row.row_id, reverse=True)


def build_identity_exclusion_index(public: Sequence[NNInputRow]) -> dict[str, dict[str, np.ndarray]]:
    """Index every declared identity key for exact exclusion before scoring."""

    indexes: dict[str, dict[str, list[int]]] = {column: {} for column in IDENTITY_COLUMNS}
    for position, row in enumerate(public):
        for column in IDENTITY_COLUMNS:
            key = getattr(row, column)
            indexes[column].setdefault(key, []).append(position)
    return {
        column: {key: np.asarray(value, dtype=np.int64) for key, value in mapping.items()}
        for column, mapping in indexes.items()
    }


def excluded_public_positions(
    query: NNInputRow, identity_index: Mapping[str, Mapping[str, np.ndarray]]
) -> np.ndarray:
    """Return all public candidates equivalent at strict, connectivity or parent level."""

    matches: list[np.ndarray] = []
    for column in IDENTITY_COLUMNS:
        positions = identity_index[column].get(getattr(query, column))
        if positions is not None:
            matches.append(positions)
    if not matches:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(matches)).astype(np.int64, copy=False)


_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


def _tanimoto_scores_cpu(query_words: np.ndarray, public_words: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """Reference exact packed-bit Tanimoto implementation for fixture checks."""

    inter = _POPCOUNT[np.bitwise_and(public_words, query_words).view(np.uint8)].sum(axis=1, dtype=np.uint32)
    union = _POPCOUNT[np.bitwise_or(public_words, query_words).view(np.uint8)].sum(axis=1, dtype=np.uint32)
    scores = inter.astype(np.float64) / union.astype(np.float64)
    scores[excluded] = -1.0
    return scores


def exact_nn_cpu(query_words: np.ndarray, public_words: np.ndarray, excluded: np.ndarray) -> tuple[int | None, float | None]:
    """Reference full scan with the same deterministic tie convention as CUDA."""

    if len(excluded) >= len(public_words):
        return None, None
    scores = _tanimoto_scores_cpu(query_words, public_words, excluded)
    position = int(np.argmax(scores))
    score = float(scores[position])
    if score < 0:
        return None, None
    return position, score


_CUDA_SOURCE = r'''
extern "C" __global__
void exact_tanimoto_2048(
    const unsigned long long* query,
    const unsigned long long* public_words,
    const unsigned char* eligible,
    const long long row_n,
    double* scores) {
  const long long row = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (row >= row_n) return;
  if (eligible[row] == 0) { scores[row] = -1.0; return; }
  unsigned int intersection = 0;
  unsigned int uni = 0;
  #pragma unroll
  for (int word = 0; word < 32; ++word) {
    const unsigned long long candidate = public_words[row * 32 + word];
    intersection += __popcll(query[word] & candidate);
    uni += __popcll(query[word] | candidate);
  }
  scores[row] = uni == 0 ? -1.0 : static_cast<double>(intersection) / static_cast<double>(uni);
}
'''


class ExactGpuTanimotoIndex:
    """One-device, exact public fingerprint index with per-query identity masks."""

    def __init__(self, public_words: np.ndarray, device_id: int = 0) -> None:
        if public_words.ndim != 2 or public_words.shape[1] != FINGERPRINT_WORDS:
            raise ValueError("GPU NN requires an N x 32 uint64 public fingerprint matrix")
        try:
            import cupy as cp
        except Exception as exc:  # pragma: no cover - optional CUDA runtime
            raise RuntimeError("CuPy CUDA runtime is required for the GPU NN backend") from exc
        self._cp = cp
        self.device_id = int(device_id)
        with cp.cuda.Device(self.device_id):
            self._public = cp.asarray(np.ascontiguousarray(public_words, dtype=np.uint64))
            self._kernel = cp.RawKernel(_CUDA_SOURCE, "exact_tanimoto_2048")
            self._peak_pool_bytes = int(cp.get_default_memory_pool().total_bytes())

    @property
    def public_n(self) -> int:
        return int(self._public.shape[0])

    @property
    def peak_memory_pool_bytes(self) -> int:
        """Upper bound recorded by CuPy's allocator, not an OS-wide memory claim."""

        return self._peak_pool_bytes

    def runtime_metadata(self) -> dict[str, object]:
        """Return device/runtime details for the run manifest."""

        cp = self._cp
        with cp.cuda.Device(self.device_id):
            properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
            name = properties["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            return {
                "cupy_version": cp.__version__,
                "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "device_id": self.device_id,
                "device_name": str(name),
                "memory_pool_peak_bytes": self.peak_memory_pool_bytes,
            }

    def query(self, query_words: np.ndarray, excluded: np.ndarray) -> tuple[int | None, float | None]:
        """Return the exact maximum after masking all declared identity equivalents."""

        if len(excluded) >= self.public_n:
            return None, None
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            query_gpu = cp.asarray(np.ascontiguousarray(query_words, dtype=np.uint64))
            eligible = cp.ones((self.public_n,), dtype=cp.uint8)
            if len(excluded):
                eligible[cp.asarray(excluded, dtype=cp.int64)] = 0
            scores = cp.empty((self.public_n,), dtype=cp.float64)
            block = 256
            grid = ((self.public_n + block - 1) // block,)
            self._kernel(grid, (block,), (query_gpu, self._public, eligible, self.public_n, scores))
            position = int(cp.argmax(scores).get())
            score = float(scores[position].get())
            self._peak_pool_bytes = max(self._peak_pool_bytes, int(cp.get_default_memory_pool().total_bytes()))
        return (None, None) if score < 0 else (position, score)


def fixture_gpu_equivalence(device_id: int = 0) -> dict[str, object]:
    """Verify CUDA output equals the CPU exact reference on identity-mask fixtures."""

    fixture_smiles = ("c1ccccc1", "CCO", "CC(=O)O", "Oc1ccccc1")
    public = np.stack([fingerprint_words(smiles) for smiles in fixture_smiles])
    queries = (fingerprint_words("c1ccccc1"), fingerprint_words("Oc1ccccc1"))
    masks = (np.asarray([0], dtype=np.int64), np.asarray([], dtype=np.int64))
    gpu = ExactGpuTanimotoIndex(public, device_id=device_id)
    comparisons: list[dict[str, object]] = []
    for query, excluded in zip(queries, masks, strict=True):
        cpu_position, cpu_score = exact_nn_cpu(query, public, excluded)
        gpu_position, gpu_score = gpu.query(query, excluded)
        if cpu_position != gpu_position or gpu_score is None or cpu_score is None or abs(cpu_score - gpu_score) > 1e-12:
            raise AssertionError("exact CUDA NN result disagrees with the CPU fixture reference")
        comparisons.append(
            {
                "excluded_positions": excluded.tolist(),
                "cpu_position": cpu_position,
                "gpu_position": gpu_position,
                "similarity": cpu_score,
            }
        )
    return {
        "fixture_name": "exact_tanimoto_identity_exclusion_v1",
        "fingerprint_spec": fingerprint_spec(),
        "device_id": device_id,
        "comparison_n": len(comparisons),
        "comparisons": comparisons,
        "outcome_values_accessed": False,
    }


def make_result_rows(
    benchmark: Sequence[NNInputRow],
    benchmark_words: np.ndarray,
    public: Sequence[NNInputRow],
    public_words: np.ndarray,
    backend: ExactGpuTanimotoIndex | None = None,
) -> list[NNResult]:
    """Compute full exact public NN records, using GPU when supplied."""

    if benchmark_words.shape != (len(benchmark), FINGERPRINT_WORDS):
        raise ValueError("benchmark fingerprint matrix does not match benchmark rows")
    if public_words.shape != (len(public), FINGERPRINT_WORDS):
        raise ValueError("public fingerprint matrix does not match public rows")
    identity_index = build_identity_exclusion_index(public)
    results: list[NNResult] = []
    for position, query in enumerate(benchmark):
        excluded = excluded_public_positions(query, identity_index)
        if backend is None:
            nearest_position, similarity = exact_nn_cpu(benchmark_words[position], public_words, excluded)
            backend_name = "cpu_exact_bitpopcount"
        else:
            nearest_position, similarity = backend.query(benchmark_words[position], excluded)
            backend_name = "gpu_exact_bitpopcount"
        results.append(
            NNResult(
                benchmark_source_id=query.source_id,
                benchmark_row_id=query.row_id,
                nearest_public_row_id=None if nearest_position is None else public[nearest_position].row_id,
                nearest_similarity=similarity,
                eligible_public_n=len(public) - len(excluded),
                excluded_identity_equivalent_n=len(excluded),
                backend=backend_name,
                transform_hash=query.transform_hash,
            )
        )
    return results
