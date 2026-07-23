# MolTrustBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21502338.svg)](https://doi.org/10.5281/zenodo.21502338)

MolTrustBench is a compact, reproducible implementation package for **release-aware public-source overlap auditing** in molecular machine learning benchmarks. It provides outcome-free chemical-relation summaries, declared sensitivity analyses, figures, and reusable implementation components for molecular standardization, release indexing, exact relation checks, and identity-excluded nearest-neighbor comparisons.

## What this package supports

For the included CHEMBL32 case study, the package verifies compact public result tables covering:

- strict identity, connectivity identity, and standardized-parent overlap;
- scaffold exposure with exposed, unexposed, and not-applicable denominators;
- identity-excluded nearest public-neighbor and benchmark-training-neighbor summaries;
- declared CHEMBL30/CHEMBL32/CHEMBL33 anchor bracketing; and
- a predeclared Morgan fingerprint sensitivity grid.

## Interpretation boundary

The reported quantities are **public-source observability under declared release and chemistry contracts**. They do not establish model training-corpus membership, memorization, leakage, or causal score effects.

## Quick start

```bash
python scripts/verify_public_release.py
```

The command checks all distributed CSV and PDF SHA-256 values, verifies key table invariants, and prints a machine-readable pass/fail receipt. It uses only the Python standard library.

For the reusable RDKit components, create the pinned environment:

```bash
conda env create -f environment.yml
conda activate moltrustbench-cheminf
export PYTHONPATH=src
```

## Repository layout

- `src/`: reusable outcome-free standardization, identity, release-index, and nearest-neighbor utilities.
- `data/derived/`: compact result summaries used in the included case study.
- `figures/`: vector figures corresponding to the public tables and workflow.
- `scripts/verify_public_release.py`: hash and invariant verification for the distributed compact package.
- `docs/`: scope, provenance, and reproduction guidance.

## Scope of the public package

This repository intentionally excludes raw public-release archives, benchmark outcome values, model predictions, training artifacts, and controlled source materials. See [docs/PUBLIC_SCOPE.md](docs/PUBLIC_SCOPE.md) for the evidence boundary and [docs/REPRODUCE.md](docs/REPRODUCE.md) for reproduction instructions.

## Citation

The versioned v1.0 archive is available at [Zenodo DOI: 10.5281/zenodo.21502338](https://doi.org/10.5281/zenodo.21502338). Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

This software is released under the [MIT License](LICENSE).