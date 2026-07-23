# Reproducing the compact public package

1. Clone the repository and retain the distributed directory structure.
2. Run `python scripts/verify_public_release.py` from the repository root.
3. Inspect `data/derived/PUBLIC_DATA_MANIFEST.json` and the printed JSON receipt.
4. For implementation work with the reusable RDKit modules, create `environment.yml` and set `PYTHONPATH=src`.

The verifier checks byte-level hashes and the declared CHEMBL32 relation-count, release-bracketing, fingerprint-grid, outcome-free, and shared-transform invariants. It validates the compact outputs supplied here; it does not recreate the omitted source archives.
