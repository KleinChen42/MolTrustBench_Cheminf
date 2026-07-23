from pathlib import Path
import subprocess
import sys


def test_public_release_verifier_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "scripts/verify_public_release.py"], cwd=root, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"status": "pass"' in completed.stdout
