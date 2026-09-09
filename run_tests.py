#!/usr/bin/env python3
"""
Test Runner Script
Allows running the test suite easily from any terminal or environment.
Automatically prefers .venv or active pytest, falling back to unittest.
"""

import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTEST = PROJECT_ROOT / ".venv" / "bin" / "pytest"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

def run():
    print("=" * 60)
    print(" Running Fake News Detection System Automated Tests")
    print("=" * 60)

    # 1. Check if virtualenv pytest exists
    if VENV_PYTEST.is_file():
        cmd = [str(VENV_PYTEST), "tests/test_prediction.py", "-v"]
        print(f"Executing: {' '.join(cmd)}\n")
        res = subprocess.run(cmd, cwd=PROJECT_ROOT)
        sys.exit(res.returncode)

    # 2. Check if pytest is available in current interpreter
    try:
        import pytest
        print("Executing via active Python environment with pytest...\n")
        exit_code = pytest.main(["tests/test_prediction.py", "-v"])
        sys.exit(exit_code)
    except ImportError:
        pass

    # 3. Fallback to standard library unittest
    print("pytest not found in current environment. Running via standard library unittest...\n")
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    run()
