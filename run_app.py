#!/usr/bin/env python3
"""
Streamlit Application Launcher
Ensures the application runs within the properly configured virtual environment (.venv),
avoiding C-extension ABI conflicts and missing package issues.
"""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_STREAMLIT = PROJECT_ROOT / ".venv" / "bin" / "streamlit"
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"

def main():
    print("=" * 60)
    print(" Starting AI Fake News Detector Streamlit Application")
    print("=" * 60)

    if VENV_STREAMLIT.is_file():
        cmd = [str(VENV_STREAMLIT), "run", str(APP_PATH)]
        print(f"Launching using project virtual environment:\n{' '.join(cmd)}\n")
        try:
            res = subprocess.run(cmd, cwd=PROJECT_ROOT)
            sys.exit(res.returncode)
        except KeyboardInterrupt:
            print("\nShutting down Streamlit application.")
            sys.exit(0)
    else:
        # Fallback to system/active environment streamlit
        print("Note: .venv not found. Falling back to active environment 'streamlit run'...\n")
        cmd = ["streamlit", "run", str(APP_PATH)]
        try:
            res = subprocess.run(cmd, cwd=PROJECT_ROOT)
            sys.exit(res.returncode)
        except KeyboardInterrupt:
            print("\nShutting down Streamlit application.")
            sys.exit(0)

if __name__ == "__main__":
    main()
