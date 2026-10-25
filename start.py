"""Run the tracker with the project venv.

Usage: python start.py [run options such as --source 1 --no-show]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    if os.name == "nt":
        python = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        print("The .venv folder was not found. Run this first:  python install.py", file=sys.stderr)
        return 1
    proc = subprocess.Popen([str(python), "-m", "utra_tracker", "run", *sys.argv[1:]], cwd=ROOT)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # The tracker also receives Ctrl+C and shuts down on its own, so wait for it.
            continue


if __name__ == "__main__":
    sys.exit(main())
