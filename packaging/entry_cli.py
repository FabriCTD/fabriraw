"""PyInstaller entry point: ctd-raw.exe (console CLI)."""

import sys

from fraw.cli import main

if __name__ == "__main__":
    sys.exit(main())
