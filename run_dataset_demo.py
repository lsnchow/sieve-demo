"""Convenience wrapper for the dataset-curation CLI."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "original_code"))


def main() -> None:
    cli_module = importlib.import_module("pov_qa 2.cli")
    cli_module.main()


if __name__ == "__main__":
    main()
