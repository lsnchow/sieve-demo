"""Convenience wrapper for the dataset-curation CLI."""

from __future__ import annotations

def main() -> None:
    from backend.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
