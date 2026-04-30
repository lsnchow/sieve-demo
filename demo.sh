#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

"$PYTHON_BIN" "$ROOT/run_dataset_demo.py" show-config
"$PYTHON_BIN" "$ROOT/run_dataset_demo.py" prepare-dataset --input-dir "$ROOT/raw_input" --expected-count 10
"$PYTHON_BIN" "$ROOT/run_dataset_demo.py" run-all -i "$ROOT/raw_input" -n 10 --snapshot-output-dir "$ROOT/exports"
"$PYTHON_BIN" "$ROOT/run_dataset_demo.py" export -o "$ROOT/exports" -n 10
