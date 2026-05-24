#!/usr/bin/env bash
# Run after portfolio training completes (evaluate + figures).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source venv/bin/activate
python evaluate.py
python scripts/generate_report_figures.py
echo "Done. Update README Results from reports/comparison_results.csv"
