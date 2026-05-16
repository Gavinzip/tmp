#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

python3 scripts/run_term_project2_recommended.py
python3 scripts/build_report_assets.py
python3 scripts/build_final_report.py
