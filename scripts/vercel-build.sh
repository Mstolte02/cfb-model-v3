#!/usr/bin/env bash
# Vercel build: test the browser code, then build dist/.
# This lives in a script because Vercel rejects a buildCommand longer than 256 characters.
set -euo pipefail
node --test \
  tests/test_resume_scores.cjs \
  tests/test_ranking_history.cjs \
  tests/test_most_deserving.cjs \
  tests/test_marks_playoff.cjs \
  tests/test_power_xray.cjs \
  tests/test_war_research.cjs \
  tests/test_playoff_paths.cjs \
  tests/test_scenario_lab.cjs
python3 scripts/publish.py
