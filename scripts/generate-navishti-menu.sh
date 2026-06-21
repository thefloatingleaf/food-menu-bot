#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
set +e
python3 generate_navishti_menu.py
status=$?
set -e
if [[ "${status}" -ne 0 ]]; then
  if [[ "${status}" -ne 42 ]]; then
    exit "${status}"
  fi
  echo "Standard menu row missing; generating standard daily menu before Navishti." >&2
  python3 generate_menu.py
  python3 scripts/verify_daily_menu_freshness.py
  python3 generate_navishti_menu.py
fi
python3 scripts/verify_navishti_menu_freshness.py
